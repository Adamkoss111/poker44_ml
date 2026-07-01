"""
train_stacked.py

Stacked ensemble dla Poker44 — 5 base modeli, predykcje USREDNIANE (bez meta).

Dla KAŻDEJ kombinacji grida (fit_on x pseudo):
  Dla KAŻDEGO z 5 typów modeli (xgb, lgb, cat, rf, et) OSOBNO:
    1. szuka najlepszego progu KS (na domyślnych HP, wg sel_score),
    2. lekki HPO przy wybranym KS (ocena na test),
    3. trenuje finalny model (na train, lub train+test gdy fit_on tego wymaga).
  Stack = ZWYKŁA ŚREDNIA predykcji 5 modeli.

Reward 1:1 z walidatorem (poker44/score/scoring.py):
  reward = 0.75*AP + 0.25*bot_recall ; bot_recall = recall przy FPR<=0.05 (ranking).
  Metryka jest RANKINGOWA -> scale_pos_weight wycięte (nie zmienia rankingu).

Każdy base model może mieć INNY zestaw cech (własny najlepszy KS) — to celowe,
żeby modele się różniły. Uśrednianie odbywa się na poziomie predykcji, więc
różne wejścia nie przeszkadzają.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from itertools import product
from math import prod as _prod
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import xgboost as xgb
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


# ============ GPU detection (XGB) ============
def _detect_cuda():
    override = str(os.getenv("POKER44_XGB_DEVICE", "")).strip().lower()
    if override in {"cpu", "cuda"}:
        return override == "cuda"
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import shutil, subprocess
        if shutil.which("nvidia-smi"):
            r = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=5)
            if r.returncode == 0:
                return True
    except Exception:
        pass
    return False

_CUDA = _detect_cuda()


# ============ KONFIGURACJA ============
DEFAULT_SPLIT_DATE = "2026-06-23"
DEFAULT_PSEUDO_LOW, DEFAULT_PSEUDO_HIGH = 0.10, 0.90
DEFAULT_OUT_DIR = "stacked_models"

DEFAULT_GRID_KS     = [round(x, 2) for x in np.arange(0.30, 0.70, 0.02)]
DEFAULT_GRID_FIT_ON = ["train", "train+test"]
DEFAULT_GRID_PSEUDO = [False, True]

# 5 typów base modeli w stacku
MODEL_TYPES = ["xgb", "lgb", "cat", "rf", "et"]

# Lekki HPO per typ modelu (małe siatki — kilka prób każda)
DEFAULT_HPO_N_ITER = 8
PARAM_GRIDS = {
    "xgb": {"max_depth": [3, 4, 5], "n_estimators": [40, 80, 150],
            "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0],
            "reg_lambda": [1.0, 5.0]},
    "lgb": {"max_depth": [3, 4, 6], "n_estimators": [40, 80, 150],
            "learning_rate": [0.05, 0.1], "num_leaves": [15, 31],
            "reg_lambda": [1.0, 5.0]},
    "cat": {"depth": [3, 4, 6], "iterations": [80, 150],
            "learning_rate": [0.05, 0.1], "l2_leaf_reg": [1.0, 5.0]},
    "rf":  {"n_estimators": [200, 400, 600, 800], "max_depth": [6, 10, 16, 24, None],
            "min_samples_leaf": [1, 5], "max_features": ["sqrt", 0.5]},
    "et":  {"n_estimators": [200, 400, 600, 800], "max_depth": [6, 10, 16, 24, None],
            "min_samples_leaf": [1, 5], "max_features": ["sqrt", 0.5]},
}
DEFAULT_PARAMS = {
    "xgb": {"max_depth": 3, "n_estimators": 40},
    "lgb": {"max_depth": 3, "n_estimators": 40},
    "cat": {"depth": 3, "iterations": 80},
    "rf":  {"n_estimators": 300, "max_depth": 10},
    "et":  {"n_estimators": 300, "max_depth": 10},
}
# ======================================


def _make_model(mtype, params, spw=1.0):
    """Buduje klasyfikator danego typu z parametrami.
    spw (scale_pos_weight) mapowany per typ:
      - xgb/lgb: scale_pos_weight=spw
      - cat:     scale_pos_weight=spw
      - rf/et:   class_weight={0:1.0, 1:spw} (nie mają scale_pos_weight)
    spw=1.0 => brak ważenia (neutralny)."""
    p = {**DEFAULT_PARAMS[mtype], **(params or {})}
    spw = float(spw)
    if mtype == "xgb":
        if _CUDA:
            # XGBoost 2.0+ używa device="cuda"; starsze tree_method="gpu_hist"
            _xgb_major = int(xgb.__version__.split(".")[0])
            if _xgb_major >= 2:
                dev = {"device": "cuda", "tree_method": "hist"}
            else:
                dev = {"tree_method": "gpu_hist"}
        else:
            dev = {}
        return xgb.XGBClassifier(random_state=0, eval_metric="logloss",
                                 scale_pos_weight=spw, **dev, **p)
    if mtype == "lgb":
        if lgb is None:
            raise RuntimeError("lightgbm nie zainstalowany")
        return lgb.LGBMClassifier(random_state=0, verbose=-1,
                                  scale_pos_weight=spw, **p)
    if mtype == "cat":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost nie zainstalowany")
        task = "GPU" if _CUDA else "CPU"
        return CatBoostClassifier(random_state=0, verbose=False, task_type=task,
                                  scale_pos_weight=spw, **p)
    if mtype == "rf":
        cw = None if spw == 1.0 else {0: 1.0, 1: spw}
        return RandomForestClassifier(random_state=0, n_jobs=-1, class_weight=cw, **p)
    if mtype == "et":
        cw = None if spw == 1.0 else {0: 1.0, 1: spw}
        return ExtraTreesClassifier(random_state=0, n_jobs=-1, class_weight=cw, **p)
    raise ValueError(mtype)


# ============ REWARD (1:1 walidator) ============
def recall_at_fpr(y_score, y_true, max_fpr=0.05):
    labels = np.asarray(y_true, dtype=int); scores = np.asarray(y_score, dtype=float)
    pos = int(np.sum(labels == 1)); neg = int(np.sum(labels == 0))
    if pos <= 0 or neg <= 0 or scores.size == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort"); sl = labels[order]
    tp = np.cumsum(sl == 1); fp = np.cumsum(sl == 0)
    recall = tp / max(pos, 1); fpr = fp / max(neg, 1)
    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0
    idx = np.flatnonzero(allowed); best = int(idx[np.argmax(recall[allowed])])
    return float(recall[best]), float(fpr[best])


def validator_reward(y_prob, y_true):
    y_prob = np.asarray(y_prob, dtype=float); y_true = np.asarray(y_true, dtype=int)
    ap = float(average_precision_score(y_true, y_prob)) if (y_prob.size and np.any(y_true == 1)) else 0.0
    bot_recall, fpr = recall_at_fpr(y_prob, y_true, max_fpr=0.05)
    base = 0.75 * ap + 0.25 * bot_recall
    return base, {'ap': ap, 'bot_recall': bot_recall, 'fpr': fpr, 'reward': base}


def daily_stats(test_df, probs):
    g = test_df.copy(); g['_prob'] = probs
    rows = []
    for day, gd in g.groupby('data'):
        rew, m = validator_reward(gd['_prob'].values, gd['label'].values)
        rows.append({'day': day, 'fpr': round(m['fpr'], 3),
                     'bot_recall': round(m['bot_recall'], 3),
                     'ap': round(m['ap'], 3), 'reward': round(rew, 3)})
    return pd.DataFrame(rows)


def _fit_qt(X):
    qt = QuantileTransformer(output_distribution='normal',
                             n_quantiles=min(200, len(X)), random_state=0)
    return qt, qt.fit_transform(X)


def _pseudo_rows(feature, med, train, prd_fit, mtype, params, pseudo_low, pseudo_high, spw=1.0):
    """Pseudo-label z PRD modelem bazowym (danego typu) trenowanym TYLKO na train."""
    Xtr = train[feature].fillna(med); ytr = train['label'].values
    qt, Xtr_n = _fit_qt(Xtr)
    m = _make_model(mtype, params, spw=spw); m.fit(Xtr_n, ytr)
    p = m.predict_proba(qt.transform(prd_fit[feature].fillna(med)))[:, 1]
    mask = (p > pseudo_high) | (p < pseudo_low)
    if mask.sum() == 0:
        return None, None
    return prd_fit[feature].fillna(med)[mask], (p[mask] > pseudo_high).astype(int)


def _sel_score(day_df):
    return day_df['reward'].mean() - 0.5 * day_df['reward'].std()


def train_one_model(mtype, feature, med, train, test, prd_fit, prd_eval,
                    fit_on, pseudo, params=None, spw=1.0,
                    pseudo_low=DEFAULT_PSEUDO_LOW, pseudo_high=DEFAULT_PSEUDO_HIGH):
    """Trenuje JEDEN base model danego typu. Zwraca (artifact, prob_test, prob_eval, metrics)."""
    params = params or DEFAULT_PARAMS[mtype]
    base = train if fit_on == "train" else pd.concat([train, test], axis=0)
    Xbase = base[feature].fillna(med); ybase = base['label'].values

    qt_prd = QuantileTransformer(output_distribution='normal',
                                 n_quantiles=min(200, len(prd_fit)), random_state=0)
    qt_prd.fit(prd_fit[feature].fillna(med))

    if pseudo:
        X_ps, y_ps = _pseudo_rows(feature, med, train, prd_fit, mtype, params,
                                  pseudo_low, pseudo_high, spw=spw)
        if X_ps is not None:
            Xbase = pd.concat([Xbase, X_ps], axis=0)
            ybase = np.concatenate([ybase, y_ps])

    qt_train, Xbase_n = _fit_qt(Xbase)
    model = _make_model(mtype, params, spw=spw); model.fit(Xbase_n, ybase)

    prob_te = model.predict_proba(qt_train.transform(test[feature].fillna(med)))[:, 1]
    prob_ev = model.predict_proba(qt_prd.transform(prd_eval[feature].fillna(med)))[:, 1]
    day_df = daily_stats(test, prob_te)
    reward_full, m_full = validator_reward(prob_te, test['label'].values)

    artifact = {'mtype': mtype, 'feature': feature, 'median': med,
                'qt_train': qt_train, 'qt_prd': qt_prd, 'model': model,
                'params': dict(params), 'spw': spw}
    metrics = {'mtype': mtype, 'reward_full': round(reward_full, 4),
               'reward_daily': round(day_df['reward'].mean(), 4),
               'reward_daily_std': round(day_df['reward'].std(), 4),
               'sel_score': round(_sel_score(day_df), 4),
               'ap': round(m_full['ap'], 3), 'bot_recall': round(m_full['bot_recall'], 3),
               'n_features': len(feature)}
    return artifact, prob_te, prob_ev, metrics


def search_ks_for_model(mtype, ks_features, train, test, prd_fit, prd_eval,
                        fit_on, pseudo, pseudo_low, pseudo_high, spw=1.0):
    """Szuka najlepszego KS dla JEDNEGO typu modelu (domyślne HP). Zwraca (best_ks, log_df)."""
    best = None; rows = []
    for ks, (feat, med) in ks_features.items():
        # SELEKCJA KS: ZAWSZE trenuj na train, oceniaj na held-out test (uczciwie).
        # Prawdziwe fit_on wchodzi dopiero do finalnego artefaktu (build_stack).
        _, _, _, m = train_one_model(mtype, feat, med, train, test, prd_fit, prd_eval,
                                     "train", pseudo, params=DEFAULT_PARAMS[mtype], spw=spw,
                                     pseudo_low=pseudo_low, pseudo_high=pseudo_high)
        rows.append({'ks': ks, **m})
        if best is None or m['sel_score'] > best[1]:
            best = (ks, m['sel_score'])
    return best[0], pd.DataFrame(rows).sort_values('ks')


def hpo_for_model(mtype, feature, med, train, test, prd_fit, prd_eval,
                  fit_on, pseudo, n_iter, pseudo_low, pseudo_high, spw=1.0):
    """Lekki HPO dla jednego modelu przy ustalonym KS. Ocena na test. Zwraca (best_params, log)."""
    grid = PARAM_GRIDS[mtype]
    keys = list(grid.keys())
    all_combos = list(product(*[grid[k] for k in keys]))
    if n_iter and 0 < n_iter < len(all_combos):
        idx = np.random.RandomState(0).choice(len(all_combos), size=n_iter, replace=False)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos

    best = None; rows = []
    for vals in combos:
        params = dict(zip(keys, vals))
        # HPO trenuje tylko na train (uczciwa ocena na test), final dopiero potem
        _, prob_te, _, m = train_one_model(mtype, feature, med, train, test,
                                           prd_fit, prd_eval, "train", pseudo,
                                           params=params, spw=spw, pseudo_low=pseudo_low,
                                           pseudo_high=pseudo_high)
        rows.append({**params, 'reward_full': m['reward_full'], 'sel_score': m['sel_score']})
        if best is None or m['sel_score'] > best[1]:
            best = (params, m['sel_score'])
    return best[0], pd.DataFrame(rows).sort_values('sel_score', ascending=False)


def build_stack(ks_features, train, test, prd_fit, prd_eval, fit_on, pseudo,
                do_hpo, hpo_n_iter, pseudo_low, pseudo_high, analysis_dir, spw=1.0):
    """Buduje JEDEN stack: dla każdego z 5 modeli szuka KS+HPO, trenuje final,
    uśrednia predykcje. Zwraca (stack_artifact, metrics, prob_te_avg, prob_ev_avg)."""
    members = []
    probs_te = []; probs_ev = []
    per_model_rows = []

    for mtype in MODEL_TYPES:
        if mtype == "lgb" and lgb is None:      continue
        if mtype == "cat" and CatBoostClassifier is None:  continue

        # 1) najlepszy KS dla tego modelu
        best_ks, ks_log = search_ks_for_model(
            mtype, ks_features, train, test, prd_fit, prd_eval,
            fit_on, pseudo, pseudo_low, pseudo_high, spw=spw)
        ks_log.to_csv(os.path.join(analysis_dir, f"ks_{mtype}.csv"), index=False)
        feat, med = ks_features[best_ks]

        # 2) lekki HPO
        best_params = DEFAULT_PARAMS[mtype]
        if do_hpo:
            best_params, hpo_log = hpo_for_model(
                mtype, feat, med, train, test, prd_fit, prd_eval,
                fit_on, pseudo, hpo_n_iter, pseudo_low, pseudo_high, spw=spw)
            hpo_log.to_csv(os.path.join(analysis_dir, f"hpo_{mtype}.csv"), index=False)

        # 3a) UCZCIWE predykcje członka (train -> held-out test) — TE idą do metryk
        #     i sortu stacka. Bez tego członkowie train+test dawaliby in-sample prob_te.
        art_honest, prob_te, prob_ev, m = train_one_model(
            mtype, feat, med, train, test, prd_fit, prd_eval,
            "train", pseudo, params=best_params, spw=spw,
            pseudo_low=pseudo_low, pseudo_high=pseudo_high)

        # 3b) FINALNY artefakt produkcyjny: refit na prawdziwym fit_on (train+test gdy
        #     wybrane). prob_te z tego modelu byłoby in-sample, więc go NIE używamy do metryk.
        if fit_on != "train":
            art, _, _, _ = train_one_model(
                mtype, feat, med, train, test, prd_fit, prd_eval,
                fit_on, pseudo, params=best_params, spw=spw,
                pseudo_low=pseudo_low, pseudo_high=pseudo_high)
        else:
            art = art_honest

        art['best_ks'] = best_ks
        members.append(art)
        probs_te.append(prob_te); probs_ev.append(prob_ev)
        per_model_rows.append({'mtype': mtype, 'best_ks': best_ks,
                               'reward_full': m['reward_full'],
                               'sel_score': m['sel_score'], 'ap': m['ap'],
                               'params': json.dumps(best_params)})
        print(f"      [{mtype}] KS={best_ks} reward_full={m['reward_full']} "
              f"ap={m['ap']} sel={m['sel_score']}")

    # STACK = zwykła średnia predykcji
    prob_te_avg = np.mean(np.column_stack(probs_te), axis=1)
    prob_ev_avg = np.mean(np.column_stack(probs_ev), axis=1)

    day_df = daily_stats(test, prob_te_avg)
    reward_full, m_full = validator_reward(prob_te_avg, test['label'].values)
    balance = min(np.mean(prob_ev_avg < 0.2), np.mean(prob_ev_avg > 0.8))

    stack_artifact = {'kind': 'mean_stack', 'members': members,
                      'model_types': [a['mtype'] for a in members]}
    metrics = {'reward_full': round(reward_full, 4),
               'reward_daily': round(day_df['reward'].mean(), 4),
               'reward_daily_std': round(day_df['reward'].std(), 4),
               'sel_score': round(_sel_score(day_df), 4),
               'ap': round(m_full['ap'], 3), 'bot_recall': round(m_full['bot_recall'], 3),
               'balance_eval': round(balance, 3), 'n_members': len(members)}
    pd.DataFrame(per_model_rows).to_csv(
        os.path.join(analysis_dir, "members.csv"), index=False)
    return stack_artifact, metrics


def run(df, prd, auc_,
        split_date=DEFAULT_SPLIT_DATE, out_dir=DEFAULT_OUT_DIR,
        grid_ks=DEFAULT_GRID_KS, grid_fit_on=DEFAULT_GRID_FIT_ON,
        grid_pseudo=DEFAULT_GRID_PSEUDO, grid_spw=[1.0],
        pseudo_low=DEFAULT_PSEUDO_LOW, pseudo_high=DEFAULT_PSEUDO_HIGH,
        do_hpo=True, hpo_n_iter=DEFAULT_HPO_N_ITER, auc_min=0.50):
    """Buduje siatkę stacków (fit_on x pseudo x spw). Każdy stack = średnia 5 modeli,
    każdy model z własnym najlepszym KS + lekkim HPO.

    grid_spw: lista wartości scale_pos_weight. DOMYŚLNIE [1.0] = WYŁĄCZONY (neutralny),
      bo pod rankingowy reward spw ma znikomy wpływ. Żeby zrobić eksperyment kontrolny
      (czy spw w ogóle pomaga), podaj np. grid_spw=[0.3, 0.7, 1.0] na jednej kombinacji
      fit_on/pseudo i porównaj reward_full między stackami."""
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "model"); os.makedirs(models_dir, exist_ok=True)

    train    = df[df['data'] <  split_date]
    test     = df[df['data'] >= split_date]
    prd_fit  = prd[prd['data'] <  split_date]
    prd_eval = prd[prd['data'] >= split_date]

    ks_features = {}
    for ks in grid_ks:
        feat = auc_[(auc_['ks'] < ks) & (auc_['auc'] >= auc_min)]['feature'].tolist()
        if feat:
            ks_features[ks] = (feat, train[feat].median())

    combos = list(product(grid_fit_on, grid_pseudo, grid_spw))
    avail = [m for m in MODEL_TYPES
             if not (m == "lgb" and lgb is None) and not (m == "cat" and CatBoostClassifier is None)]
    _spw_note = "" if grid_spw == [1.0] else f" x spw{grid_spw}"
    print(f"XGB device: {'cuda' if _CUDA else 'cpu'} | modele w stacku: {avail}")
    print(f"{len(combos)} stacków (fit_on x pseudo{_spw_note}), każdy = średnia {len(avail)} modeli\n")

    final_rows = []
    for cid, (fit_on, pseudo, spw) in enumerate(combos, 1):
        spw_tag = "" if spw == 1.0 else f"_spw{spw}"
        stack_name = f"stack{cid}"                                        # prosta nazwa
        full_name  = f"stack{cid}_{fit_on.replace('+','-')}_ps{int(pseudo)}{spw_tag}"  # opisowa
        analysis_dir = os.path.join(out_dir, "analysis", stack_name)
        os.makedirs(analysis_dir, exist_ok=True)
        print("=" * 80)
        print(f"[{cid}/{len(combos)}] {full_name}")

        stack_art, m = build_stack(ks_features, train, test, prd_fit, prd_eval,
                                   fit_on, pseudo, do_hpo, hpo_n_iter,
                                   pseudo_low, pseudo_high, analysis_dir, spw=spw)
        stack_art['config'] = {'fit_on': fit_on, 'pseudo': pseudo, 'spw': spw,
                               'split_date': split_date}
        stack_art['tag'] = full_name
        joblib.dump(stack_art, os.path.join(models_dir, f"{stack_name}.joblib"))

        with open(os.path.join(analysis_dir, "summary.json"), "w") as f:
            json.dump({'stack': stack_name, 'full_name': full_name, 'fit_on': fit_on,
                       'pseudo': pseudo, 'spw': spw, **m}, f, indent=2)

        print(f"   STACK reward_full={m['reward_full']} reward_daily={m['reward_daily']} "
              f"(std={m['reward_daily_std']}) ap={m['ap']} balance={m['balance_eval']}")

        final_rows.append({'stack': stack_name, 'full_name': full_name,
                           'fit_on': fit_on, 'pseudo': pseudo, 'spw': spw,
                           'model_path': os.path.join(models_dir, f"{stack_name}.joblib"), **m})

    final = pd.DataFrame(final_rows).sort_values('sel_score', ascending=False)
    final.to_csv(os.path.join(out_dir, "final_stacks.csv"), index=False)
    print("\n" + "=" * 80)
    print("FINALNE STACKI (sort wg sel_score):")
    print(final[['stack', 'fit_on', 'pseudo', 'spw', 'reward_full', 'reward_daily',
                 'reward_daily_std', 'ap', 'balance_eval', 'sel_score']].to_string(index=False))
    return final


if __name__ == "__main__":
    df   = pd.read_parquet("df.parquet")
    prd  = pd.read_parquet("prd.parquet")
    auc_ = pd.read_csv("auc_.csv")
    final = run(df, prd, auc_)


# ============ INFERENCE ============
def predict_stack(stack_artifact, prd_df):
    """Predykcja uśrednionego stacka na nowych danych PRD.
    Każdy member ma własny feature/median/qt_prd — uśredniamy ich predykcje.
    prd_df: DataFrame z policzonymi cechami (te same create_features*)."""
    cols = []
    for member in stack_artifact['members']:
        feat = member['feature']; med = member['median']
        qt_prd = member['qt_prd']; model = member['model']
        X = prd_df.copy()
        for f in feat:
            if f not in X.columns:
                X[f] = np.nan
        X = X[feat].fillna(med)
        Xn = qt_prd.transform(X)
        cols.append(model.predict_proba(Xn)[:, 1])
    return np.mean(np.column_stack(cols), axis=1)


def load_and_predict(model_path, prd_df):
    """Wygodny wrapper: ładuje stack z pliku i predykuje."""
    stack = joblib.load(model_path)
    return predict_stack(stack, prd_df)