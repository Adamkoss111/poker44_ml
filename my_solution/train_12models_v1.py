"""
train_12models.py

12 kombinacji ZEWNĘTRZNYCH: scale_pos_weight x fit_on x pseudo (3*2*2).
Dla KAŻDEJ kombinacji:
  - iterujemy po progach KS,
  - wybieramy NAJLEPSZY próg KS (wg sel_score),
  - zapisujemy JEDEN wytrenowany model z tym najlepszym KS.
Wynik: 12 gotowych produkcyjnie modeli + tabela.

sel_score (wybór najlepszego KS w obrębie kombinacji) premiuje stabilność:
  sel_score = reward_daily - 0.5*reward_daily_std

Reward liczony jest 1:1 jak walidator (poker44/score/scoring.py):
  reward = 0.75*AP + 0.25*bot_recall,  bot_recall = recall przy FPR <= 0.05,
  human_safety_penalty = 1.0 (brak twardego klifu FPR).
"""

import os
import joblib
import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import average_precision_score
import xgboost as xgb


def _detect_cuda():
    """Wykrywa CUDA dla XGBoost. Override: POKER44_XGB_DEVICE='cuda'|'cpu'."""
    override = str(os.getenv("POKER44_XGB_DEVICE", "")).strip().lower()
    if override in {"cpu", "cuda"}:
        return override == "cuda"
    try:                              # 1) torch jeśli zainstalowany
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:                             # 2) nvidia-smi
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


def _xgb_device_params():
    """Parametry XGB do GPU (gdy CUDA dostępne), dobrane wg wersji XGBoost."""
    if not _CUDA:
        return {}
    try:
        major = int(str(xgb.__version__).split(".")[0])
    except Exception:
        major = 2
    if major >= 2:
        return {"device": "cuda", "tree_method": "hist"}   # XGBoost 2.x
    return {"tree_method": "gpu_hist"}                      # XGBoost 1.x


# ============ DOMYŚLNA KONFIGURACJA (używana przez __main__ i jako defaulty run()) ============
DEFAULT_SPLIT_DATE = "2026-06-23"
DEFAULT_PSEUDO_LOW, DEFAULT_PSEUDO_HIGH = 0.10, 0.90
DEFAULT_OUT_DIR = "production_models"

DEFAULT_GRID_KS     = [round(x, 2) for x in np.arange(0.30, 0.70, 0.02)]
DEFAULT_GRID_SPW    = [0.1, 0.3, 0.5]
DEFAULT_GRID_FIT_ON = ["train", "train+test"]
DEFAULT_GRID_PSEUDO = [False, True]

# Hiperparametry XGB. DEFAULT_PARAMS uzywane podczas szukania KS (szybko),
# DEFAULT_PARAM_GRID przeszukiwany w HPO PO wyborze KS. scale_pos_weight NIE jest
# tu strojony — siedzi w zewnetrznej petli (spw).
# Liczba kroków HPO na 1 kombinację. Randomized search: losuje tyle kombinacji z
# pełnej siatki (pełny grid = 432 kroki, za dużo). None -> pełny grid search.
DEFAULT_HPO_N_ITER = 40
DEFAULT_PARAMS = {"max_depth": 3, "n_estimators": 40}
DEFAULT_PARAM_GRID = {
    "max_depth":        [3, 4, 5],
    "n_estimators":     [40, 80, 150],
    "learning_rate":    [0.05, 0.1, 0.3],
    "subsample":        [0.8, 1.0],   # ułamek WIERSZY (próbek) na drzewo
    "colsample_bytree": [0.8, 1.0],   # ułamek KOLUMN (cech) na drzewo
    "min_child_weight": [1, 5],       # regularyzacja: min. waga liści (anty-overfit)
    "reg_lambda":       [1.0, 5.0],   # L2 na wagi liści
}
# =============================================================================================


def recall_at_fpr(y_score, y_true, max_fpr=0.05):
    """Najlepszy bot-recall osiągalny przy FPR ludzi <= max_fpr.
    Wierne odwzorowanie poker44/score/scoring.py::_recall_at_fpr — walidator NIE
    progu je przy 0.5, tylko przesuwa próg po rankingu prawdopodobieństw."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    pos = int(np.sum(labels == 1)); neg = int(np.sum(labels == 0))
    if pos <= 0 or neg <= 0 or scores.size == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    tp = np.cumsum(sl == 1); fp = np.cumsum(sl == 0)
    recall = tp / max(pos, 1); fpr = fp / max(neg, 1)
    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0
    idx = np.flatnonzero(allowed)
    best = int(idx[np.argmax(recall[allowed])])
    return float(recall[best]), float(fpr[best])


def validator_reward(y_prob, y_true):
    """Reward IDENTYCZNY z poker44/score/scoring.py::reward.
    reward = 0.75*AP + 0.25*bot_recall ; human_safety_penalty = 1.0 (brak klifu).
    AP i bot_recall liczone na prawdopodobieństwach (ranking), nie na binarce 0.5."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if y_prob.size and np.any(y_true == 1):
        ap = float(average_precision_score(y_true, y_prob))
    else:
        ap = 0.0
    bot_recall, fpr = recall_at_fpr(y_prob, y_true, max_fpr=0.05)
    base = 0.75 * ap + 0.25 * bot_recall
    return base, {'ap': ap, 'bot_recall': bot_recall, 'fpr': fpr,
                  'base_score': base, 'reward': base}


def daily_stats(test_df, preds, probs):
    # preds (próg 0.5) NIE wpływa na reward walidatora — zostaje tylko jako diagnostyka.
    g = test_df.copy(); g['_prob'] = probs
    rows = []
    for day, gd in g.groupby('data'):
        yt = gd['label'].values
        rew, m = validator_reward(gd['_prob'].values, yt)
        rows.append({'day': day, 'fpr': round(m['fpr'], 3),
                     'bot_recall': round(m['bot_recall'], 3),
                     'ap': round(m['ap'], 3), 'reward': round(rew, 3)})
    return pd.DataFrame(rows)


def _fit_qt(X):
    """QuantileTransformer(normal) dopasowany na X. Zwraca (qt, X_znormalizowane)."""
    qt = QuantileTransformer(output_distribution='normal',
                             n_quantiles=min(200, len(X)), random_state=0)
    return qt, qt.fit_transform(X)


def _xgb(params, spw):
    """XGBClassifier z domyslnymi + nadpisanymi hiperparametrami i danym spw.
    Gdy CUDA dostepne -> trenuje na GPU (dotyczy KS search, HPO, pseudo i finalu)."""
    p = {**DEFAULT_PARAMS, **_xgb_device_params(), **(params or {})}
    return xgb.XGBClassifier(scale_pos_weight=spw, random_state=0, **p)


def _pseudo_rows(feature, med, train, prd_fit, spw, params, pseudo_low, pseudo_high):
    """Generuje wiersze pseudo-label z PRD modelem bazowym trenowanym TYLKO na train
    (nie na test — zeby ocena na test pozostala uczciwa). Zwraca (X_ps, y_ps) lub (None, None).

    UWAGA covariate-shift: model bazowy trenuje na train (qt_tr), ale PRD scoruje przez
    qt_prd DOPASOWANY NA PRD — spójnie z inferencją produkcyjną. Użycie qt_tr na PRD
    dawałoby przesunięte, nie-normalne wartości -> tendencyjne (jednostronne) pseudo-labele."""
    Xtr = train[feature].fillna(med); ytr = train['label'].values
    qt_tr, Xtr_n = _fit_qt(Xtr)
    m = _xgb(params, spw); m.fit(Xtr_n, ytr)
    Xprd = prd_fit[feature].fillna(med)
    qt_pr, _ = _fit_qt(Xprd)                       # QT dopasowany na PRD (nie na train)
    p = m.predict_proba(qt_pr.transform(Xprd))[:, 1]
    mask = (p > pseudo_high) | (p < pseudo_low)
    if mask.sum() == 0:
        return None, None
    X_ps = Xprd[mask]
    y_ps = (p[mask] > pseudo_high).astype(int)
    return X_ps, y_ps


def _eval_on_test(qt_train, model, test, feature, med):
    """Liczy day_df + sel_score na TEST dla gotowego (qt, model)."""
    prob_te = model.predict_proba(qt_train.transform(test[feature].fillna(med)))[:, 1]
    day_df = daily_stats(test, (prob_te > 0.5).astype(int), prob_te)
    sel = day_df['reward'].mean() - 0.5 * day_df['reward'].std()
    return prob_te, day_df, sel


def optimize_hyperparams(feature, med, train, test, prd_fit, spw, pseudo,
                         param_grid=None, objective="sel_score", n_iter=DEFAULT_HPO_N_ITER,
                         pseudo_low=DEFAULT_PSEUDO_LOW, pseudo_high=DEFAULT_PSEUDO_HIGH):
    """HPO PO wyborze KS. Trenuje na train (+pseudo gdy pseudo=True), ocenia na TEST.
    NIGDY nie trenuje na test — dlatego ocena jest uczciwa nawet dla kombinacji
    fit_on='train+test' (final na train+test robi sie dopiero pozniej, w train_one).

    objective:
      'reward'    -> DOKŁADNIE metryka walidatora: validator_reward(prob, y) na całym
                     teście (= poker44/score/scoring.py, 1:1, bez żadnych dodatków).
      'sel_score' -> reward liczony PER DZIEŃ (jak walidator ocenia porcjami) minus
                     0.5*std między dniami; ta sama metryka bazowa, ale premiuje
                     stabilność (domyślne, spójne z wyborem KS).
    n_iter: liczba prób (randomized search). None lub >= rozmiar siatki -> pełny grid.
    Zwraca (best_params, search_log_df)."""
    param_grid = param_grid or DEFAULT_PARAM_GRID
    if objective not in {"reward", "sel_score"}:
        raise ValueError("objective musi być 'reward' albo 'sel_score'")

    # Baza HPO: train (+pseudo). Bez test.
    Xbase = train[feature].fillna(med); ybase = train['label'].values
    if pseudo:
        X_ps, y_ps = _pseudo_rows(feature, med, train, prd_fit, spw,
                                  DEFAULT_PARAMS, pseudo_low, pseudo_high)
        if X_ps is not None:
            Xbase = pd.concat([Xbase, X_ps], axis=0)
            ybase = np.concatenate([ybase, y_ps])

    keys = list(param_grid.keys())
    all_combos = list(product(*[param_grid[k] for k in keys]))
    # Randomized search: gdy n_iter < rozmiar siatki, losujemy n_iter kombinacji
    # (seed stały -> powtarzalne). n_iter=None lub >= siatka -> pełny grid search.
    if n_iter is not None and 0 < n_iter < len(all_combos):
        sel_idx = np.random.RandomState(0).choice(len(all_combos), size=n_iter, replace=False)
        combos = [all_combos[i] for i in sel_idx]
    else:
        combos = all_combos

    rows = []; best = None
    for vals in combos:
        params = dict(zip(keys, vals))
        qt, Xn = _fit_qt(Xbase)
        m = _xgb(params, spw); m.fit(Xn, ybase)
        prob_te, day_df, sel = _eval_on_test(qt, m, test, feature, med)
        reward_full, _ = validator_reward(prob_te, test['label'].values)  # dokładnie jak walidator
        score = reward_full if objective == "reward" else sel
        rows.append({**params,
                     'reward_full': round(reward_full, 4),
                     'sel_score': round(sel, 4),
                     'reward_daily': round(day_df['reward'].mean(), 4),
                     'reward_daily_std': round(day_df['reward'].std(), 4)})
        if best is None or score > best[1]:
            best = (params, score)

    log = pd.DataFrame(rows).sort_values(
        'reward_full' if objective == "reward" else 'sel_score',
        ascending=False).reset_index(drop=True)
    return best[0], log


def train_one(feature, med, train, test, prd_fit, prd_eval, spw, fit_on, pseudo,
              params=None, pseudo_low=DEFAULT_PSEUDO_LOW, pseudo_high=DEFAULT_PSEUDO_HIGH):
    """Trenuje jeden FINALNY model dla danego zestawu cech + hiperparametrow (params).
    fit_on='train+test' => final trenuje na train+test (po ewentualnym HPO).
    pseudo=True => baza rozszerzona o pewne PRD (pseudo-labele z modelu na train).
    Zwraca (artifact_dict, metryki_dict)."""
    params = params or DEFAULT_PARAMS
    base = train if fit_on == "train" else pd.concat([train, test], axis=0)
    Xbase = base[feature].fillna(med); ybase = base['label'].values

    qt_prd = QuantileTransformer(output_distribution='normal',
                                 n_quantiles=min(200, len(prd_fit)), random_state=0)
    qt_prd.fit(prd_fit[feature].fillna(med))

    n_ps_bot = n_ps_human = 0
    if pseudo:
        X_ps, y_ps = _pseudo_rows(feature, med, train, prd_fit, spw,
                                  params, pseudo_low, pseudo_high)
        if X_ps is not None:
            Xbase = pd.concat([Xbase, X_ps], axis=0)
            ybase = np.concatenate([ybase, y_ps])
            n_ps_bot = int(np.sum(y_ps == 1)); n_ps_human = int(np.sum(y_ps == 0))

    qt_train, Xbase_n = _fit_qt(Xbase)
    model = _xgb(params, spw); model.fit(Xbase_n, ybase)

    prob_te = model.predict_proba(qt_train.transform(test[feature].fillna(med)))[:, 1]
    preds_te = (prob_te > 0.5).astype(int)
    day_df = daily_stats(test, preds_te, prob_te)

    # Reward na całym teście — identyczny z walidatorem (na prawdopodobieństwach).
    reward_full, m_full = validator_reward(prob_te, test['label'].values)
    fpr = m_full['fpr']; recall = m_full['bot_recall']; ap = m_full['ap']

    p_ev = model.predict_proba(qt_prd.transform(prd_eval[feature].fillna(med)))[:, 1]
    balance = min(np.mean(p_ev<0.2), np.mean(p_ev>0.8))

    # Per-dzień (tak jak walidator ocenia porcjami) — premiujemy stabilność.
    # Brak członu cliff: nowy reward nie ma twardego klifu FPR, więc nie ma czego karać.
    r_daily = day_df['reward'].mean()
    r_std = day_df['reward'].std()
    sel_score = r_daily - 0.5*r_std

    artifact = {'feature': feature, 'median': med,
                'qt_train': qt_train, 'qt_prd': qt_prd, 'model': model,
                'params': dict(params)}
    metrics = {'reward_full':round(reward_full,4),'reward_daily':round(r_daily,4),
               'reward_daily_std':round(r_std,4),
               'fpr':round(fpr,3),'bot_recall':round(recall,3),'ap':round(ap,3),
               'balance_eval':round(balance,3),'sel_score':round(sel_score,4),
               'pseudo_n_bot':n_ps_bot,'pseudo_n_human':n_ps_human,
               'n_features':len(feature),'params':dict(params)}
    return artifact, metrics


def run(df, prd, auc_,
        split_date=DEFAULT_SPLIT_DATE,
        out_dir=DEFAULT_OUT_DIR,
        grid_ks=DEFAULT_GRID_KS,
        grid_spw=DEFAULT_GRID_SPW,
        grid_fit_on=DEFAULT_GRID_FIT_ON,
        grid_pseudo=DEFAULT_GRID_PSEUDO,
        pseudo_low=DEFAULT_PSEUDO_LOW,
        pseudo_high=DEFAULT_PSEUDO_HIGH,
        param_grid=DEFAULT_PARAM_GRID,
        do_hpo=True,
        hpo_objective="sel_score",
        hpo_n_iter=DEFAULT_HPO_N_ITER,
        auc_min=0.50):
    """Trenuje siatkę modeli (spw x fit_on x pseudo), w każdej kombinacji:
      1) optymalizuje próg KS (trening na train, ocena na held-out test — uczciwie),
      2) PO wyborze KS robi HPO hiperparametrów XGB (trening na train, ocena na test),
      3) liczy uczciwe metryki porównawcze (train -> test) do sortu 12 modeli,
      4) trenuje FINALNY artefakt z najlepszymi HP (refit na train+test jeśli fit_on
         tego wymaga) — dopiero ten krok pokazuje modelowi test.
    Selekcja (KS, HPO, porównanie) NIGDY nie trenuje na test; train+test wchodzi
    wyłącznie do produkcyjnego .joblib. Zwraca DataFrame z podsumowaniem.

    df, prd     : DataFrame z kolumną 'data' i (df) 'label'
    auc_        : DataFrame z kolumnami 'feature', 'auc', 'ks'
    param_grid    : siatka hiperparametrów XGB do HPO (dict {nazwa: [wartości]})
    do_hpo        : False -> pomija HPO, używa DEFAULT_PARAMS
    hpo_objective : 'sel_score' (domyśl., reward per-dzień - 0.5*std, premiuje stabilność)
                    lub 'reward' (DOKŁADNIE metryka walidatora na całym teście)
    hpo_n_iter    : liczba kroków HPO na kombinację (randomized search). None -> pełny
                    grid (=iloczyn rozmiarów param_grid, domyślnie 432). Domyślnie 40.
    """
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "model")        # wszystkie modele razem
    os.makedirs(models_dir, exist_ok=True)

    train    = df[df['data'] <  split_date]
    test     = df[df['data'] >= split_date]
    prd_fit  = prd[prd['data'] <  split_date]
    prd_eval = prd[prd['data'] >= split_date]

    # pre-licz cechy i mediany per próg KS (wspólne dla wszystkich kombinacji)
    ks_features = {}
    for ks in grid_ks:
        feat = auc_[(auc_['ks'] < ks) & (auc_['auc'] >= auc_min)]['feature'].tolist()
        if feat:
            ks_features[ks] = (feat, train[feat].median())

    combos = list(product(grid_spw, grid_fit_on, grid_pseudo))
    print(f"XGBoost device: {'cuda (GPU)' if _CUDA else 'cpu'}"
          + (f" | tree_method={_xgb_device_params().get('tree_method')}" if _CUDA else ""))
    print(f"{len(combos)} kombinacji, każda optymalizuje KS po {len(ks_features)} progach\n")

    final_rows = []

    for combo_id, (spw, fit_on, pseudo) in enumerate(combos, 1):
        combo_tag = f"spw{spw}_{fit_on.replace('+','-')}_ps{int(pseudo)}"
        model_name   = f"model{combo_id}"                            # prosta nazwa pliku
        full_name    = f"model{combo_id}_{combo_tag}"                # opisowa, do analiz
        analysis_dir = os.path.join(out_dir, "analysis", model_name)  # analizy per model
        os.makedirs(analysis_dir, exist_ok=True)
        print("="*80)
        print(f"[{combo_id}/{len(combos)}] {combo_tag} — szukam najlepszego KS...")

        best = None              # (ks, artifact, metrics)
        ks_search_combo = []     # log KS TYLKO dla tej kombinacji
        for ks, (feat, med) in ks_features.items():
            # SELEKCJA KS: ZAWSZE trenuj na train, oceniaj na held-out test.
            # NIE używamy tu fit_on='train+test' — inaczej model widziałby test
            # w treningu i sel_score byłby in-sample (~1.0), a wybór KS nieuczciwy.
            # Prawdziwe fit_on wchodzi dopiero przy finalnym refit (niżej).
            art, m = train_one(feat, med, train, test, prd_fit, prd_eval,
                               spw, "train", pseudo,
                               pseudo_low=pseudo_low, pseudo_high=pseudo_high)
            ks_search_combo.append({'ks':ks, **m})
            if best is None or m['sel_score'] > best[2]['sel_score']:
                best = (ks, art, m)

        best_ks, best_art, best_m = best
        best_feat, best_med = ks_features[best_ks]

        # --- HPO PO wyborze KS: szukaj HP na train (+pseudo), oceniaj na TEST ---
        best_params = dict(DEFAULT_PARAMS)
        if do_hpo:
            from math import prod as _prod
            _grid_size = _prod(len(v) for v in (param_grid or DEFAULT_PARAM_GRID).values())
            _steps = min(hpo_n_iter, _grid_size) if hpo_n_iter else _grid_size
            print(f"   KS={best_ks} wybrany -> HPO: {_steps} kroków "
                  f"({'random' if hpo_n_iter and hpo_n_iter < _grid_size else 'full grid'} "
                  f"z {_grid_size}), cel='{hpo_objective}', ocena na test...")
            best_params, hpo_log = optimize_hyperparams(
                best_feat, best_med, train, test, prd_fit, spw, pseudo,
                param_grid=param_grid, objective=hpo_objective, n_iter=hpo_n_iter,
                pseudo_low=pseudo_low, pseudo_high=pseudo_high)
            hpo_log.to_csv(os.path.join(analysis_dir, "hpo_search.csv"), index=False)

        # --- METRYKI DO PORÓWNANIA 12 MODELI: uczciwe, out-of-sample (train -> test) ---
        # Model liczony na train, oceniany na held-out test. Te metryki (best_m) idą do
        # final_12_models.csv i sortu — dzięki temu modele train+test NIE wygrywają
        # sztucznie sel_score'em ~1.0 liczonym in-sample.
        best_art, best_m = train_one(
            best_feat, best_med, train, test, prd_fit, prd_eval,
            spw, "train", pseudo, params=best_params,
            pseudo_low=pseudo_low, pseudo_high=pseudo_high)

        # --- FINALNY ARTEFAKT PRODUKCYJNY: refit na PRAWDZIWYM fit_on ---
        # Gdy wybrano fit_on='train+test', wybrany (KS+HP) model trenujemy TERAZ na
        # train+test i to on ląduje w .joblib. best_m zostaje uczciwe (z train->test).
        if fit_on != "train":
            best_art, _ = train_one(
                best_feat, best_med, train, test, prd_fit, prd_eval,
                spw, fit_on, pseudo, params=best_params,
                pseudo_low=pseudo_low, pseudo_high=pseudo_high)

        tag = f"{full_name}_ks{best_ks}"
        best_art['config'] = {'ks':best_ks,'scale_pos_weight':spw,
                              'fit_on':fit_on,'pseudo':pseudo,'split_date':split_date,
                              'params':best_params}
        best_art['tag'] = tag

        # --- model -> wspólny model/<nazwa>.joblib , analizy -> analysis/<nazwa>/ ---
        joblib.dump(best_art, os.path.join(models_dir, f"{model_name}.joblib"))

        ks_log_combo = pd.DataFrame(ks_search_combo).sort_values('ks')
        ks_log_combo['is_best'] = ks_log_combo['ks'] == best_ks
        ks_log_combo.to_csv(os.path.join(analysis_dir, "ks_search.csv"), index=False)

        try:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(9,4))
            ax1.plot(ks_log_combo['ks'], ks_log_combo['sel_score'], 'o-', label='sel_score')
            ax1.plot(ks_log_combo['ks'], ks_log_combo['reward_daily'], 's-',
                     label='reward_daily', alpha=0.6)
            ax1.axvline(best_ks, ls='--', color='green', label=f'best KS={best_ks}')
            ax1.set_xlabel('KS threshold'); ax1.set_ylabel('score'); ax1.legend(loc='upper left')
            ax2 = ax1.twinx()
            ax2.plot(ks_log_combo['ks'], ks_log_combo['reward_daily_std'], '^--',
                     color='red', alpha=0.4)
            ax2.set_ylabel('reward_daily_std', color='red')
            plt.title(combo_tag); plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, "ks_search.png"), dpi=90)
            plt.close()
        except Exception as e:
            print(f"   (wykres pominięty: {e})")

        import json
        with open(os.path.join(analysis_dir, "summary.json"), "w") as f:
            json.dump({'tag':tag,'best_ks':best_ks,'spw':spw,'fit_on':fit_on,
                       'pseudo':pseudo, **best_m}, f, indent=2)

        print(f"   -> najlepszy KS={best_ks} | reward_daily={best_m['reward_daily']} "
              f"(std={best_m['reward_daily_std']}) bot_recall={best_m['bot_recall']} "
              f"balance={best_m['balance_eval']}")
        if pseudo:
            print(f"      pseudo-labele: bot={best_m['pseudo_n_bot']} "
                  f"human={best_m['pseudo_n_human']} "
                  f"(jednostronne pseudo -> niski balance)")
        print(f"      HP={best_params}")
        print(f"      model/{model_name}.joblib  +  analysis/{model_name}/")

        final_rows.append({'model':model_name, 'full_name':full_name,
                           'model_path':os.path.join(models_dir, f"{model_name}.joblib"),
                           'spw':spw,'fit_on':fit_on,
                           'pseudo':pseudo,'best_ks':best_ks, **best_m})

    final = pd.DataFrame(final_rows).sort_values('sel_score', ascending=False)
    final.to_csv(os.path.join(out_dir, "final_12_models.csv"), index=False)

    print("\n" + "="*80)
    print("12 FINALNYCH MODELI (sort wg sel_score):")
    print(final[['model','spw','fit_on','pseudo','best_ks','reward_daily',
                 'reward_daily_std','bot_recall','balance_eval','sel_score']].to_string(index=False))
    print(f"\nStruktura w {out_dir}/:")
    print("  model/                          <- WSZYSTKIE modele razem")
    print("    model01_..._ps0.joblib")
    print("    model02_..._ps1.joblib  ...")
    print("  analysis/<nazwa_modelu>/        <- analizy per model")
    print("    ks_search.csv | ks_search.png | summary.json")
    print(f"Zbiorcza tabela: {out_dir}/final_12_models.csv")
    return final


if __name__ == "__main__":
    df    = pd.read_parquet("df.parquet")
    prd   = pd.read_parquet("prd.parquet")
    auc_  = pd.read_csv("auc_.csv")
    final = run(df, prd, auc_)