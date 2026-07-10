"""
train_fusion.py — architektura TOP1 + nasze dodatki.

OD NICH (top1):
  - StackedEnsemble (stacked_top1.py): base modele + meta-learner (logistic)
    trenowany na OUT-OF-FOLD predykcjach baz.
  - Selekcja cech top-K po importance (opcjonalna).

OD NAS:
  1. Cechy: WSZYSTKIE nasze zestawy (v1..v4) z parametrem `feature_sets`
     decydującym, które prefiksy wchodzą do modelu:
       v1 = brak prefiksu, v2 = 'new_', v3 = 'new2_', v4 = 'new3_' (= ich schema_).
  2. Selekcja/raport po sel_score = reward_daily - 0.5*std (reward per porcja,
     jak walidator, nie na całym zbiorze).
  3. Ewaluacja TIME-SPLIT (train < split_date <= test), nie losowy K-fold.
     (K-fold zostaje TYLKO wewnątrz train do OOF meta-learnera — to legalne.)
  4. (w ramach 1) v3: profil pozycyjny / GTO mixing / sizing buckets.
  5. Opcjonalny filtr stabilności KS (`ks_max`) PRZED top-K po importance.
  6. Monitoring rozkładu: balance + KDE na danych kalibracyjnych.
  7. Opcjonalny quantile per-zbiór (`use_qt`) jako eksperyment A/B.

ARTEFAKT jest zgodny z miner2 (feature / median / qt_calib / model):
  model = StackedEnsemble z .predict_proba(rows) -> miner2 działa BEZ ZMIAN.
  Gdy use_qt=False, qt_calib = FunctionTransformer (identyczność) — miner2 dalej
  woła .transform() i dostaje surowe wartości.

Użycie:
    from train_fusion import run
    final = run(df, calibration_data, auc_,
                feature_sets=['v1','v2','v4'],   # które zestawy cech
                ks_max=None,                     # np. 0.4 włącza filtr KS
                top_k=150,                       # top-K po importance (None=off)
                use_qt=False,                    # True = nasz quantile per-zbiór
                meta='logistic')                 # 'logistic' (ich) | 'mean' (nasz)
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import average_precision_score

import xgboost as xgb
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from stacked_top1 import StackedEnsemble
from calibration_top1 import BlendedQuantileCalibrator

# ---------------- GPU (XGB) ----------------
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

def _xgb_device():
    if not _CUDA:
        return {}
    major = int(str(xgb.__version__).split(".")[0])
    return {"device": "cuda", "tree_method": "hist"} if major >= 2 \
        else {"tree_method": "gpu_hist"}

# ---------------- zestawy cech (prefiksy) ----------------
PREFIXES = {'v2': 'new_', 'v3': 'new2_', 'v4': 'new3_', 'v5': 'new4_'}
_ALL_PREFIXES = tuple(PREFIXES.values())
NON_FEATURES = {'label', 'data'}

MODEL_TYPES = ["xgb", "lgb", "cat", "rf", "et"]
DEFAULT_PARAMS = {
    "xgb": {"max_depth": 3, "n_estimators": 80},
    "lgb": {"max_depth": 3, "n_estimators": 80},
    "cat": {"depth": 3, "iterations": 120},
    "rf":  {"n_estimators": 300, "max_depth": 10},
    "et":  {"n_estimators": 300, "max_depth": 10},
}


def _make_model(mtype, params=None):
    p = {**DEFAULT_PARAMS[mtype], **(params or {})}
    if mtype == "xgb":
        return xgb.XGBClassifier(random_state=0, eval_metric="logloss",
                                 **_xgb_device(), **p)
    if mtype == "lgb":
        if lgb is None: raise RuntimeError("brak lightgbm")
        return lgb.LGBMClassifier(random_state=0, verbose=-1, **p)
    if mtype == "cat":
        if CatBoostClassifier is None: raise RuntimeError("brak catboost")
        return CatBoostClassifier(random_state=0, verbose=False, **p)
    if mtype == "rf":
        return RandomForestClassifier(random_state=0, n_jobs=-1, **p)
    if mtype == "et":
        return ExtraTreesClassifier(random_state=0, n_jobs=-1, **p)
    raise ValueError(mtype)


class MeanMeta:
    """Meta-'model' uśredniający kolumny (nasza alternatywa dla logistic).
    Zgodny interfejsem z meta_model w StackedEnsemble."""
    def fit(self, z, y=None):
        return self
    def predict_proba(self, z):
        p = np.clip(np.asarray(z, dtype=float).mean(axis=1), 0.0, 1.0)
        return np.stack([1.0 - p, p], axis=1)


# ---------------- reward 1:1 walidator ----------------
def recall_at_fpr(y_score, y_true, max_fpr=0.05):
    """1:1 z poker44/score/scoring.py::_recall_at_fpr"""
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


def threshold_metrics(y_score, y_true, threshold=0.5):
    """1:1 z poker44/score/scoring.py::_threshold_metrics.

    UWAGA: threshold_sanity_quality == 0 gdy model NIGDY nie przekracza 0.5
    na bocie (true_positives == 0). Wtedy CAŁY reward = 0."""
    labels = np.asarray(y_true, dtype=int); scores = np.asarray(y_score, dtype=float)
    pos = int(np.sum(labels == 1)); neg = int(np.sum(labels == 0))
    if scores.size == 0:
        return {"hard_bot_recall": 0.0, "hard_fpr": 0.0,
                "positive_prediction_rate": 0.0, "threshold_sanity_quality": 0.0}
    hard = scores >= float(threshold)
    ppr = float(np.mean(hard))
    tp = int(np.sum(hard & (labels == 1)))
    fp = int(np.sum(hard & (labels == 0)))
    hard_recall = tp / max(pos, 1) if pos > 0 else 0.0
    hard_fpr = fp / max(neg, 1) if neg > 0 else 0.0
    if pos <= 0 or neg <= 0:
        tsq = 1.0
    elif tp <= 0:
        tsq = 0.0                       # <-- ZERO REWARDU
    elif hard_fpr <= 0.10:
        tsq = 1.0
    else:
        tsq = max(0.0, 1.0 - (hard_fpr - 0.10) / 0.90)
    return {"hard_bot_recall": float(hard_recall), "hard_fpr": float(hard_fpr),
            "positive_prediction_rate": ppr, "threshold_sanity_quality": float(tsq)}


# wagi 1:1 z oficjalnego scoring.py (commit "Update validator scoring formula")
AP_WEIGHT = 0.35
BOT_RECALL_WEIGHT = 0.30
HUMAN_SAFETY_WEIGHT = 0.20
CALIBRATION_WEIGHT = 0.10
LATENCY_WEIGHT = 0.05


def validator_reward(y_prob, y_true):
    """1:1 z poker44/score/scoring.py::reward (NOWA formuła).
    Ranking to już tylko 0.35 (AP) + 0.30 (bot_recall). Pozostałe 0.30
    zależy od PROGU 0.5 -> kalibracja przestała być neutralna!"""
    scores = np.asarray(y_prob, dtype=float); labels = np.asarray(y_true, dtype=int)
    ap = float(average_precision_score(labels, scores)) if (scores.size and np.any(labels == 1)) else 0.0
    bot_recall, fpr = recall_at_fpr(scores, labels, max_fpr=0.05)
    tm = threshold_metrics(scores, labels, threshold=0.5)
    hsp = tm["threshold_sanity_quality"]
    calib_q = hsp
    latency_q = 1.0
    if hsp <= 0:
        base = 0.0; rew = 0.0
    else:
        base = (AP_WEIGHT * ap + BOT_RECALL_WEIGHT * bot_recall
                + HUMAN_SAFETY_WEIGHT * hsp + CALIBRATION_WEIGHT * calib_q
                + LATENCY_WEIGHT * latency_q)
        rew = float(np.clip(base, 0.0, 1.0))
    return rew, {"ap": ap, "bot_recall": bot_recall, "fpr": fpr,
                 "human_safety_penalty": hsp, **tm}


def daily_stats(test_df, probs):
    g = test_df.copy(); g['_prob'] = probs
    rows = []
    for day, gd in g.groupby('data'):
        rew, m = validator_reward(gd['_prob'].values, gd['label'].values)
        rows.append({'day': day, 'ap': round(m['ap'], 3),
                     'bot_recall': round(m['bot_recall'], 3), 'reward': round(rew, 3)})
    return pd.DataFrame(rows)


# ---------------- selekcja cech ----------------
def select_features(columns, feature_sets, auc_=None, ks_max=None, auc_min=0.50,
                    whitelist=None):
    """Prefiksy zestawów -> opcjonalny filtr KS/AUC. Jeśli podano whitelist
    (jawna lista cech, np. z selekcji po transferze), ma PIERWSZEŃSTWO."""
    cols = [c for c in columns if c not in NON_FEATURES]
    if whitelist is not None:
        wl = set(whitelist)
        chosen = [c for c in cols if c in wl]
        missing = wl - set(chosen)
        if missing:
            print(f"UWAGA: {len(missing)} cech z whitelisty nie ma w danych, "
                  f"np. {sorted(missing)[:3]}")
        return chosen
    chosen = []
    for c in cols:
        pref = next((k for k, p in PREFIXES.items() if c.startswith(p)), 'v1')
        if pref in feature_sets:
            chosen.append(c)
    if ks_max is not None and auc_ is not None:
        ok = set(auc_[(auc_['ks'] < ks_max) & (auc_['auc'] >= auc_min)]['feature'])
        chosen = [c for c in chosen if c in ok]
    return chosen


def top_k_by_importance(Xtr, ytr, feats, k):
    """Ich metoda: top-K po importance (szybki XGB jako sędzia)."""
    m = xgb.XGBClassifier(max_depth=4, n_estimators=120, random_state=0,
                          eval_metric="logloss", **_xgb_device())
    m.fit(Xtr, ytr)
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    return imp.head(k).index.tolist(), imp


# ---------------- KDE monitoring PRD ----------------
def plot_prob_kde(p_calib, out_path, title, low=0.2, high=0.8):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    p = np.asarray(p_calib, dtype=float); p = p[np.isfinite(p)]
    if p.size == 0:
        return
    frac_lo = float(np.mean(p < low)); frac_hi = float(np.mean(p > high))
    fig, ax = plt.subplots(figsize=(9, 4))
    try:
        from scipy.stats import gaussian_kde
        if np.ptp(p) > 1e-9:
            grid = np.linspace(0, 1, 400)
            ax.plot(grid, gaussian_kde(p)(grid), lw=2)
        else:
            raise ValueError
    except Exception:
        ax.hist(p, bins=40, range=(0, 1), density=True, alpha=0.7)
    ax.axvline(low, ls='--', color='green'); ax.axvline(high, ls='--', color='red')
    ax.set_xlim(0, 1)
    ax.set_title(f"{title}\nfrac<{low}={frac_lo:.3f}  frac>{high}={frac_hi:.3f}  "
                 f"balance={min(frac_lo, frac_hi):.3f}")
    import os as _os
    plt.tight_layout(); plt.savefig(out_path, dpi=90); plt.close()


# ---------------- główny trening ----------------
def run(df, calibration_data, auc_=None,
        feature_sets=('v1', 'v2', 'v3', 'v4', 'v5'),
        feature_whitelist=None,      # jawna lista cech (np. core/broad z transferu) — nadpisuje feature_sets
        ks_max=None,                 # np. 0.4 -> filtr stabilności KS (pkt 5)
        auc_min=0.50,
        top_k=None,                  # np. 150 -> ich top-K po importance
        use_qt=False,                # True -> nasz quantile per-zbiór (pkt 7)
        meta='logistic',             # 'logistic' (ich, OOF) | 'mean' (nasz)
        refit_full=False,            # True: artefakt = refit na train+test (metryki zostają z train-only)
        target_live_pos_rate=0.20,   # kalibracja progu: frakcja LIVE chunków > 0.5 (odporna na prior-shift)
        fpr_upper_cap=0.06,          # konserwatywne weto: max dopuszczalny FPR_upper (95% CI, release-boot)
        logit_sharpness=4.0,         # ostrość logit-shiftu (wyższe = ostrzejsze przejście przez 0.5)
        use_calibrator=False,        # ich BlendedQuantileCalibrator (spreader score'ów;
                                     # NEUTRALNY dla rankingowego rewardu — kosmetyka/ubezpieczenie)
        n_folds=5,                   # K-fold TYLKO do OOF meta wewnątrz train
        split_date="2026-06-23",     # time-split (pkt 3)
        model_types=MODEL_TYPES,
        out_dir="fusion_models",
        tag="fusion1"):
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "model"); os.makedirs(models_dir, exist_ok=True)
    analysis_dir = os.path.join(out_dir, "analysis", tag); os.makedirs(analysis_dir, exist_ok=True)

    # --- time split (pkt 3) ---
    train    = df[df['data'] <  split_date]
    test     = df[df['data'] >= split_date]
    calib_fit  = calibration_data[calibration_data['data'] <  split_date]
    calib_eval = calibration_data[calibration_data['data'] >= split_date]

    # --- cechy: prefiksy + opcjonalny KS (pkt 1 + 5) ---
    feats = select_features(df.columns, list(feature_sets), auc_, ks_max, auc_min,
                            whitelist=feature_whitelist)
    sel_info = f"whitelist({len(feature_whitelist)})" if feature_whitelist is not None else f"zestawy={list(feature_sets)} ks_max={ks_max}"
    print(f"{sel_info} -> {len(feats)} cech")
    med = train[feats].median()
    Xtr_df = train[feats].fillna(med)
    ytr = train['label'].values

    # --- opcjonalny quantile per-zbiór (pkt 7) ---
    if use_qt:
        qt_train = QuantileTransformer(output_distribution='normal',
                                       n_quantiles=min(200, len(Xtr_df)), random_state=0)
        Xtr = qt_train.fit_transform(Xtr_df)
        qt_calib = QuantileTransformer(output_distribution='normal',
                                     n_quantiles=min(200, len(calib_fit)), random_state=0)
        qt_calib.fit(calib_fit[feats].fillna(med))
        Xte = qt_train.transform(test[feats].fillna(med))
        Xcalib = qt_calib.transform(calib_eval[feats].fillna(med))
    else:
        qt_calib = FunctionTransformer()          # identyczność, miner2-compatible
        Xtr = Xtr_df.values
        Xte = test[feats].fillna(med).values
        Xcalib = calib_eval[feats].fillna(med).values

    # --- opcjonalne top-K po importance (ich metoda) NA JUŻ przefiltrowanych ---
    importances = None
    if top_k is not None and top_k < len(feats):
        keep, importances = top_k_by_importance(Xtr, ytr, feats, top_k)
        idx = [feats.index(f) for f in keep]
        Xtr = Xtr[:, idx]; Xte = Xte[:, idx]; Xcalib = Xcalib[:, idx]
        feats = keep
        med = med[feats]
        # qt fitowane na pełnym zestawie kolumn nie pasuje po cięciu -> refit
        if use_qt:
            qt_train = QuantileTransformer(output_distribution='normal',
                                           n_quantiles=min(200, len(train)), random_state=0)
            Xtr = qt_train.fit_transform(train[feats].fillna(med))
            qt_calib = QuantileTransformer(output_distribution='normal',
                                         n_quantiles=min(200, len(calib_fit)), random_state=0)
            qt_calib.fit(calib_fit[feats].fillna(med))
            Xte = qt_train.transform(test[feats].fillna(med))
            Xcalib = qt_calib.transform(calib_eval[feats].fillna(med))
        print(f"top_k={top_k} po importance -> {len(feats)} cech")
        importances.to_csv(os.path.join(analysis_dir, "importances.csv"))

    avail = [m for m in model_types
             if not (m == "lgb" and lgb is None)
             and not (m == "cat" and CatBoostClassifier is None)]
    print(f"XGB device: {'cuda' if _CUDA else 'cpu'} | base modele: {avail} | meta={meta}")

    # --- OOF predykcje baz (ich metoda; K-fold TYLKO wewnątrz train) ---
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    oof = np.zeros((len(ytr), len(avail)))
    for j, mtype in enumerate(avail):
        for tr_idx, va_idx in skf.split(Xtr, ytr):
            m = _make_model(mtype)
            m.fit(Xtr[tr_idx], ytr[tr_idx])
            oof[va_idx, j] = m.predict_proba(Xtr[va_idx])[:, 1]

    # --- finalne bazy na całym train ---
    base_models = []
    for mtype in avail:
        m = _make_model(mtype); m.fit(Xtr, ytr)
        base_models.append(m)

    # --- meta: logistic na OOF (ich) albo średnia (nasza) ---
    if meta == 'logistic':
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(oof, ytr)
    else:
        meta_model = MeanMeta().fit(oof, ytr)

    # --- opcjonalny kalibrator (ich spreader): fit na OOF-owych score'ach stacka,
    #     czyli out-of-sample; monotoniczny -> NIE zmienia rankingu ani rewardu ---
    calibrator = None
    if use_calibrator:
        oof_stack_scores = (meta_model.predict_proba(oof)[:, 1]
                            if hasattr(meta_model, 'predict_proba') else oof.mean(axis=1))
        calibrator = BlendedQuantileCalibrator().fit(oof_stack_scores)

    ensemble = StackedEnsemble(base_models=base_models, meta_model=meta_model,
                               calibrator=calibrator, feature_indices=None, score_shift=0.0)

    # --- ewaluacja: sel_score per porcja na time-split test (pkt 2+3) ---
    prob_te = ensemble.predict_proba(Xte)[:, 1]
    day_df = daily_stats(test, prob_te)
    reward_full, m_full = validator_reward(prob_te, test['label'].values)
    sel = day_df['reward'].mean() - 0.5 * day_df['reward'].std()

    # per-member diagnostyka (czy meta bije pojedyncze bazy)
    member_rows = []
    for name, m in zip(avail, base_models):
        p = m.predict_proba(Xte)[:, 1]
        r, mm = validator_reward(p, test['label'].values)
        member_rows.append({'mtype': name, 'reward_full': round(r, 4),
                            'ap': round(mm['ap'], 3)})
    members_df = pd.DataFrame(member_rows)
    members_df.to_csv(os.path.join(analysis_dir, "members.csv"), index=False)

    # --- monitoring PRD (pkt 6) ---
    p_calib = ensemble.predict_proba(Xcalib)[:, 1]
    balance = min(float(np.mean(p_calib < 0.2)), float(np.mean(p_calib > 0.8)))
    plot_prob_kde(p_calib, os.path.join(analysis_dir, "prob_kde.png"),
                  f"{tag} | sets={list(feature_sets)} qt={use_qt} meta={meta}")

    # --- opcjonalny REFIT na train+test do artefaktu produkcyjnego ---
    # Metryki wyżej są z modelu train-only (uczciwe). Gdy refit_full=True,
    # do .joblib idzie NOWY ensemble trenowany na train+test (więcej danych
    # w finalnym ficie; jego jakości nie da się już lokalnie zmierzyć).
    ensemble_to_save = ensemble
    if refit_full:
        full = pd.concat([train, test], axis=0)
        med_full = full[feats].median()
        yfull = full['label'].values
        if use_qt:
            qt_full = QuantileTransformer(output_distribution='normal',
                                          n_quantiles=min(200, len(full)), random_state=0)
            Xfull = qt_full.fit_transform(full[feats].fillna(med_full))
        else:
            Xfull = full[feats].fillna(med_full).values
        oof_f = np.zeros((len(yfull), len(avail)))
        skf_f = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        for j, mtype in enumerate(avail):
            for tr_idx, va_idx in skf_f.split(Xfull, yfull):
                m = _make_model(mtype); m.fit(Xfull[tr_idx], yfull[tr_idx])
                oof_f[va_idx, j] = m.predict_proba(Xfull[va_idx])[:, 1]
        base_full = []
        for mtype in avail:
            m = _make_model(mtype); m.fit(Xfull, yfull)
            base_full.append(m)
        if meta == 'logistic':
            meta_full = LogisticRegression(max_iter=1000); meta_full.fit(oof_f, yfull)
        else:
            meta_full = MeanMeta().fit(oof_f, yfull)
        cal_full = None
        if use_calibrator:
            s = (meta_full.predict_proba(oof_f)[:, 1]
                 if hasattr(meta_full, 'predict_proba') else oof_f.mean(axis=1))
            cal_full = BlendedQuantileCalibrator().fit(s)
        ensemble_to_save = StackedEnsemble(base_models=base_full, meta_model=meta_full,
                                           calibrator=cal_full, feature_indices=None,
                                           score_shift=0.0)
        med = med_full
        print("refit_full: artefakt = ensemble trenowany na train+test "
              f"({len(full)} wierszy); metryki pozostają z wariantu train-only")

    # --- KALIBRACJA PROGU: LOGIT-SHIFT (mapuje thr_live -> 0.5 bez zgniatania) ---
    # s = sigmoid(a * (logit(p) - logit(thr_live))).  Monotoniczne -> AP/recall
    # NIETKNIĘTE. Zamiast liniowego (p+shift), który spłaszcza masę przy 0/1,
    # logit-shift zachowuje kształt rozkładu (rada 3. konsultanta).
    # Wybór thr_live: kandydaci z rozkładu LIVE (prior-shift), kryterium =
    # mean_reward - alpha*P(zero) na oknach o różnych bot_frac + konserwatywny
    # bootstrap FPR po releasach (FPR_upper <= fpr_cap zamiast punktowego 0.10).
    y_te_cal = test['label'].values
    idx_bot = np.flatnonzero(y_te_cal == 1)
    idx_hum = np.flatnonzero(y_te_cal == 0)
    test_dates = test['data'].values
    LOGIT_A = float(logit_sharpness)
    FPR_UPPER_CAP = float(fpr_upper_cap)

    def _logit_shift(p, thr, a=LOGIT_A):
        p = np.clip(p, 1e-6, 1 - 1e-6); thr = min(max(thr, 1e-6), 1 - 1e-6)
        z = a * (np.log(p / (1 - p)) - np.log(thr / (1 - thr)))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def _fpr_upper_release(ps, n_boot=300, seed=0):
        """Górny 95% CI dla hard_fpr, bootstrap po RELEASACH (nie chunkach)."""
        rng = np.random.default_rng(seed)
        uniq = np.unique(test_dates)
        if len(uniq) < 2:
            hard = ps >= 0.5
            return float((hard & (y_te_cal == 0)).sum() / max((y_te_cal == 0).sum(), 1))
        fprs = []
        for _ in range(n_boot):
            days = rng.choice(uniq, len(uniq), replace=True)
            mask = np.isin(test_dates, days)
            yb = y_te_cal[mask]; pb = ps[mask]
            neg = (yb == 0).sum()
            if neg == 0:
                continue
            fprs.append(((pb >= 0.5) & (yb == 0)).sum() / neg)
        return float(np.quantile(fprs, 0.95)) if fprs else 1.0

    def _window_obj(ps, W=40, n_win=400, alpha=3.0, seed=0):
        if len(idx_bot) < 2 or len(idx_hum) < 2:
            r, _ = validator_reward(ps, y_te_cal); return r
        rng = np.random.default_rng(seed); rews = []
        for bf in [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]:
            nb = max(1, min(W - 1, int(round(W * bf))))
            for _ in range(n_win // 6):
                wi = np.concatenate([rng.choice(idx_bot, nb, replace=len(idx_bot) < nb),
                                     rng.choice(idx_hum, W - nb, replace=len(idx_hum) < W - nb)])
                r, _ = validator_reward(ps[wi], y_te_cal[wi]); rews.append(r)
        rews = np.asarray(rews)
        return float(rews.mean() - alpha * (rews == 0).mean())

    if len(p_calib) >= 50:
        base_lpr = target_live_pos_rate if target_live_pos_rate is not None else 0.20
        cands = sorted({base_lpr, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50})
        best_obj, thr_live = -1e9, 0.5
        for lpr in cands:
            thr_c = float(np.quantile(p_calib, 1.0 - lpr))
            ps_c = _logit_shift(prob_te, thr_c)
            fpr_up = _fpr_upper_release(ps_c)
            if fpr_up > FPR_UPPER_CAP:          # konserwatywne weto: odrzuć zbyt ryzykowne
                continue
            obj = _window_obj(ps_c)
            if obj > best_obj:
                best_obj, thr_live = obj, thr_c
        if best_obj <= -1e9:                    # nic nie przeszło weta -> najostrożniejszy
            thr_live = float(np.quantile(p_calib, 1.0 - 0.10))
    else:
        thr_live = 0.5

    # zapamiętaj parametry logit-shiftu (miner musi je odtworzyć)
    score_shift = 0.0                            # legacy: liniowy shift wyłączony
    logit_thr, logit_a = float(thr_live), float(LOGIT_A)

    p_te_cal = _logit_shift(prob_te, logit_thr)
    r_cal, m_cal = validator_reward(p_te_cal, y_te_cal)
    live_pos_after = float((_logit_shift(p_calib, logit_thr) >= 0.5).mean())
    fpr_up = _fpr_upper_release(p_te_cal)
    _pz_bot5 = None
    if len(idx_bot) >= 2 and len(idx_hum) >= 2:
        rng = np.random.default_rng(1); zeros = tot = 0
        for _ in range(500):
            nb = max(1, int(round(40 * 0.07)))
            wi = np.concatenate([rng.choice(idx_bot, nb, replace=len(idx_bot) < nb),
                                 rng.choice(idx_hum, 40 - nb, replace=len(idx_hum) < 40 - nb)])
            r, _ = validator_reward(p_te_cal[wi], y_te_cal[wi]); zeros += (r == 0); tot += 1
        _pz_bot5 = zeros / tot
    print(f"\n  KALIBRACJA (logit-shift): thr_live={logit_thr:.4f} a={logit_a} "
          f"| live frac>0.5: {live_pos_after:.1%}")
    print(f"    reward: {reward_full:.4f} -> {r_cal:.4f} | hsp {m_full['human_safety_penalty']:.2f} -> "
          f"{m_cal['human_safety_penalty']:.2f} | hard_fpr {m_full['hard_fpr']:.3f} -> {m_cal['hard_fpr']:.3f} "
          f"| FPR_upper(95%, release-boot)={fpr_up:.3f}")
    if _pz_bot5 is not None:
        print(f"    P(reward=0) na oknie ubogim w boty (~7%): {_pz_bot5:.1%}")

    # --- artefakt zgodny z miner2 (+ logit-shift do kalibracji progu) ---
    artifact = {'feature': feats, 'median': med, 'qt_calib': qt_calib,
                'model': ensemble_to_save, 'score_shift': score_shift,
                'logit_thr': logit_thr, 'logit_a': logit_a,
                'config': {'feature_sets': list(feature_sets), 'ks_max': ks_max,
                           'whitelist_used': feature_whitelist is not None,
                           'top_k': top_k, 'use_qt': use_qt, 'meta': meta,
                           'use_calibrator': use_calibrator, 'refit_full': refit_full,
                           'split_date': split_date, 'model_types': avail,
                           'target_live_pos_rate': target_live_pos_rate,
                           'logit_thr': logit_thr, 'logit_a': logit_a,
                           'fpr_upper': round(fpr_up, 4)}}
    joblib.dump(artifact, os.path.join(models_dir, f"{tag}.joblib"))

    # --- DIAGNOSTYKA PROGOWA (NOWY reward: 0.30 wagi zależy od progu 0.5) ---
    tm = threshold_metrics(prob_te, test['label'].values, threshold=0.5)
    if tm['threshold_sanity_quality'] <= 0:
        print("\n  *** UWAGA: REWARD = 0! Model nigdy nie przekracza 0.5 na bocie. ***")
        print("      Konieczna kalibracja (use_calibrator=True lub przesuniecie score'ow).")
    elif tm['hard_fpr'] > 0.10:
        print(f"\n  UWAGA: hard_fpr={tm['hard_fpr']:.3f} > 0.10 -> hsp={tm['threshold_sanity_quality']:.3f} "
              f"(tracisz czesc z 0.30 wagi progowej)")
    print(f"  PRÓG 0.5: hard_bot_recall={tm['hard_bot_recall']:.3f} hard_fpr={tm['hard_fpr']:.3f} "
          f"pos_rate={tm['positive_prediction_rate']:.3f} hsp={tm['threshold_sanity_quality']:.3f}")

    # --- ANALIZA PROGU: jaki pos_rate daje hard_fpr <= 0.10? ---
    y_te = test['label'].values
    thr_rows = []
    for pr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        thr_q = float(np.quantile(prob_te, 1 - pr))
        hard = prob_te >= thr_q
        tp = int((hard & (y_te == 1)).sum()); fp = int((hard & (y_te == 0)).sum())
        fpr_q = fp / max((y_te == 0).sum(), 1)
        hsp_q = 1.0 if (tp > 0 and fpr_q <= 0.10) else (
            0.0 if tp == 0 else max(0.0, 1.0 - (fpr_q - 0.10) / 0.90))
        thr_rows.append({'pos_rate': pr, 'threshold': round(thr_q, 4),
                         'hard_fpr': round(fpr_q, 3), 'tp': tp, 'hsp': round(hsp_q, 3)})
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(os.path.join(analysis_dir, "threshold_scan.csv"), index=False)
    ok = thr_df[thr_df['hsp'] >= 1.0]
    print("\n  SKAN PROGU (ranking bez zmian, tylko przesuniecie):")
    print(thr_df.to_string(index=False))
    if len(ok):
        best = ok.iloc[ok['pos_rate'].argmax()]
        print(f"  -> najwyzszy bezpieczny pos_rate={best['pos_rate']:.2f} "
              f"(thr={best['threshold']:.4f}, hard_fpr={best['hard_fpr']:.3f})")
    else:
        print("  -> UWAGA: zaden pos_rate nie daje hsp=1.0 (ranking za slaby)")

    metrics = {'tag': tag, 'feature_sets': '+'.join(feature_sets),
               'n_features': len(feats), 'ks_max': ks_max, 'top_k': top_k,
               'use_qt': use_qt, 'meta': meta,
               'reward_full': round(reward_full, 4),
               'reward_daily': round(day_df['reward'].mean(), 4),
               'reward_daily_std': round(day_df['reward'].std(), 4),
               'sel_score': round(sel, 4),
               'ap': round(m_full['ap'], 3),
               'bot_recall': round(m_full['bot_recall'], 3),
               'balance_eval': round(balance, 3),
               'hard_bot_recall': round(tm['hard_bot_recall'], 3),
               'hard_fpr': round(tm['hard_fpr'], 3),
               'positive_rate': round(tm['positive_prediction_rate'], 3),
               'hsp': round(tm['threshold_sanity_quality'], 3),
               'reward_calibrated': round(r_cal, 4),
               'hsp_calibrated': round(m_cal['human_safety_penalty'], 3),
               'score_shift': round(score_shift, 4),
               '_prob_test': prob_te, '_y_test': y_te, '_prob_calib': p_calib,
               '_threshold_scan': thr_df}
    with open(os.path.join(analysis_dir, "summary.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith('_')}, f, indent=2)
    day_df.to_csv(os.path.join(analysis_dir, "daily.csv"), index=False)

    print("\nCZŁONKOWIE (na tescie):")
    print(members_df.to_string(index=False))
    print(f"\nSTACK: reward_full={metrics['reward_full']} "
          f"reward_daily={metrics['reward_daily']} (std={metrics['reward_daily_std']}) "
          f"sel={metrics['sel_score']} ap={metrics['ap']} balance={balance:.3f}")
    print(f"zapisano: {models_dir}/{tag}.joblib + {analysis_dir}/")
    return metrics


if __name__ == "__main__":
    df   = pd.read_parquet("df.parquet")
    calibration_data = pd.read_parquet("calibration.parquet")
    auc_ = pd.read_csv("auc_.csv") if os.path.exists("auc_.csv") else None
    run(df, calibration_data, auc_)
