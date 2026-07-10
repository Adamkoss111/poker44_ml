"""
compute_robust.py — LICZY robustness cech z danych (benchmark vs live/calibration),
zamiast sztywnej listy. Metryka Travisa: z = |mean_live - mean_bench| / std_bench.

Wysokie z = cecha strukturalnie OOD na live -> drzewa dzielą na progach
benchmarkowych, które collapse'ują. Travis użył progu z>5 ręcznie; tu liczymy
dla KAŻDEJ Twojej cechy automatycznie, więc działa też na v1/v2/v3/v5 i
aktualizuje się z nowymi synapsami.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NON_FEATURES = {"label", "data"}


def robustness_table(df: pd.DataFrame, live: pd.DataFrame,
                     z_max: float = 5.0) -> pd.DataFrame:
    """Dla każdej cechy: z-score OOD benchmark->live + flaga robust.
    df   = benchmark (z etykietami), live = calibration_data (synapsy)."""
    feats = [c for c in df.columns if c not in NON_FEATURES and c in live.columns]
    rows = []
    for c in feats:
        b = pd.to_numeric(df[c], errors="coerce").dropna()
        l = pd.to_numeric(live[c], errors="coerce").dropna()
        if len(b) < 20 or len(l) < 20:
            rows.append({"feature": c, "z_ood": np.nan, "robust": False,
                         "reason": "za mało danych"})
            continue
        sb = b.std()
        if sb < 1e-9:                       # near-stała na benchmarku (jak Travis)
            rows.append({"feature": c, "z_ood": np.inf, "robust": False,
                         "reason": "near-const benchmark"})
            continue
        z = abs(l.mean() - b.mean()) / sb
        rows.append({"feature": c, "z_ood": round(float(z), 2),
                     "robust": z <= z_max,
                     "reason": "OOD" if z > z_max else "ok"})
    return pd.DataFrame(rows).sort_values("z_ood", ascending=False)


def robust_features(df, live, z_max=5.0):
    """Lista cech odpornych (z <= z_max)."""
    t = robustness_table(df, live, z_max)
    return t[t["robust"]]["feature"].tolist()
