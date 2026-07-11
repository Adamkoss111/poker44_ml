# prep_hand_data.py — przygotowanie danych PER RĘKA + trening hand-modelu.
# Struktura jak Twoje funkcje notebookowe: process_*_file per plik + ProcessPool.
#
# WAŻNE: hand-model NIE używa compute_features ani train_fusion.run().
# Dane = dokumenty tokenów (hand_tokens.tokenize_hand), trening = TF-IDF -> LR+LGBM.
# hand_tokens.py musi leżeć obok.

import os, json, hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

SYNAPSE_DIR = "../my_solution/saved_synapses"


# ============ 1a. BENCHMARK (etykiety) -> hands_bench.parquet ============
def process_api_file_hands(path):
    """api_data/data_*.json -> wiersze per RĘKA: doc, label, data, chunk_id."""
    from hand_tokens import tokenize_hand
    import json as _json
    rows = []
    data = _json.load(open(path, encoding="utf-8"))
    file_date = "-".join(str(path).split("/")[-1][:-5].split("_")[1:])
    for mi, mc in enumerate(data["data"]["chunks"]):
        chunks, labels = mc.get("chunks") or [], mc.get("groundTruth") or []
        if len(chunks) != len(labels):
            continue
        date = mc.get("windowStart") or mc.get("sourceDate") or file_date
        for ci, (label, chunk) in enumerate(zip(labels, chunks)):
            cid = f"{file_date}#{mi}#{ci}"
            for hi, hand in enumerate(chunk):
                rows.append({"doc": tokenize_hand(hand), "label": int(label),
                             "data": str(date), "chunk_id": cid, "n_hands": len(chunk)})
    return rows


# ============ 1b. PRD (bez etykiet) -> hands_prd.parquet ============
def process_synapse_file_hands(file):
    """saved_synapses -> wiersze per RĘKA: doc, data, chunk_hash.
    Dedupe po haszu chunka rób PO zbudowaniu df (drop_duplicates na chunk_hash
    zostawia... UWAGA: nie dedupuj po wierszach-rękach! patrz main niżej)."""
    from hand_tokens import tokenize_hand
    import hashlib as _hl, json as _json, os as _os
    rows = []
    try:
        with open(_os.path.join(SYNAPSE_DIR, file), encoding="utf-8") as f:
            chunks = _json.load(f)
    except Exception:
        return rows
    date = file.split("_")[1]
    for chunk in chunks:
        h = _hl.md5(_json.dumps(chunk, sort_keys=True).encode()).hexdigest()
        for hi, hand in enumerate(chunk):
            rows.append({"doc": tokenize_hand(hand), "data": date,
                         "chunk_hash": h, "hand_idx": hi, "n_hands": len(chunk)})
    return rows


# ============ 2. TRENING (LR + LGBM na TF-IDF, wagi 1/H, split po chunku) ====
def train_hand_model(hands: pd.DataFrame, split_date="2026-07-05",
                     out="hand_model.joblib"):
    """hands: kolumny doc/label/data/chunk_id/n_hands (z process_api_file_hands)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    import joblib

    tr = hands[hands["data"] < split_date]
    te = hands[hands["data"] >= split_date]
    print(f"rąk: train={len(tr)} test={len(te)} | chunków: "
          f"{tr['chunk_id'].nunique()}/{te['chunk_id'].nunique()}")

    vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3, ngram_range=(1, 1))
    Xtr = vec.fit_transform(tr["doc"]); Xte = vec.transform(te["doc"])
    w_tr = 1.0 / tr["n_hands"].values          # chunk-balanced: kazdy chunk ~1

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xtr, tr["label"], sample_weight=w_tr)
    lgb = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         verbose=-1, random_state=0)
    lgb.fit(Xtr, tr["label"], sample_weight=w_tr)

    # per-hand proby -> chunk = srednia (mean pooling)
    p_hand = 0.6 * lgb.predict_proba(Xte)[:, 1] + 0.4 * lr.predict_proba(Xte)[:, 1]
    te = te.copy(); te["p"] = p_hand
    chunk_p = te.groupby("chunk_id").agg(p=("p", "mean"), label=("label", "first"))

    from sklearn.metrics import average_precision_score, roc_auc_score
    ap = average_precision_score(chunk_p["label"], chunk_p["p"])
    auc = roc_auc_score(chunk_p["label"], chunk_p["p"])
    print(f"CHUNK-LEVEL (mean pooling): AP={ap:.3f} AUC={auc:.3f} "
          f"(porownaj z fuzja: AP~0.92)")

    joblib.dump({"vectorizer": vec, "lr": lr, "lgb": lgb, "blend": (0.6, 0.4)}, out)
    print(f"zapisano {out}")
    return chunk_p


if __name__ == "__main__":
    from pathlib import Path

    # --- benchmark per reka ---
    files = sorted(Path("api_data").glob("data_*.json"))
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in tqdm(ex.map(process_api_file_hands, files), total=len(files)):
            rows.extend(r)
    hands = pd.DataFrame(rows)
    hands.to_parquet("hands_bench.parquet")
    print("benchmark:", hands.shape)

    # --- trening ---
    train_hand_model(hands)

    # --- PRD per reka (do diagnostyki/kalibracji hand-modelu) ---
    # dedupe: NAJPIERW zbierz unikalne chunk_hash z jednego przebiegu, potem
    # (jak w Twoim notebooku) mozesz przefiltrowac df: po zbudowaniu
    #   prd_hands = prd_hands[~prd_hands.duplicated(['chunk_hash','hand_idx'])]
    files = os.listdir(SYNAPSE_DIR)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in tqdm(ex.map(process_synapse_file_hands, files), total=len(files)):
            rows.extend(r)
    prd_hands = pd.DataFrame(rows)
    prd_hands = prd_hands[~prd_hands.duplicated(["chunk_hash", "hand_idx"])]
    prd_hands.to_parquet("hands_prd.parquet")
    print("PRD:", prd_hands.shape)
