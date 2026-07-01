"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

import time
from collections import Counter
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import json
from datetime import datetime

# Dodajemy folder główny do PATH, aby bez problemu zaimportować create_features2
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "my_solution"))

from create_features3 import extract_chunk_features
from create_features4 import extract_chunk_features_v2
from create_features5 import extract_chunk_features_v3
from create_features6 import extract_chunk_features_v4
from manifest_utils import build_manifest


def _predict_stack(stack_artifact, prd_df):
    """Uśredniona predykcja stacka. Każdy member ma własny feature/median/qt_prd.
    (Kopia predict_stack z train_stacked.py — wklejona, żeby miner nie zależał
    od importu skryptu treningowego.)"""
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
    


class Miner(BaseMinerNeuron):
    """
    Miner wykorzystujący model XGBoost (.pkl) na cechach z create_features2.py.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 Stacked Miner started")

        model_file = os.getenv("POKER44_MODEL_FILE", "published/stack3.joblib")
        model_path = ROOT_DIR / model_file
        try:
            self.stack = joblib.load(model_path)
            n_members = len(self.stack.get("members", []))
            types = self.stack.get("model_types", [])
            bt.logging.info(f"Loaded stack from {model_path}: {n_members} members {types}")
        except Exception as e:
            bt.logging.warning(f"Could NOT load stack from {model_path}. "
                               f"Predicts default to 0.5. Error: {e}")
            self.stack = None
        
        repo_root = Path(__file__).resolve().parents[1]
        self.model_manifest, self.manifest_compliance, self.manifest_digest = build_manifest(
            repo_root=ROOT_DIR,
            model_path=model_path,
            model_name="miner-23-stack",
            model_version="23",
            framework="stacked-xgboost",
            implementation_files=[
                Path(__file__).resolve(),
                ROOT_DIR / "my_solution" / "create_features3.py",
                ROOT_DIR / "my_solution" / "create_features4.py",
                ROOT_DIR / "my_solution" / "create_features5.py",
                ROOT_DIR / "my_solution" / "create_features6.py",
                ROOT_DIR / "poker44" / "validator" / "payload_view.py",
            ],
        )
        bt.logging.info(
            f"Manifest: status={self.manifest_compliance['status']} "
            f"repo_commit={self.model_manifest.get('repo_commit','')[:12]} "
            f"artifact_sha256={self.model_manifest.get('artifact_sha256','')[:12]} "
            f"violations={self.manifest_compliance.get('policy_violations')}"
        )

        bt.logging.info(f"Axon created: {self.axon}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Przypisz score predykcyjny bota na podstawie XGBoost dla każdego chunku."""
        chunks = synapse.chunks or []
        
        # Zapis synapsy z datą i godziną (co do sekundy)
        try:
            # Tworzymy folder 'saved_synapses' w folderze głównym projektu
            out_dir = ROOT_DIR / "saved_synapses"
            out_dir.mkdir(exist_ok=True)
            
            # Format: 'synapse_20260308_190530.json' (Z RRRRMMDD_GGMMSS)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = out_dir / f"synapse1_{timestamp}.json"
            
          #  with open(filename, "w", encoding="utf-8") as f:
           #     json.dump(chunks, f, indent=2)
            bt.logging.info(f"Saved incoming synapse chunks to {filename.name}")
        except Exception as e:
            bt.logging.warning(f"Failed to save synapse to JSON: {e}")

        scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [bool(s >= 0.5) for s in scores]
        
        # Manifest added to the prediction/synapse
        synapse.model_manifest = dict(self.model_manifest)
        
        bt.logging.info(f"Miner Predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with XGBoost model.")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def score_chunk(self, chunk: list[dict]) -> float:
        if not chunk:
            return 0.0
        if self.stack is None:
            return 0.5   # brak modelu -> neutralnie (nie 0.0!)

        # 1. Ekstrakcja cech: v1 + v2 + v3
        try:
            f1 = extract_chunk_features(chunk) or {}
            f2 = extract_chunk_features_v2(chunk) or {}
            f3 = extract_chunk_features_v3(chunk) or {}
            f4 = extract_chunk_features_v4(chunk) or {}
            
            f2 = {"new_" + key: val for key, val in f2.items()}
            f3 = {"new2_" + key: val for key, val in f3.items()}
            f4 = {"new3_" + key: val for key, val in f4.items()}
            
            features = {**f1, **f2, **f3, **f4}
            if not features:
                bt.logging.warning("Empty features for chunk.")
                return 0.5
            df = pd.DataFrame([features])
        except Exception as e:
            bt.logging.warning(f"Error calculating features: {e}")
            return 0.5

        try:
            p = _predict_stack(self.stack, df)
            return self._clamp01(round(float(p[0]), 6))
        except Exception as e:
            bt.logging.error(f"Stack prediction error: {e}")
            return 0.5

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(60)
