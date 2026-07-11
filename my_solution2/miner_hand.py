"""Poker44 miner — HAND-LEVEL model (bag-of-ngrams per ręka, mean pooling).

Artefakt: published/hand_model.joblib
  {'vectorizer', 'lr', 'lgb', 'blend': (w_lgb, w_lr), opcjonalnie 'logit_thr'/'logit_a'}
Score chunka = średnia skalibrowanych prawdopodobieństw jego rąk.
Bez windowingu — średnia po rękach jest z natury niewrażliwa na długość chunka.
"""

import time
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
import numpy as np
import joblib

import json
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "my_solution"))

from hand_tokens import tokenize_hand


def _apply_calibration(p, artifact):
    """Logit-shift (jak w fuzji) — mapuje thr_live na 0.5 bez zmiany rankingu."""
    thr = artifact.get('logit_thr'); a = artifact.get('logit_a')
    if thr is not None and a is not None:
        pc = np.clip(p, 1e-6, 1 - 1e-6); t = min(max(float(thr), 1e-6), 1 - 1e-6)
        z = float(a) * (np.log(pc / (1 - pc)) - np.log(t / (1 - t)))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    return np.clip(p + artifact.get('score_shift', 0.0), 0.0, 1.0)


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🃏 Poker44 Hand-level Miner started")

        model_file = os.getenv("POKER44_MODEL_FILE", "published/hand_model.joblib")
        model_path = ROOT_DIR / model_file
        try:
            self.artifact = joblib.load(model_path)
            self.vec = self.artifact["vectorizer"]
            self.lr = self.artifact["lr"]
            self.lgb = self.artifact["lgb"]
            self.w_lgb, self.w_lr = self.artifact.get("blend", (0.6, 0.4))
            cal = "logit-shift" if self.artifact.get("logit_thr") is not None else "BRAK (surowe p!)"
            bt.logging.info(f"Loaded hand-model from {model_path} | kalibracja: {cal}")
        except Exception as e:
            bt.logging.warning(f"Could NOT load artifact from {model_path}. "
                               f"Predicts will default to 0.5. Error: {e}")
            self.artifact = None

        self.model_manifest = build_local_model_manifest(
            repo_root=ROOT_DIR,
            implementation_files=[
                Path(__file__).resolve(),
                ROOT_DIR / "my_solution" / "hand_tokens.py",
                ROOT_DIR / "poker44" / "validator" / "payload_view.py",
            ],
            defaults={
                "model_name": "miner-hand-ngram",
                "model_version": "1",
                "framework": "tfidf-lr-lgbm",
                "license": "MIT",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"Manifest: status={self.manifest_compliance['status']} "
            f"artifact_sha256={self.model_manifest.get('artifact_sha256','')[:12]}"
        )
        bt.logging.info(f"Axon created: {self.axon}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []

        try:
            out_dir = ROOT_DIR / "saved_synapses"
            out_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = out_dir / f"synapseH_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)
        except Exception as e:
            bt.logging.warning(f"Failed to save synapse to JSON: {e}")

        scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [bool(s >= 0.5) for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        bt.logging.info(f"Scored {len(chunks)} chunks (hand-level, mean pooling).")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def score_chunk(self, chunk: list[dict]) -> float:
        if not chunk or self.artifact is None:
            return 0.5
        try:
            docs = [tokenize_hand(h) for h in chunk]
            X = self.vec.transform(docs)
            p = (self.w_lgb * self.lgb.predict_proba(X)[:, 1]
                 + self.w_lr * self.lr.predict_proba(X)[:, 1])
            p_cal = _apply_calibration(p, self.artifact)
            return self._clamp01(round(float(np.mean(p_cal)), 6))
        except Exception as e:
            bt.logging.error(f"Hand-model prediction error: {e}")
            return 0.5

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Hand-level miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(60)
