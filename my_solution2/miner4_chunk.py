"""Poker44 miner — fusion StackedEnsemble na cechach z pakietu features/."""

import time
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import json
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT_DIR / "my_solution2"
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SOLUTION_DIR))

# pakiet features/ — JEDNO źródło prawdy o cechach (v1..v5, prefiksy w środku)
from features import compute_features
# scentralizowany builder manifestu (repo_url/repo_commit/atestacje -> "transparent")
from manifest_utils import build_manifest


def _window_chunk(chunk, window=40, min_tail=20):
    """Tnie długi chunk live (80-100 rąk) na okna ~40 rąk = reżim treningowy.
    Model trenuje na chunkach 30-40 rąk; cechy CV/std/entropy zależą od długości
    serii, więc długi chunk NASYCA model (wszystko ~0.95, std 0.02 — zmierzone).
    Okna przywracają rozdzielczość (std 0.15). Ogon < min_tail doklejany jest
    pominięty tylko gdy są już inne okna."""
    if len(chunk) <= window + min_tail:
        return [chunk]
    wins = [chunk[i:i + window] for i in range(0, len(chunk), window)]
    if len(wins) > 1 and len(wins[-1]) < min_tail:
        wins = wins[:-1]
    return wins


def _apply_calibration(p, artifact):
    """Kalibracja progu: logit-shift (nowy) lub liniowy score_shift (legacy)."""
    thr = artifact.get('logit_thr'); a = artifact.get('logit_a')
    if thr is not None and a is not None:
        pc = np.clip(p, 1e-6, 1 - 1e-6); t = min(max(float(thr), 1e-6), 1 - 1e-6)
        z = float(a) * (np.log(pc / (1 - pc)) - np.log(t / (1 - t)))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    return np.clip(p + artifact.get('score_shift', 0.0), 0.0, 1.0)


class Miner(BaseMinerNeuron):
    """Miner ładujący artefakt fusion (train_fusion.py):
    {'feature','median','qt_calib','model': StackedEnsemble}.
    Obsługuje też stary format mean-stacka ({'members': [...]}) jako fallback."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 Fusion Miner started")

        model_file = os.getenv("POKER44_MODEL_FILE", "published/model4_chunk.joblib")
        model_path = ROOT_DIR / model_file
        try:
            self.artifact = joblib.load(model_path)
            if "members" in self.artifact:
                self.mode = "members"                       # stary mean-stack
                info = f"{len(self.artifact['members'])} members (stary format)"
            else:
                self.mode = "fusion"                        # StackedEnsemble
                self.model   = self.artifact["model"]
                self.feature = self.artifact["feature"]
                self.median  = self.artifact["median"]
                self.qt_calib = self.artifact["qt_calib"]
                cfg = self.artifact.get("config", {})
                info = (f"fusion: {len(self.feature)} cech, "
                        f"meta={cfg.get('meta')} bazy={cfg.get('model_types')} "
                        f"refit_full={cfg.get('refit_full')}")
            bt.logging.info(f"Loaded artifact from {model_path}: {info}")
        except Exception as e:
            bt.logging.warning(f"Could NOT load artifact from {model_path}. "
                               f"Predicts will default to 0.5. Error: {e}")
            self.artifact = None
            self.mode = None

        self.model_manifest, self.manifest_compliance, self.manifest_digest = build_manifest(
            repo_root=ROOT_DIR,
            model_path=model_path,
            model_name="miner-4-chunk",
            model_version="4",
            framework="stacked-ensemble",
            implementation_files=[
                Path(__file__).resolve(),
                SOLUTION_DIR / "features" / "__init__.py",
                SOLUTION_DIR / "features" / "base.py",
                SOLUTION_DIR / "features" / "temporal.py",
                SOLUTION_DIR / "features" / "literature.py",
                SOLUTION_DIR / "features" / "schema.py",
                SOLUTION_DIR / "features" / "tells.py",
                SOLUTION_DIR / "stacked_top1.py",
                SOLUTION_DIR / "calibration_top1.py",
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
        chunks = synapse.chunks or []

        try:
            out_dir = ROOT_DIR / "saved_synapses"
            out_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = out_dir / f"synapse1_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)
            bt.logging.info(f"Saved incoming synapse chunks to {filename.name}")
        except Exception as e:
            bt.logging.warning(f"Failed to save synapse to JSON: {e}")

        scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [bool(s >= 0.5) for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        bt.logging.info(f"Miner Predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks (mode={self.mode}).")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _predict_fusion(self, feat_df: pd.DataFrame) -> float:
        """StackedEnsemble: dopasuj cechy -> mediana -> qt_calib -> predict_proba."""
        df = feat_df.copy()
        for col in self.feature:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature].fillna(self.median)
        Xn = self.qt_calib.transform(df)
        p = self.model.predict_proba(np.asarray(Xn, dtype=float))[:, 1]
        p_cal = _apply_calibration(p, self.artifact)
        return float(np.mean(p_cal))   # średnia po oknach chunka

    def _predict_members(self, feat_df: pd.DataFrame) -> float:
        """Fallback: stary mean-stack (lista members, każdy z własnym qt)."""
        cols = []
        for member in self.artifact["members"]:
            feat = member["feature"]; med = member["median"]
            df = feat_df.copy()
            for col in feat:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[feat].fillna(med)
            Xn = member["qt_calib"].transform(df) if "qt_calib" in member else member["qt_prd"].transform(df)
            cols.append(member["model"].predict_proba(Xn)[:, 1])
        return float(np.mean(np.column_stack(cols)))   # średnia po oknach i members

    def score_chunk(self, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5
        if self.artifact is None:
            return 0.5

        # 1. cechy — pakiet features/, per OKNO ~40 rąk (reżim treningowy)
        try:
            windows = _window_chunk(chunk)
            rows = [compute_features(w) for w in windows]
            rows = [r for r in rows if r]
            if not rows:
                bt.logging.warning("Empty features for chunk.")
                return 0.5
            df = pd.DataFrame(rows)
        except Exception as e:
            bt.logging.warning(f"Error calculating features: {e}")
            return 0.5

        # 2. predykcja wg formatu artefaktu
        try:
            if self.mode == "fusion":
                p = self._predict_fusion(df)
            else:
                p = self._predict_members(df)
            return self._clamp01(round(p, 6))
        except Exception as e:
            bt.logging.error(f"Prediction error: {e}")
            return 0.5

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Fusion miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(60)
