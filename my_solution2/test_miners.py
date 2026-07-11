"""Testy offline minerów (miner5 + miner4_chunk) — bez sieci, bez czekania na synaps.

Odpalają PRAWDZIWY kod scoringu (Miner.score_chunk / _predict_fusion /
_window_chunk / _apply_calibration) na realnych zapisanych synapsach, omijając
tylko sieciowy BaseMinerNeuron.__init__ (instancja tworzona przez __new__ +
ręczne załadowanie artefaktu — dokładnie jak w __init__ minera).

Uruchomienie (z roota repo):
    python -m pytest my_solution2/test_miners.py -v
    # albo:
    python -m unittest my_solution2/test_miners.py
"""
from __future__ import annotations

import glob
import json
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # poker44.* musi być importowalny NAJPIERW
sys.path.insert(0, str(ROOT / "my_solution2"))   # features / manifest_utils / stacked_top1

import importlib
import joblib

miner5_mod = importlib.import_module("neurons.miner5")
miner4_mod = importlib.import_module("neurons.miner4_chunk")


# --- lokalizacja artefaktów: published/ (runtime) -> fusion_models/model (źródło) ---
def _resolve_model(published_name: str, source_name: str) -> Path | None:
    for cand in (ROOT / "published" / published_name,
                 ROOT / "my_solution2" / "fusion_models" / "model" / source_name):
        if cand.exists():
            return cand
    return None


MODEL5 = _resolve_model("model5.joblib", "S1_core.joblib")
MODEL4 = _resolve_model("model4_chunk.joblib", "S1_core_chunk.joblib")


def _load_sample_chunks(max_chunks: int = 15) -> list[list[dict]]:
    """Pierwszy sensowny plik synapsy = lista chunków (lista rąk)."""
    for f in sorted(glob.glob(str(ROOT / "my_solution" / "saved_synapses" / "*.json"))):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list) and data and isinstance(data[0], list) and data[0]:
            return [c for c in data if isinstance(c, list) and c][:max_chunks]
    return []


CHUNKS = _load_sample_chunks()


def _bare_miner(MinerClass, artifact_path: Path):
    """Instancja minera BEZ sieci: __new__ + załadowanie artefaktu jak w __init__."""
    m = MinerClass.__new__(MinerClass)
    art = joblib.load(artifact_path)
    m.artifact = art
    if "members" in art:
        m.mode = "members"
    else:
        m.mode = "fusion"
        m.model = art["model"]
        m.feature = art["feature"]
        m.median = art["median"]
        m.qt_calib = art["qt_calib"]
    return m


class _MinerTestBase:
    """Wspólny zestaw testów; podklasy ustawiają MINER_CLASS + MODEL_PATH."""
    MINER_CLASS = None
    MODEL_PATH = None

    @classmethod
    def setUpClass(cls):
        if cls.MODEL_PATH is None:
            raise unittest.SkipTest(f"Brak artefaktu dla {cls.__name__}")
        if not CHUNKS:
            raise unittest.SkipTest("Brak przykładowych synaps w my_solution/saved_synapses/")
        cls.miner = _bare_miner(cls.MINER_CLASS, cls.MODEL_PATH)
        cls.scores = [cls.miner.score_chunk(c) for c in CHUNKS]

    def test_artifact_loaded(self):
        self.assertIsNotNone(self.miner.artifact, "artefakt nie załadowany")
        self.assertEqual(self.miner.mode, "fusion")

    def test_scores_in_unit_interval(self):
        for s in self.scores:
            self.assertIsInstance(s, float)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_not_degenerate(self):
        # model faktycznie różnicuje: nie same 0.5 i nie jedna stała wartość
        self.assertGreater(len(set(round(s, 4) for s in self.scores)), 1,
                           "wszystkie score identyczne — model prawdopodobnie nie działa")
        self.assertFalse(all(abs(s - 0.5) < 1e-9 for s in self.scores),
                         "same 0.5 — artefakt nie załadowany / wyjątek w predykcji")

    def test_empty_chunk_returns_half(self):
        self.assertEqual(self.miner.score_chunk([]), 0.5)

    def test_malformed_chunk_graceful(self):
        # śmieciowy chunk nie może wywalić procesu — score_chunk łapie wyjątki
        s = self.miner.score_chunk([{"garbage": 1}, {"x": None}])
        self.assertIsInstance(s, float)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_forward_contract_one_score_per_chunk(self):
        # walidator odrzuca odpowiedź, gdy liczba score != liczba chunków
        scores = [self.miner.score_chunk(c) for c in CHUNKS]
        preds = [bool(s >= 0.5) for s in scores]
        self.assertEqual(len(scores), len(CHUNKS))
        self.assertEqual(len(preds), len(CHUNKS))


class TestMiner5(_MinerTestBase, unittest.TestCase):
    MINER_CLASS = miner5_mod.Miner
    MODEL_PATH = MODEL5


class TestMiner4Chunk(_MinerTestBase, unittest.TestCase):
    MINER_CLASS = miner4_mod.Miner
    MODEL_PATH = MODEL4

    def test_window_chunk_splits_long(self):
        # długi chunk (>window+min_tail) tnie się na okna ~40; krótki -> 1 okno
        long_chunk = [{"h": i} for i in range(100)]
        short_chunk = [{"h": i} for i in range(30)]
        self.assertGreater(len(miner4_mod._window_chunk(long_chunk)), 1)
        self.assertEqual(len(miner4_mod._window_chunk(short_chunk)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
