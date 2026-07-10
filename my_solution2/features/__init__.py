"""
features — pakiet liczenia cech Poker44. JEDNO źródło prawdy.

Struktura (jeden moduł = jedna rodzina cech; kod przeniesiony VERBATIM ze
sprawdzonych create_features3..6 + port tells z repo top-minera):

    features/
    ├── __init__.py    rejestr + publiczne API (ten plik)
    ├── base.py        v1  ''       bazowe, cały stół            (~75 cech)
    ├── temporal.py    v2  'new_'   sekwencje / i.i.d. czasowe    (~39 cech)
    ├── literature.py  v3  'new2_'  GTO / pozycja / sizing        (~22 cechy)
    ├── schema.py      v4  'new3_'  top1: per-hand + signatures   (~293 cechy)
    └── tells.py       v5  'new4_'  hero tells: bot/gto/advanced  (~176 cech)

Prefiksy są NIEZMIENIONE względem starego pipeline'u (new_/new2_/new3_/new4_),
więc istniejące DataFrame'y, tabele auc_ i artefakty modeli pozostają ważne.

Publiczne API:
    from features import compute_features, compute_features_df, ALL_SETS

    feats = compute_features(chunk)                    # dict, wszystkie zestawy
    feats = compute_features(chunk, sets=['v2','v4'])  # wybrane
    df    = compute_features_df(groups)                # DataFrame, wiersz per chunk
    feature_set_of('new4_bt_hero_vpip')  -> 'v5'

Dodanie nowego zestawu = nowy moduł + jedna linia w REGISTRY.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .base import extract_chunk_features
from .temporal import extract_chunk_features_v2
from .literature import extract_chunk_features_v3
from .schema import extract_chunk_features_v4
from .tells import extract_chunk_features_v5

# (nazwa zestawu) -> (prefiks kolumn, funkcja ekstrakcji)
REGISTRY = {
    "v1": ("",      extract_chunk_features),
    "v2": ("new_",  extract_chunk_features_v2),
    "v3": ("new2_", extract_chunk_features_v3),
    "v4": ("new3_", extract_chunk_features_v4),
    "v5": ("new4_", extract_chunk_features_v5),
}

ALL_SETS = list(REGISTRY.keys())


def compute_features(chunk: List[dict],
                     sets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Cechy jednego chunka dla wybranych zestawów (domyślnie wszystkie).
    Zwraca scalony dict z prefiksami. Pusty chunk -> {}."""
    if not chunk:
        return {}
    out: Dict[str, Any] = {}
    for name in (sets or ALL_SETS):
        prefix, fn = REGISTRY[name]
        feats = fn(chunk) or {}
        if prefix:
            feats = {prefix + k: v for k, v in feats.items()}
        out.update(feats)
    return out


def compute_features_df(chunks: Iterable[List[dict]],
                        sets: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Cechy dla listy chunków -> DataFrame (wiersz per chunk)."""
    return pd.DataFrame([compute_features(c, sets) for c in chunks])


def feature_set_of(column: str) -> str:
    """Do którego zestawu należy kolumna (po prefiksie)."""
    for name, (prefix, _) in sorted(REGISTRY.items(),
                                    key=lambda kv: -len(kv[1][0])):
        if prefix and column.startswith(prefix):
            return name
    return "v1"
