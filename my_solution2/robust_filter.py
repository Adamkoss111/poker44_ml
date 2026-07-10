"""
robust_filter.py — filtr cech odpornych na rozjazd benchmark -> live validator.

Reguły przeniesione z Travis861/Poker44_v1 (training/robust_features.py), autora
z DOSTĘPEM do żywych chunków validatora. Zweryfikowane przez niego 2026-07-01
na 300 live chunkach vs 724 batche benchmarku. Kluczowe fakty o live, których
nie widać z samego benchmarku:

  * SKALA (_bb): live poty/bety ~POŁOWA benchmarku -> drzewa dzielą na progach
    benchmarkowych, które collapse'ują na live. Wywalamy WSZYSTKIE absolutne
    wielkości w BB.
  * ROZMIAR STOŁU/CHUNKA: live ma do 9 graczy (benchmark cap 6) i 80-100 rąk
    na chunk (benchmark 30-40). To przesuwa player_count, hand_count i KAŻDĄ
    statystykę porządkową. Wywalamy cechy zależne od rozmiaru.
  * PASYWNOŚĆ: live preflop call/check rate 10-18x benchmark. Cechy oparte na
    call/check rate to fikcja benchmarkowa. Wywalamy je.
  * BUTTON: button_seat = 0 na benchmarku I live -> cechy button to martwe stałe.

Travis potwierdził: usunięcie tego zbioru NIE obniża benchmark AP
(0.8830 -> 0.8842), a usuwa splity najbardziej podatne na collapse na live.

To jest TRZECI, niezależny filtr — obok Twojego transferu (empiryczny, z Twoich
synapsów) i KS. Cecha odrzucona przez transfer ORAZ robust = na pewno leci.

Użycie:
    from robust_filter import is_robust, filter_robust, robust_report
    robust_cols = filter_robust(df.columns)
    print(robust_report(df.columns))
"""

from __future__ import annotations

from typing import Iterable, Sequence

# --- podłańcuchy wskazujące cechę kruchą na live (niezależne od Twoich prefiksów) ---
_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    # SKALA: absolutne wielkości w big-blindach (live ~połowa benchmarku)
    "_bb",
    # PASYWNOŚĆ: rate'y call/check (live 10-18x benchmark)
    "passive_share",
    "call_share",
    "check_share",
    "call_to_share",
    # ROZMIAR STOŁU/CHUNKA (live: 9 graczy / 80-100 rąk vs benchmark 6 / 30-40)
    "player_count",
    "hand_count",
    "seat_utilization",
    "frac_headsup",
    "num_players",
    "n_players",
    # BUTTON: martwe stałe (button_seat=0 wszędzie)
    "button_action",
    "hero_button_same",
    "button_seat",
    "seats_from_btn",
    # n-gramy pasywne (pCs/pK0 tokens — pasywność jw.)
    "ngram_pcs",
    "ngram_pk0",
    "ngram_fcs",
    "ngram_fbs",
    "ngram_tcs",
    "ngram_rcs",
)

# --- dokładne nazwy z Twojego zestawu, które mimo braku podłańcucha są kruche ---
# (near-stałe na benchmarku albo strukturalnie OOD; rozszerzaj po analizie transferu)
_EXCLUDE_EXACT: frozenset[str] = frozenset({
    # przykłady z Twojej tabeli transferu o transfer<0 (fikcje benchmarkowe):
    "fold_cv_time",
    "new_vpip_runs_z_mean",
    "new_hero_seat_entropy",
    "new_hero_seat_mode_share",
})


def is_robust(name: str) -> bool:
    """True gdy cecha jest bezpieczna dla generalizacji na live validatora."""
    n = str(name).strip().lower()
    if not n:
        return False
    if n in _EXCLUDE_EXACT:
        return False
    if any(tok in n for tok in _EXCLUDE_SUBSTRINGS):
        return False
    return True


def filter_robust(names: Sequence[str]) -> list[str]:
    """Zwraca tylko cechy odporne na live (kolejność zachowana)."""
    return [n for n in names if is_robust(n)]


def robust_report(names: Sequence[str]) -> dict:
    """Podsumowanie: ile zostaje, ile leci, z jakiego powodu."""
    kept, dropped = [], []
    reason: dict[str, str] = {}
    for n in names:
        low = str(n).strip().lower()
        if low in _EXCLUDE_EXACT:
            dropped.append(n); reason[n] = "exact"
        else:
            hit = next((t for t in _EXCLUDE_SUBSTRINGS if t in low), None)
            if hit:
                dropped.append(n); reason[n] = hit
            else:
                kept.append(n)
    from collections import Counter
    by_reason = Counter(reason.values())
    return {
        "total": len(names),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_by_reason": dict(by_reason.most_common()),
    }
