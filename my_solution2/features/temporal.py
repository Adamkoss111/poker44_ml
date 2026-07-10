"""
Chunk-level feature engineering v2 for Poker44 bot detection.

Design assumption: amount/pot fields are noised by the sanitizer and
outcome/cards are removed, so v2 features rely only on signals that
survive sanitization:

  A. Stack continuity across consecutive hands (generator artifacts)
  B. Action-sequence structure (n-grams, Markov conditional entropy)
  C. Position proxy from preflop action ORDER (button is hidden)
  D. Temporal i.i.d.-ness (autocorrelation, dispersion, runs test)
  E. Scale-invariant within-hand sizing RATIOS (robust to per-hand
     multiplicative noise)
  F. Table composition dynamics (seat turnover, hero_seat, walks)

Usage:
    from create_features4 import extract_chunk_features_v2

    feats = extract_chunk_features_v2(chunk)          # standalone
    feats = {**extract_chunk_features(chunk),         # or merged with v1
             **extract_chunk_features_v2(chunk)}
"""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np
import pandas as pd

VOLUNTARY   = {"call", "raise", "bet", "all_in"}
AGG         = {"bet", "raise", "all_in"}
BLIND_TYPES = {"small_blind", "big_blind", "ante"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _m(arr) -> float:
    return round(float(np.mean(arr)), 4) if len(arr) else np.nan


def _s(arr) -> float:
    return round(float(np.std(arr)), 4) if len(arr) else np.nan


def _safe_cv(arr) -> float:
    if len(arr) < 2:
        return np.nan
    m = float(np.mean(arr))
    if m == 0:
        return np.nan
    return round(float(np.std(arr)) / m, 4)


def _entropy_from_counts(counts) -> float:
    counts = np.asarray(list(counts), dtype=float)
    if counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    return round(float(-np.sum(p * np.log2(p + 1e-12))), 4)


def _lag1_autocorr(x: List[float]) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 5 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return np.nan
    return round(float(np.corrcoef(x[:-1], x[1:])[0, 1]), 4)


def _runs_test_z(binary: List[int]) -> float:
    """Wald-Wolfowitz runs test z-score. i.i.d. sequence -> z ~ N(0,1).
    Humans (streaky / adaptive) tend to produce |z| > 0."""
    x = np.asarray(binary, dtype=int)
    n1, n0 = int(x.sum()), int(len(x) - x.sum())
    if n1 < 3 or n0 < 3:
        return np.nan
    runs = 1 + int(np.sum(x[1:] != x[:-1]))
    n = n1 + n0
    mu = 2.0 * n1 * n0 / n + 1.0
    var = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n) / (n ** 2 * (n - 1.0))
    if var <= 0:
        return np.nan
    return round(float((runs - mu) / np.sqrt(var)), 4)


def _nonblind(actions: List[dict]) -> List[dict]:
    return [a for a in actions if a.get("action_type") not in BLIND_TYPES]


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_chunk_features_v2(chunk: List[dict]) -> Optional[dict]:
    if not chunk:
        return None

    n_hands = len(chunk)
    out: dict = {}

    # =======================================================================
    # A. STACK CONTINUITY
    # In a real continuous session, seat stacks in hand k+1 follow from
    # hand k. A synthetic generator may reset/redraw stacks differently.
    # =======================================================================
    seat_stack_seq: Dict[int, List[tuple]] = defaultdict(list)  # seat -> [(hand_idx, stack)]
    all_stacks: List[float] = []
    for idx, hand in enumerate(chunk):
        for p in hand.get("players") or []:
            seat, st = p.get("seat"), p.get("starting_stack")
            if seat is not None and st is not None:
                seat_stack_seq[seat].append((idx, float(st)))
                all_stacks.append(float(st))

    if all_stacks:
        cap = max(all_stacks)  # buy-in cap, e.g. 4.8
        out["stack_at_cap_rate"] = _m([abs(s - cap) < 1e-9 for s in all_stacks])

        deltas_adj, zero_adj, reset_after_loss = [], [], []
        for seq in seat_stack_seq.values():
            for (i0, s0), (i1, s1) in zip(seq, seq[1:]):
                if i1 - i0 == 1:  # consecutive hands only
                    d = s1 - s0
                    deltas_adj.append(d)
                    zero_adj.append(abs(d) < 1e-9)
                    reset_after_loss.append(
                        abs(s1 - cap) < 1e-9 and s0 < cap - 1e-9
                    )
        out["stack_delta_abs_mean"]  = _m([abs(d) for d in deltas_adj])
        out["stack_delta_std"]       = _s(deltas_adj)
        out["stack_delta_zero_rate"] = _m(zero_adj)            # frozen stacks
        out["stack_topup_rate"]      = _m(reset_after_loss)    # auto top-up to cap
        out["stack_gain_rate"]       = _m([d > 1e-9 for d in deltas_adj])
        # how granular are the stack values
        out["stack_unique_ratio"] = round(len(set(all_stacks)) / len(all_stacks), 4)
    else:
        for k in ("stack_at_cap_rate", "stack_delta_abs_mean", "stack_delta_std",
                  "stack_delta_zero_rate", "stack_topup_rate", "stack_gain_rate",
                  "stack_unique_ratio"):
            out[k] = np.nan

    # =======================================================================
    # B. ACTION-SEQUENCE STRUCTURE (n-grams, Markov)
    # =======================================================================
    bigram_counts: Counter = Counter()
    unigram_counts: Counter = Counter()
    repeat_hits, repeat_total = 0, 0

    for hand in chunk:
        actions = hand.get("actions") or []
        # reset the chain at street boundaries
        by_street: Dict[str, List[str]] = defaultdict(list)
        for a in _nonblind(actions):
            st, at = a.get("street", ""), a.get("action_type", "")
            if st and at:
                by_street[st].append(at)
        for seq in by_street.values():
            for at in seq:
                unigram_counts[at] += 1
            for a0, a1 in zip(seq, seq[1:]):
                bigram_counts[(a0, a1)] += 1
                repeat_total += 1
                if a0 == a1:
                    repeat_hits += 1

    out["action_unigram_entropy"] = _entropy_from_counts(unigram_counts.values())
    out["action_bigram_entropy"]  = _entropy_from_counts(bigram_counts.values())
    out["action_repeat_rate"] = (
        round(repeat_hits / repeat_total, 4) if repeat_total else np.nan
    )
    if bigram_counts:
        total_bi = sum(bigram_counts.values())
        top = bigram_counts.most_common(3)
        out["bigram_top1_share"] = round(top[0][1] / total_bi, 4)
        out["bigram_top3_share"] = round(sum(c for _, c in top) / total_bi, 4)
        # conditional entropy H(next | prev): bots with deterministic
        # policies in common spots -> low; humans -> higher / noisier
        prev_tot: Counter = Counter()
        for (a0, _), c in bigram_counts.items():
            prev_tot[a0] += c
        h_cond = 0.0
        for (a0, _), c in bigram_counts.items():
            p_joint = c / total_bi
            p_cond = c / prev_tot[a0]
            h_cond -= p_joint * np.log2(p_cond + 1e-12)
        out["action_cond_entropy"] = round(float(h_cond), 4)
    else:
        out["bigram_top1_share"] = np.nan
        out["bigram_top3_share"] = np.nan
        out["action_cond_entropy"] = np.nan

    # =======================================================================
    # C. POSITION PROXY FROM PREFLOP ACTION ORDER
    # button_seat is hidden, but the ORDER of preflop actions encodes
    # position. Bots usually have a very regular positional profile.
    # =======================================================================
    pos_fold = defaultdict(lambda: [0, 0])   # order_idx -> [folds, total]
    pos_raise = defaultdict(lambda: [0, 0])
    first_action_counter: Counter = Counter()

    for hand in chunk:
        pre = [a for a in _nonblind(hand.get("actions") or [])
               if a.get("street") == "preflop"]
        if not pre:
            continue
        first_action_counter[pre[0].get("action_type", "")] += 1
        for k, a in enumerate(pre[:6]):
            at = a.get("action_type")
            pos_fold[k][1] += 1
            pos_raise[k][1] += 1
            if at == "fold":
                pos_fold[k][0] += 1
            if at in {"raise", "all_in"}:
                pos_raise[k][0] += 1

    def _rate(d, k):
        f, t = d[k]
        return f / t if t else np.nan

    out["fold_rate_first_to_act"] = round(_rate(pos_fold, 0), 4) if pos_fold[0][1] else np.nan
    out["raise_rate_first_to_act"] = round(_rate(pos_raise, 0), 4) if pos_raise[0][1] else np.nan
    # slope of fold rate vs action order (positional awareness gradient)
    xs, ys = [], []
    for k in range(6):
        r = _rate(pos_fold, k)
        if not np.isnan(r) and pos_fold[k][1] >= 10:
            xs.append(k); ys.append(r)
    out["fold_pos_slope"] = (
        round(float(np.polyfit(xs, ys, 1)[0]), 4) if len(xs) >= 3 else np.nan
    )
    tot_first = sum(first_action_counter.values())
    out["first_action_fold_share"] = (
        round(first_action_counter.get("fold", 0) / tot_first, 4) if tot_first else np.nan
    )

    # =======================================================================
    # D. TEMPORAL i.i.d.-NESS
    # Bots are i.i.d. hand-to-hand; humans tilt, adapt, take breaks.
    # =======================================================================
    hand_agg_series: List[float] = []   # per-hand aggression ratio
    hand_vpip_series: List[float] = []  # per-hand mean VPIP across seats
    seat_vpip_binary: Dict[int, List[int]] = defaultdict(list)

    for hand in chunk:
        actions = _nonblind(hand.get("actions") or [])
        if actions:
            hand_agg_series.append(
                sum(1 for a in actions if a.get("action_type") in AGG) / len(actions)
            )
        vpips = []
        seats = {p.get("seat") for p in (hand.get("players") or []) if p.get("seat")}
        for seat in seats:
            pre = [a for a in actions
                   if a.get("actor_seat") == seat and a.get("street") == "preflop"]
            if pre:
                v = int(any(a.get("action_type") in VOLUNTARY for a in pre))
                vpips.append(v)
                seat_vpip_binary[seat].append(v)
        if vpips:
            hand_vpip_series.append(float(np.mean(vpips)))

    out["agg_lag1_autocorr"]  = _lag1_autocorr(hand_agg_series)
    out["vpip_lag1_autocorr"] = _lag1_autocorr(hand_vpip_series)
    if len(hand_agg_series) >= 10:
        h = len(hand_agg_series) // 2
        out["agg_halves_absdiff"] = round(
            abs(float(np.mean(hand_agg_series[:h]) - np.mean(hand_agg_series[h:]))), 4
        )
    else:
        out["agg_halves_absdiff"] = np.nan

    # Wald-Wolfowitz runs test on per-seat VPIP series
    zs = [_runs_test_z(v) for v in seat_vpip_binary.values() if len(v) >= 12]
    zs = [z for z in zs if not np.isnan(z)]
    out["vpip_runs_z_mean"]     = _m(zs)
    out["vpip_runs_absz_mean"]  = _m([abs(z) for z in zs])

    # Overdispersion: window VPIP variance vs binomial expectation.
    # i.i.d. bot -> ratio ~ 1; context-driven human -> ratio > 1.
    disp = []
    W = 10
    for v in seat_vpip_binary.values():
        if len(v) >= 3 * W:
            p = float(np.mean(v))
            if 0.05 < p < 0.95:
                wins = [float(np.mean(v[i:i + W])) for i in range(0, len(v) - W + 1, W)]
                expected = p * (1 - p) / W
                if expected > 0 and len(wins) >= 3:
                    disp.append(float(np.var(wins)) / expected)
    out["vpip_dispersion_ratio"] = _m(disp)

    # =======================================================================
    # E. SCALE-INVARIANT SIZING RATIOS
    # If the sanitizer noise is ~multiplicative per hand, within-hand
    # ratios of raise_to values keep their meaning.
    # =======================================================================
    reraise_ratios: List[float] = []     # 3bet_to / open_to etc.
    flop_vs_pf_ratios: List[float] = []  # first flop bet / last preflop raise_to
    cap_raise_rate_hits, raise_to_total = 0, 0

    max_stack_global = max(all_stacks) if all_stacks else np.nan

    for hand in chunk:
        actions = hand.get("actions") or []
        pf_raises = [float(a["raise_to"]) for a in actions
                     if a.get("street") == "preflop"
                     and a.get("action_type") in {"raise", "all_in"}
                     and a.get("raise_to")]
        for r0, r1 in zip(pf_raises, pf_raises[1:]):
            if r0 > 0:
                reraise_ratios.append(r1 / r0)
        for a in actions:
            rt = a.get("raise_to")
            if rt:
                raise_to_total += 1
                if all_stacks and float(rt) >= 0.93 * max_stack_global:
                    cap_raise_rate_hits += 1
        if pf_raises:
            flop_bets = [float(a.get("amount") or 0) for a in actions
                         if a.get("street") == "flop"
                         and a.get("action_type") in {"bet", "raise", "all_in"}]
            if flop_bets and flop_bets[0] > 0 and pf_raises[-1] > 0:
                flop_vs_pf_ratios.append(flop_bets[0] / pf_raises[-1])

    out["reraise_ratio_mean"]   = _m(reraise_ratios)
    out["reraise_ratio_std"]    = _s(reraise_ratios)
    out["reraise_ratio_cv"]     = _safe_cv(reraise_ratios)
    out["flop_vs_pf_ratio_mean"] = _m(flop_vs_pf_ratios)
    out["flop_vs_pf_ratio_cv"]   = _safe_cv(flop_vs_pf_ratios)
    out["near_cap_raise_rate"] = (
        round(cap_raise_rate_hits / raise_to_total, 4) if raise_to_total else np.nan
    )

    # =======================================================================
    # F. TABLE COMPOSITION DYNAMICS
    # =======================================================================
    pc_series = [len(h.get("players") or []) for h in chunk]
    out["frac_headsup"]  = _m([pc == 2 for pc in pc_series])
    out["frac_fullring"] = _m([pc >= 5 for pc in pc_series])
    out["table_size_change_rate"] = _m(
        [int(a != b) for a, b in zip(pc_series, pc_series[1:])]
    ) if len(pc_series) > 1 else np.nan

    # seat-set turnover between consecutive hands (Jaccard)
    seat_sets = [
        {p.get("seat") for p in (h.get("players") or []) if p.get("seat")}
        for h in chunk
    ]
    jacc = []
    for s0, s1 in zip(seat_sets, seat_sets[1:]):
        u = s0 | s1
        if u:
            jacc.append(len(s0 & s1) / len(u))
    out["seat_jaccard_mean"] = _m(jacc)
    out["seat_jaccard_min"]  = round(min(jacc), 4) if jacc else np.nan

    # hero_seat distribution
    hero_counts: Counter = Counter()
    for h in chunk:
        hs = (h.get("metadata") or {}).get("hero_seat")
        if hs is not None:
            hero_counts[hs] += 1
    out["hero_seat_entropy"] = _entropy_from_counts(hero_counts.values())
    out["hero_seat_mode_share"] = (
        round(hero_counts.most_common(1)[0][1] / sum(hero_counts.values()), 4)
        if hero_counts else np.nan
    )

    # "walk" rate: no voluntary action in the whole hand
    walks = 0
    for h in chunk:
        if not any(a.get("action_type") in VOLUNTARY
                   for a in _nonblind(h.get("actions") or [])):
            walks += 1
    out["walk_rate"] = round(walks / n_hands, 4)

    # hand length distribution shape (beyond mean/std from v1)
    n_acts = [len(_nonblind(h.get("actions") or [])) for h in chunk]
    out["hand_len_entropy"] = _entropy_from_counts(Counter(n_acts).values())
    out["one_action_hand_rate"] = _m([n <= 1 for n in n_acts])

    return out


# ---------------------------------------------------------------------------
# Streaming JSON readers — never load the whole file into memory
# ---------------------------------------------------------------------------

def _iter_hands(path: Path) -> Generator[dict, None, None]:
    """
    Yield hand dicts one by one from a flat top-level JSON array: [h1, h2, ...]
    Memory usage: O(1) — one object at a time.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    CHUNK = 256 * 1024

    with opener(path, "rt", encoding="utf-8") as f:
        in_string = escape = False
        depth = 0
        collecting = False
        seen_array = False
        buf: list[str] = []

        while True:
            block = f.read(CHUNK)
            if not block:
                break

            for ch in block:
                if collecting:
                    buf.append(ch)

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch == "[":
                    depth += 1
                    seen_array = True
                    continue

                if ch == "{":
                    if seen_array and depth == 1 and not collecting:
                        collecting = True
                        buf = ["{"]
                    depth += 1
                    continue

                if ch == "}":
                    depth -= 1
                    if collecting and depth == 1:
                        try:
                            yield json.loads("".join(buf))
                        except json.JSONDecodeError:
                            pass
                        collecting = False
                        buf = []
                    continue

                if ch == "]":
                    depth -= 1


def _iter_sessions(path: Path) -> Generator[List[dict], None, None]:
    """
    Yield sessions (lists of hands) from a nested top-level JSON array:
    [[h1, h2, ...], [h1, h2, ...], ...]
    Each inner list is yielded as one session = one chunk.
    Memory usage: O(session_size) — one session at a time.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    CHUNK = 256 * 1024

    with opener(path, "rt", encoding="utf-8") as f:
        in_string = escape = False
        depth = 0
        collecting = False
        buf: list[str] = []

        while True:
            block = f.read(CHUNK)
            if not block:
                break

            for ch in block:
                if collecting:
                    buf.append(ch)

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch == "[":
                    depth += 1
                    # depth==2 means we just entered an inner list (a session)
                    if depth == 2 and not collecting:
                        collecting = True
                        buf = ["["]
                    continue

                if ch == "{":
                    depth += 1
                    continue

                if ch == "}":
                    depth -= 1
                    continue

                if ch == "]":
                    depth -= 1
                    if collecting and depth == 1:
                        try:
                            yield json.loads("".join(buf))
                        except json.JSONDecodeError:
                            pass
                        collecting = False
                        buf = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chunk_features(
    path: str | Path,
    label: int,
    chunk_size: int = 100,
    session_mode: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame of v2 chunk-level features from a hand history file.
    Memory usage: O(chunk_size) — only one chunk in memory at a time.

    Analogous to ``create_features3.build_chunk_features`` but uses the v2
    feature set (``extract_chunk_features_v2``).

    Args:
        path:         .json or .json.gz file
        label:        1 = bot, 0 = human
        chunk_size:   hands per chunk — used only when session_mode=False
        session_mode: True  → file is [[h1,h2,...], [h1,h2,...], ...]
                               each inner list becomes one row (bot format)
                      False → file is [h1, h2, h3, ...]
                               hands are grouped into chunk_size (human format)

    Returns:
        DataFrame with one row per chunk + 'label' column

    Example:
        df_bot   = build_chunk_features("bot_hands.json",               label=1, session_mode=True)
        df_human = build_chunk_features("poker_hands_combined.json.gz", label=0, session_mode=False)
        df = pd.concat([df_bot, df_human], ignore_index=True)
    """
    path = Path(path)
    rows: List[dict] = []

    if session_mode:
        # Bot format: [[h1,...,hN], [h1,...,hN], ...]
        # each inner list = one pre-built session → one row
        for session in _iter_sessions(path):
            feats = extract_chunk_features_v2(session)
            if feats:
                feats["label"] = label
                rows.append(feats)
    else:
        # Human format: [h1, h2, h3, ...]
        # group consecutive hands into chunks of chunk_size
        chunk: List[dict] = []
        for hand in _iter_hands(path):
            chunk.append(hand)
            if len(chunk) == chunk_size:
                feats = extract_chunk_features_v2(chunk)
                if feats:
                    feats["label"] = label
                    rows.append(feats)
                chunk = []
        # last incomplete chunk dropped — too few hands for reliable features

    df = pd.DataFrame(rows)
    print(f"{path.name}: {len(df)} chunks (label={label}, session_mode={session_mode})")
    return df
