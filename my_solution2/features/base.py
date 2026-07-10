"""
Chunk-level feature engineering for Poker44 bot detection.

Unit of analysis: one chunk = N consecutive hands from one table session.
One row in the output DataFrame = one chunk.

All features are computable from sanitized miner-visible data:
  - actor_seat used instead of player_uid
  - hero_seat / hole_cards not used
  - amounts via normalized_amount_bb

Usage:
    from create_features2 import build_chunk_features

    df_bot   = build_chunk_features("bot_hands.json",                       label=1)
    df_human = build_chunk_features("poker_hands_combined.json.gz",         label=0)
    df_more  = build_chunk_features("more_human_hands.json",                label=0)

    import pandas as pd
    df = pd.concat([df_bot, df_human, df_more], ignore_index=True)
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np
import pandas as pd

VOLUNTARY    = {"call", "raise", "bet", "all_in"}
AGG          = {"bet", "raise", "all_in"}
BLIND_TYPES  = {"small_blind", "big_blind", "ante"}
ROUND_FRACS  = (0.25, 0.33, 0.50, 0.66, 0.75, 1.00)
ROUND_FRAC_TOL = 0.04


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_cv(arr: List[float]) -> float:
    if len(arr) < 2:
        return np.nan
    m = float(np.mean(arr))
    if m == 0:
        return np.nan
    return round(float(np.std(arr)) / m, 4)


def _entropy(arr: List[float], bucket: float = 0.5) -> float:
    if not arr:
        return np.nan
    buckets = [round(x / bucket) * bucket for x in arr]
    _, counts = np.unique(buckets, return_counts=True)
    probs = counts / counts.sum()
    return round(float(-np.sum(probs * np.log2(probs + 1e-12))), 4)


# ---------------------------------------------------------------------------
# Streaming JSON reader — never loads the whole file into memory
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
# Per-hand, per-seat signals
# ---------------------------------------------------------------------------

def _hand_per_seat(hand: dict) -> Dict[int, dict]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    seats   = {p["seat"] for p in players if "seat" in p}
    out: Dict[int, dict] = {}

    for seat in seats:
        mine = [a for a in actions if a.get("actor_seat") == seat]
        pre  = [a for a in mine
                if a.get("street") == "preflop"
                and a.get("action_type") not in BLIND_TYPES]

        is_vpip  = any(a["action_type"] in VOLUNTARY for a in pre)
        is_pfr   = any(a["action_type"] in {"raise", "all_in"} for a in pre)
        did_fold = any(a["action_type"] == "fold" for a in mine)

        bet_sizes_bb = [
            float(a.get("normalized_amount_bb") or 0.0)
            for a in mine
            if a.get("action_type") in {"bet", "raise"}
            and float(a.get("normalized_amount_bb") or 0.0) > 0
        ]

        streets_seen = {a["street"] for a in mine if a.get("street")}

        out[seat] = {
            "is_vpip":      is_vpip,
            "is_pfr":       is_pfr,
            "did_fold":     did_fold,
            "bet_sizes_bb": bet_sizes_bb,
            "saw_flop":     "flop"  in streets_seen,
            "saw_river":    "river" in streets_seen,
        }

    return out


# ---------------------------------------------------------------------------
# Chunk-level feature extraction
# ---------------------------------------------------------------------------

def extract_chunk_features(chunk: List[dict]) -> Optional[dict]:
    """
    Compute bot-detection features for one chunk of hands.
    Returns a flat dict of features, or None if chunk is empty.
    Usable directly on miner-received sanitized chunks.
    """
    if not chunk:
        return None

    n_hands = len(chunk)

    meaningful_actions: List[str]   = []
    all_bet_sizes_bb:   List[float] = []
    all_pot_pcts:       List[float] = []
    streets_counts:     List[int]   = []
    players_counts:     List[int]   = []

    seat_vpip: Dict[int, List[int]]   = defaultdict(list)
    seat_pfr:  Dict[int, List[int]]   = defaultdict(list)
    seat_fold: Dict[int, List[int]]   = defaultdict(list)

    # BB-sizing splits (analog to bet_mean_bb but per-street and per-seat)
    bet_sizes_bb_by_street: Dict[str, List[float]] = defaultdict(list)
    seat_bet_sizes_bb: Dict[int, List[float]]      = defaultdict(list)

    # ---- new accumulators ----
    n_3bet_hands       = 0       # ≥2 raises preflop (excl. blinds)
    n_4bet_hands       = 0       # ≥3 raises preflop
    n_limp_hands       = 0       # at least one preflop call before any raise
    n_cbet_opportunity = 0       # preflop aggressor saw the flop
    n_cbet             = 0       # preflop aggressor opened flop with bet/raise
    n_checkraise       = 0       # check then raise by same seat same postflop street
    n_allin_actions    = 0       # action_type == "all_in"
    n_showdown         = 0       # outcome.showdown == True
    actions_per_hand: List[int]  = []
    seat_pot_pcts: Dict[int, List[float]] = defaultdict(list)

    # per-street aggression counts (non-blind only)
    street_action_counts = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
    street_agg_counts    = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}

    for hand in chunk:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        streets_counts.append(len(streets))
        players_counts.append(len(players))
        if outcome.get("showdown"):
            n_showdown += 1

        hand_meaningful = 0
        for a in actions:
            at = a.get("action_type", "")
            if at not in BLIND_TYPES:
                meaningful_actions.append(at)
                hand_meaningful += 1
                street = a.get("street", "")
                if street in street_action_counts:
                    street_action_counts[street] += 1
                    if at in AGG:
                        street_agg_counts[street] += 1
            if at == "all_in":
                n_allin_actions += 1
            if at in {"bet", "raise"}:
                bb_amt = float(a.get("normalized_amount_bb") or 0.0)
                if bb_amt > 0:
                    all_bet_sizes_bb.append(bb_amt)
                    street_name = a.get("street", "")
                    if street_name:
                        bet_sizes_bb_by_street[street_name].append(bb_amt)
                    seat = a.get("actor_seat")
                    if seat:
                        seat_bet_sizes_bb[seat].append(bb_amt)
                    pot_before = float(a.get("pot_before") or 0.0)
                    amount     = float(a.get("amount")     or 0.0)
                    if pot_before > 0:
                        pct = amount / pot_before
                        all_pot_pcts.append(pct)
                        if seat:
                            seat_pot_pcts[seat].append(pct)
        actions_per_hand.append(hand_meaningful)

        # -- preflop dynamics --
        preflop_acts = [a for a in actions
                        if a.get("street") == "preflop"
                        and a.get("action_type") not in BLIND_TYPES]
        raise_count_pf = sum(1 for a in preflop_acts
                             if a.get("action_type") in {"raise", "all_in"})
        if raise_count_pf >= 2:
            n_3bet_hands += 1
        if raise_count_pf >= 3:
            n_4bet_hands += 1

        # Limp: a call appears in preflop BEFORE any raise
        saw_raise = False
        for a in preflop_acts:
            at = a.get("action_type")
            if at in {"raise", "all_in"}:
                saw_raise = True
                break
            if at == "call":
                n_limp_hands += 1
                break  # one limp counts the hand once

        # C-bet: preflop aggressor (last raiser) opens flop with bet/raise
        pf_aggressor_seat = None
        for a in preflop_acts:
            if a.get("action_type") in {"raise", "all_in"}:
                pf_aggressor_seat = a.get("actor_seat")
        if pf_aggressor_seat is not None:
            flop_acts = [a for a in actions if a.get("street") == "flop"]
            if flop_acts:
                n_cbet_opportunity += 1
                # first non-check flop action — if by aggressor and aggressive → cbet
                for a in flop_acts:
                    at = a.get("action_type")
                    if at == "check":
                        continue
                    if at in AGG and a.get("actor_seat") == pf_aggressor_seat:
                        n_cbet += 1
                    break

        # Check-raise: per postflop street, did any seat check then later raise
        for street_name in ("flop", "turn", "river"):
            per_seat_seq: Dict[int, List[str]] = defaultdict(list)
            for a in actions:
                if a.get("street") != street_name:
                    continue
                seat = a.get("actor_seat")
                at = a.get("action_type")
                if seat and at in {"check", "call", "bet", "raise", "all_in", "fold"}:
                    per_seat_seq[seat].append(at)
            for seq in per_seat_seq.values():
                if "check" in seq:
                    idx_check = seq.index("check")
                    if any(x in {"raise", "all_in"} for x in seq[idx_check + 1:]):
                        n_checkraise += 1

        for seat, sig in _hand_per_seat(hand).items():
            seat_vpip[seat].append(int(sig["is_vpip"]))
            seat_pfr[seat].append(int(sig["is_pfr"]))
            seat_fold[seat].append(int(sig["did_fold"]))

    n_m = max(1, len(meaningful_actions))
    fold_rate  = meaningful_actions.count("fold")  / n_m
    call_rate  = meaningful_actions.count("call")  / n_m
    check_rate = meaningful_actions.count("check") / n_m
    raise_rate = (meaningful_actions.count("raise")
                  + meaningful_actions.count("bet")
                  + meaningful_actions.count("all_in")) / n_m

    player_vpips = [float(np.mean(v)) for v in seat_vpip.values() if v]
    player_pfrs  = [float(np.mean(v)) for v in seat_pfr.values()  if v]
    player_folds = [float(np.mean(v)) for v in seat_fold.values() if v]

    # Temporal consistency — split into ~8 windows
    win = max(5, n_hands // 8)
    window_folds: List[float] = []
    window_vpips: List[float] = []

    for i in range(0, n_hands, win):
        w = chunk[i:i + win]
        if len(w) < 3:
            continue
        w_acts: List[str]   = []
        w_vpip: List[float] = []
        for hand in w:
            actions = hand.get("actions") or []
            players = hand.get("players") or []
            for a in actions:
                at = a.get("action_type", "")
                if at not in BLIND_TYPES:
                    w_acts.append(at)
            for p in players:
                seat = p.get("seat")
                if not seat:
                    continue
                pre = [a for a in actions
                       if a.get("actor_seat") == seat
                       and a.get("street") == "preflop"
                       and a.get("action_type") not in BLIND_TYPES]
                if pre:
                    w_vpip.append(int(any(a["action_type"] in VOLUNTARY for a in pre)))

        n_w = max(1, sum(1 for a in w_acts
                         if a in {"call", "check", "bet", "raise", "fold", "all_in"}))
        window_folds.append(w_acts.count("fold") / n_w)
        if w_vpip:
            window_vpips.append(float(np.mean(w_vpip)))

    def _m(arr): return round(float(np.mean(arr)), 4) if arr else np.nan
    def _s(arr): return round(float(np.std(arr)),  4) if arr else np.nan

    # n_distinct_pot_pcts and bet_entropy use pot-percentage buckets (not raw BB).
    # A bot betting a fixed fraction of pot has LOW distinct pct values and LOW entropy,
    # regardless of how varied the absolute BB amounts are across different pot sizes.
    n_distinct_pot_pcts = (
        len(set(round(x / 0.1) * 0.1 for x in all_pot_pcts))
        if all_pot_pcts else np.nan
    )

    # ---- new feature computations ----
    # Sizing concentration: how clumped are pot_pcts into a small set of values?
    if all_pot_pcts:
        buckets = [round(x / 0.1) * 0.1 for x in all_pot_pcts]
        bucket_counts = {}
        for b in buckets:
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
        sorted_counts = sorted(bucket_counts.values(), reverse=True)
        total = float(len(buckets))
        bet_mode_concentration = round(sorted_counts[0] / total, 4)
        bet_top3_concentration = round(sum(sorted_counts[:3]) / total, 4)
        # Round-fraction signal: bots tend to bet at exact common pot fractions.
        round_hits = sum(
            1 for p in all_pot_pcts
            if any(abs(p - f) < ROUND_FRAC_TOL for f in ROUND_FRACS)
        )
        round_frac_rate = round(round_hits / total, 4)
        # Distribution spread (robust to outliers)
        pot_pct_p10 = round(float(np.percentile(all_pot_pcts, 10)), 4)
        pot_pct_p90 = round(float(np.percentile(all_pot_pcts, 90)), 4)
        pot_pct_p90_p10 = round(pot_pct_p90 - pot_pct_p10, 4)
    else:
        bet_mode_concentration = np.nan
        bet_top3_concentration = np.nan
        round_frac_rate = np.nan
        pot_pct_p10 = np.nan
        pot_pct_p90 = np.nan
        pot_pct_p90_p10 = np.nan

    # Per-seat consistency: bots with fixed sizing tiers → low per-seat CV
    seat_cvs = [_safe_cv(v) for v in seat_pot_pcts.values() if len(v) >= 3]
    seat_cvs = [c for c in seat_cvs if isinstance(c, float) and not np.isnan(c)]
    seat_betsize_cv_mean = _m(seat_cvs)
    seat_betsize_cv_min  = round(min(seat_cvs), 4) if seat_cvs else np.nan

    seat_means = [float(np.mean(v)) for v in seat_pot_pcts.values() if len(v) >= 3]
    if len(seat_means) >= 2:
        seat_pot_pct_range       = round(max(seat_means) - min(seat_means), 4)
        seat_pot_pct_std_of_mean = round(float(np.std(seat_means)), 4)
    else:
        seat_pot_pct_range       = np.nan
        seat_pot_pct_std_of_mean = np.nan

    # Per-street aggression frequencies
    def _agg_freq(street: str) -> float:
        c = street_action_counts[street]
        return round(street_agg_counts[street] / c, 4) if c > 0 else np.nan

    agg_freq_preflop = _agg_freq("preflop")
    agg_freq_flop    = _agg_freq("flop")
    agg_freq_turn    = _agg_freq("turn")
    agg_freq_river   = _agg_freq("river")
    street_agg_list  = [v for v in (agg_freq_preflop, agg_freq_flop,
                                    agg_freq_turn, agg_freq_river)
                        if isinstance(v, float) and not np.isnan(v)]
    agg_freq_street_cv = _safe_cv(street_agg_list)

    # Preflop dynamics
    threebet_rate = round(n_3bet_hands / n_hands, 4)
    fourbet_rate  = round(n_4bet_hands / n_hands, 4)
    limp_rate     = round(n_limp_hands / n_hands, 4)

    # Postflop dynamics
    cbet_rate          = round(n_cbet / n_cbet_opportunity, 4) if n_cbet_opportunity else np.nan
    checkraise_per_hand = round(n_checkraise / n_hands, 4)

    # All-in & showdown
    allin_rate    = round(n_allin_actions / max(1, len(meaningful_actions)), 4)
    showdown_rate = round(n_showdown / n_hands, 4)

    # ---- Per-street BB sizing (analog of bet_mean_bb, split by street) ----
    def _street_bet_mean_bb(street: str):
        vals = bet_sizes_bb_by_street.get(street) or []
        return _m(vals)

    bet_mean_bb_preflop = _street_bet_mean_bb("preflop")
    bet_mean_bb_flop    = _street_bet_mean_bb("flop")
    bet_mean_bb_turn    = _street_bet_mean_bb("turn")
    bet_mean_bb_river   = _street_bet_mean_bb("river")
    street_means_bb = [v for v in (bet_mean_bb_preflop, bet_mean_bb_flop,
                                    bet_mean_bb_turn, bet_mean_bb_river)
                       if isinstance(v, float) and not np.isnan(v)]
    bet_mean_bb_street_cv = _safe_cv(street_means_bb)

    # ---- BB sizing distribution stats (analog of pot_pct_p10/p90 but in BB) ----
    if all_bet_sizes_bb:
        bet_median_bb = round(float(np.percentile(all_bet_sizes_bb, 50)), 4)
        bet_p10_bb    = round(float(np.percentile(all_bet_sizes_bb, 10)), 4)
        bet_p90_bb    = round(float(np.percentile(all_bet_sizes_bb, 90)), 4)
        bet_iqr_bb    = round(float(np.percentile(all_bet_sizes_bb, 75)
                                    - np.percentile(all_bet_sizes_bb, 25)), 4)
        bet_max_bb    = round(float(max(all_bet_sizes_bb)), 4)
        # Entropy of bet sizes bucketed at 1 BB — bot with rigid sizing → low entropy.
        bet_bb_entropy = _entropy(all_bet_sizes_bb, bucket=1.0)
    else:
        bet_median_bb = bet_p10_bb = bet_p90_bb = bet_iqr_bb = bet_max_bb = np.nan
        bet_bb_entropy = np.nan

    # ---- Round-BB rate (analog of round_frac_rate but on BB scale) ----
    # Humans often bet at near-integer BB (e.g. exact 3 BB open, 10 BB 3-bet).
    # Bots with proportional sizing + random jitter rarely land on integers.
    if all_bet_sizes_bb:
        near_int = sum(1 for v in all_bet_sizes_bb
                       if abs(v - round(v)) < 0.15)
        near_integer_bb_rate = round(near_int / len(all_bet_sizes_bb), 4)
        # Hit rate near common preflop open sizes
        common_bb_targets = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0)
        near_common = sum(
            1 for v in all_bet_sizes_bb
            if any(abs(v - t) < 0.30 for t in common_bb_targets)
        )
        common_bb_rate = round(near_common / len(all_bet_sizes_bb), 4)

        # ---- Extensions of the near_integer_bb signal ----
        # 1. Continuous version — mean distance from each bet to nearest integer BB.
        #    Bot with proportional jitter: ~0.25 on average. Human: << 0.1.
        dist_to_int = [abs(v - round(v)) for v in all_bet_sizes_bb]
        mean_distance_to_int_bb   = round(float(np.mean(dist_to_int)), 4)
        median_distance_to_int_bb = round(float(np.median(dist_to_int)), 4)

        # 2. Near half-BB rate — also catches 2.5, 3.5, 5.5 etc. (some sites use 0.5-BB grids).
        near_half = sum(1 for v in all_bet_sizes_bb
                        if abs(v - round(v * 2) / 2) < 0.10)
        near_half_bb_rate = round(near_half / len(all_bet_sizes_bb), 4)

        # 3. Decimal-part entropy: distribution of fractional parts.
        #    Human clicks presets → decimals cluster at 0.0/0.5 → low entropy.
        #    Bot random jitter → decimals roughly uniform → high entropy (~log2(10)=3.32).
        decimals = [v - int(v) for v in all_bet_sizes_bb]
        bet_decimal_entropy = _entropy(decimals, bucket=0.1)

        # 4. Per-street near_integer rate — preflop is strongest (humans love 3 BB opens).
        def _near_int_rate(vals):
            return (round(sum(1 for v in vals if abs(v - round(v)) < 0.15) / len(vals), 4)
                    if vals else np.nan)
        near_integer_bb_rate_preflop = _near_int_rate(bet_sizes_bb_by_street.get("preflop") or [])
        post_bb = ((bet_sizes_bb_by_street.get("flop")  or [])
                 + (bet_sizes_bb_by_street.get("turn")  or [])
                 + (bet_sizes_bb_by_street.get("river") or []))
        near_integer_bb_rate_postflop = _near_int_rate(post_bb)

        # 5. Per-seat near_integer rate — bot personality means whole seats look mechanical.
        per_seat_near = []
        for vals in seat_bet_sizes_bb.values():
            if len(vals) >= 3:
                per_seat_near.append(
                    sum(1 for v in vals if abs(v - round(v)) < 0.15) / len(vals)
                )
        if per_seat_near:
            seat_near_integer_bb_min   = round(min(per_seat_near), 4)
            seat_near_integer_bb_max   = round(max(per_seat_near), 4)
            seat_near_integer_bb_range = round(max(per_seat_near) - min(per_seat_near), 4)
        else:
            seat_near_integer_bb_min   = np.nan
            seat_near_integer_bb_max   = np.nan
            seat_near_integer_bb_range = np.nan
    else:
        near_integer_bb_rate         = np.nan
        common_bb_rate               = np.nan
        mean_distance_to_int_bb      = np.nan
        median_distance_to_int_bb    = np.nan
        near_half_bb_rate            = np.nan
        bet_decimal_entropy          = np.nan
        near_integer_bb_rate_preflop = np.nan
        near_integer_bb_rate_postflop = np.nan
        seat_near_integer_bb_min     = np.nan
        seat_near_integer_bb_max     = np.nan
        seat_near_integer_bb_range   = np.nan

    # ---- Per-seat BB sizing consistency (analog of seat_pot_pct_range in BB) ----
    seat_bb_means = [float(np.mean(v)) for v in seat_bet_sizes_bb.values()
                     if len(v) >= 3]
    if len(seat_bb_means) >= 2:
        seat_bet_mean_bb_range       = round(max(seat_bb_means) - min(seat_bb_means), 4)
        seat_bet_mean_bb_std_of_mean = round(float(np.std(seat_bb_means)), 4)
    else:
        seat_bet_mean_bb_range       = np.nan
        seat_bet_mean_bb_std_of_mean = np.nan
    seat_bb_cvs = [_safe_cv(v) for v in seat_bet_sizes_bb.values() if len(v) >= 3]
    seat_bb_cvs = [c for c in seat_bb_cvs if isinstance(c, float) and not np.isnan(c)]
    seat_bet_bb_cv_mean = _m(seat_bb_cvs)
    seat_bet_bb_cv_min  = round(min(seat_bb_cvs), 4) if seat_bb_cvs else np.nan

    return {
        "n_hands":              n_hands,
        # action distribution
        "fold_rate":            round(fold_rate,  4),
        "call_rate":            round(call_rate,  4),
        "raise_rate":           round(raise_rate, 4),
        "check_rate":           round(check_rate, 4),
        # hand structure
        "avg_streets":          _m(streets_counts),
        "avg_players":          _m(players_counts),
        "preflop_only_rate":    round(sum(1 for s in streets_counts if s == 0) / n_hands, 4),
        "river_rate":           round(sum(1 for s in streets_counts if s >= 3) / n_hands, 4),
        # bet sizing in BB (descriptive only — use pot_pct_* for bot detection)
        "bet_mean_bb":          _m(all_bet_sizes_bb),
        "bet_std_bb":           _s(all_bet_sizes_bb),
        "bet_cv":               _safe_cv(all_bet_sizes_bb),
        # pot-percentage features — low CV/entropy/distinct = bot signal
        "bet_entropy":          _entropy(all_pot_pcts, bucket=0.1),
        "n_distinct_pot_pcts":  n_distinct_pot_pcts,
        "pot_pct_mean":         _m(all_pot_pcts),
        "pot_pct_std":          _s(all_pot_pcts),
        "pot_pct_cv":           _safe_cv(all_pot_pcts),
        # inter-player variance — low std = bots play alike
        "vpip_mean":            _m(player_vpips),
        "vpip_std":             _s(player_vpips),
        "pfr_mean":             _m(player_pfrs),
        "pfr_std":              _s(player_pfrs),
        "fold_mean":            _m(player_folds),
        "fold_std":             _s(player_folds),
        # temporal consistency — low CV over time = bot signal
        "fold_cv_time":         _safe_cv(window_folds),
        "vpip_cv_time":         _safe_cv(window_vpips),

        # ===== NEW FEATURES =====
        # sizing concentration — bots cluster at tier values
        "bet_mode_concentration":  bet_mode_concentration,
        "bet_top3_concentration":  bet_top3_concentration,
        "round_frac_rate":         round_frac_rate,
        "pot_pct_p10":             pot_pct_p10,
        "pot_pct_p90":             pot_pct_p90,
        "pot_pct_p90_p10":         pot_pct_p90_p10,
        # per-seat sizing consistency — low CV per seat = bot personality
        "seat_betsize_cv_mean":      seat_betsize_cv_mean,
        "seat_betsize_cv_min":       seat_betsize_cv_min,
        "seat_pot_pct_range":        seat_pot_pct_range,
        "seat_pot_pct_std_of_mean":  seat_pot_pct_std_of_mean,
        # per-street aggression
        "agg_freq_preflop":     agg_freq_preflop,
        "agg_freq_flop":        agg_freq_flop,
        "agg_freq_turn":        agg_freq_turn,
        "agg_freq_river":       agg_freq_river,
        "agg_freq_street_cv":   agg_freq_street_cv,
        # preflop dynamics
        "threebet_rate":        threebet_rate,
        "fourbet_rate":         fourbet_rate,
        "limp_rate":            limp_rate,
        # postflop dynamics
        "cbet_rate":            cbet_rate,
        "checkraise_per_hand":  checkraise_per_hand,
        # all-in & showdown
        "allin_rate":           allin_rate,
        "showdown_rate":        showdown_rate,
        # pace
        "actions_per_hand_mean": _m(actions_per_hand),
        "actions_per_hand_std":  _s(actions_per_hand),

        # ===== NEW BB-SIZING FEATURES (analogs of bet_mean_bb + round_frac_rate) =====
        # per-street bet size in BB — rigid bot has flat profile across streets
        "bet_mean_bb_preflop":     bet_mean_bb_preflop,
        "bet_mean_bb_flop":        bet_mean_bb_flop,
        "bet_mean_bb_turn":        bet_mean_bb_turn,
        "bet_mean_bb_river":       bet_mean_bb_river,
        "bet_mean_bb_street_cv":   bet_mean_bb_street_cv,
        # BB-distribution stats (analog of pot_pct_p10/p90 on BB scale)
        "bet_median_bb":           bet_median_bb,
        "bet_p10_bb":              bet_p10_bb,
        "bet_p90_bb":              bet_p90_bb,
        "bet_iqr_bb":              bet_iqr_bb,
        "bet_max_bb":              bet_max_bb,
        "bet_bb_entropy":          bet_bb_entropy,
        # round-number BB rate (humans like exact 3 BB opens, 10 BB 3-bets etc.)
        "near_integer_bb_rate":    near_integer_bb_rate,
        "common_bb_rate":          common_bb_rate,
        # extensions of near_integer_bb signal
        "mean_distance_to_int_bb":     mean_distance_to_int_bb,
        "median_distance_to_int_bb":   median_distance_to_int_bb,
        "near_half_bb_rate":           near_half_bb_rate,
        "bet_decimal_entropy":         bet_decimal_entropy,
        "near_integer_bb_rate_preflop":  near_integer_bb_rate_preflop,
        "near_integer_bb_rate_postflop": near_integer_bb_rate_postflop,
        "seat_near_integer_bb_min":    seat_near_integer_bb_min,
        "seat_near_integer_bb_max":    seat_near_integer_bb_max,
        "seat_near_integer_bb_range":  seat_near_integer_bb_range,
        # per-seat BB consistency
        "seat_bet_mean_bb_range":       seat_bet_mean_bb_range,
        "seat_bet_mean_bb_std_of_mean": seat_bet_mean_bb_std_of_mean,
        "seat_bet_bb_cv_mean":          seat_bet_bb_cv_mean,
        "seat_bet_bb_cv_min":           seat_bet_bb_cv_min,
    }


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
    Build a DataFrame of chunk-level features from a hand history file.
    Memory usage: O(chunk_size) — only one chunk in memory at a time.

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
            feats = extract_chunk_features(session)
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
                feats = extract_chunk_features(chunk)
                if feats:
                    feats["label"] = label
                    rows.append(feats)
                chunk = []
        # last incomplete chunk dropped — too few hands for reliable features

    df = pd.DataFrame(rows)
    print(f"{path.name}: {len(df)} chunks (label={label}, session_mode={session_mode})")
    return df
