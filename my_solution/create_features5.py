"""
create_features5.py  —  feature set v3 for Poker44 bot detection.

Motivated by the bot-detection / GTO literature:
  - Modern bots are either TOO consistent (near-GTO: stable action frequencies
    per position, fixed value/bluff mixing) OR carry a "humanization layer"
    that randomizes surface decisions but leaves higher-order stats intact.
    (SEON 2026; Poker Arena, arXiv:2606.13815)
  - Position is the strongest stable predictor of play; bots show an unnaturally
    regular positional profile across seats. (arXiv:2606.13815)
  - GTO mixing: in the same spot a solver bot randomizes in FIXED proportions;
    humans are inconsistent differently (streaky, fatigue-driven).
  - Bet sizing: bots scale sizing mathematically vs pot/stack; humans cluster on
    habitual fractions (½ pot, ¾ pot, pot) -> different granularity & context-fit.

These describe the WHOLE table by default (matching create_features3/4). If your
label is per-hero, pass hero_only=True to restrict to hero actions.

Usage:
    from create_features5 import extract_chunk_features_v3
    feats = {**extract_chunk_features(g),
             **extract_chunk_features_v2(g),
             **extract_chunk_features_v3(g)}
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np

VOLUNTARY = {"call", "raise", "bet", "all_in"}
AGG       = {"bet", "raise", "all_in"}
BLINDS    = {"small_blind", "big_blind", "ante"}


def _m(a):  return round(float(np.mean(a)), 4) if len(a) else np.nan
def _s(a):  return round(float(np.std(a)), 4) if len(a) else np.nan


def _entropy(counts) -> float:
    c = np.asarray(list(counts), dtype=float)
    if c.sum() == 0:
        return np.nan
    p = c / c.sum()
    return round(float(-np.sum(p * np.log2(p + 1e-12))), 4)


def _gini(counts) -> float:
    c = np.sort(np.asarray(list(counts), dtype=float))
    n = len(c)
    if n == 0 or c.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return round(float((2 * np.sum(idx * c)) / (n * c.sum()) - (n + 1) / n), 4)


def _nonblind(hand) -> List[dict]:
    return [a for a in (hand.get("actions") or []) if a.get("action_type") not in BLINDS]


def extract_chunk_features_v3(chunk: List[dict], hero_only: bool = False) -> Optional[dict]:
    if not chunk:
        return None
    out: dict = {}

    def actor_filter(hand, a):
        if not hero_only:
            return True
        return a.get("actor_seat") == (hand.get("metadata") or {}).get("hero_seat")

    # =====================================================================
    # 1. GTO MIXING CONSISTENCY
    # In a recurring decision context (street + facing-action), a near-GTO bot
    # mixes actions in stable proportions -> low entropy variance across similar
    # spots; humans drift. We bucket decisions by (street, n_players_in_hand,
    # is_facing_bet) and measure how mixed/consistent the action choice is.
    # =====================================================================
    ctx_actions = defaultdict(Counter)   # context -> Counter(action_type)
    for hand in chunk:
        acts = _nonblind(hand)
        n_players = len(hand.get("players") or [])
        # track running "facing a bet" flag within the hand per street
        facing = {}
        for a in acts:
            if not actor_filter(hand, a):
                # still update facing state below for context realism
                pass
            street = a.get("street", "")
            is_facing = int(facing.get(street, False))
            at = a.get("action_type")
            if actor_filter(hand, a) and at:
                ctx_actions[(street, min(n_players, 6), is_facing)][at] += 1
            if at in AGG:
                facing[street] = True

    # entropy per context, then aggregate
    ctx_entropies = []
    ctx_dominance = []   # share of most common action (1 = deterministic)
    for ctr in ctx_actions.values():
        tot = sum(ctr.values())
        if tot >= 5:                      # enough samples to be meaningful
            ctx_entropies.append(_entropy(ctr.values()))
            ctx_dominance.append(ctr.most_common(1)[0][1] / tot)
    out["gto_ctx_entropy_mean"] = _m(ctx_entropies)
    out["gto_ctx_entropy_std"]  = _s(ctx_entropies)   # bot: low variance across spots
    out["gto_ctx_dominance_mean"] = _m(ctx_dominance) # bot near-GTO: moderate, stable
    out["gto_ctx_dominance_std"]  = _s(ctx_dominance)
    out["gto_n_contexts"] = len(ctx_actions)

    # =====================================================================
    # 2. POSITIONAL PROFILE REGULARITY
    # Position proxied by action order preflop (button hidden). A bot has an
    # unnaturally smooth/monotone aggression-vs-position curve; humans are noisy.
    # =====================================================================
    pos_agg = defaultdict(lambda: [0, 0])   # order_idx -> [agg, total]
    pos_vpip = defaultdict(lambda: [0, 0])
    for hand in chunk:
        pre = [a for a in _nonblind(hand) if a.get("street") == "preflop"]
        seen = set()
        for k, a in enumerate(pre[:6]):
            if not actor_filter(hand, a):
                continue
            pos_agg[k][1] += 1
            pos_vpip[k][1] += 1
            if a.get("action_type") in {"raise", "all_in"}:
                pos_agg[k][0] += 1
            if a.get("action_type") in VOLUNTARY:
                pos_vpip[k][0] += 1

    agg_curve = [pos_agg[k][0] / pos_agg[k][1]
                 for k in range(6) if pos_agg[k][1] >= 8]
    vpip_curve = [pos_vpip[k][0] / pos_vpip[k][1]
                  for k in range(6) if pos_vpip[k][1] >= 8]
    # monotonicity: humans noisy, bots smooth. Measure via lag-1 diffs sign consistency
    def _monotonicity(curve):
        if len(curve) < 3:
            return np.nan
        diffs = np.diff(curve)
        if len(diffs) == 0 or np.all(diffs == 0):
            return np.nan
        # fraction of diffs sharing the dominant sign (1=perfectly monotone)
        pos = np.sum(diffs > 0); neg = np.sum(diffs < 0)
        return round(max(pos, neg) / len(diffs), 4)
    out["pos_agg_monotonicity"]  = _monotonicity(agg_curve)
    out["pos_vpip_monotonicity"] = _monotonicity(vpip_curve)
    out["pos_agg_range"]  = round(max(agg_curve) - min(agg_curve), 4) if len(agg_curve) >= 2 else np.nan
    out["pos_vpip_range"] = round(max(vpip_curve) - min(vpip_curve), 4) if len(vpip_curve) >= 2 else np.nan
    # smoothness: std of second differences (bot: low = smooth curve)
    out["pos_agg_roughness"] = (round(float(np.std(np.diff(agg_curve, 2))), 4)
                                if len(agg_curve) >= 3 else np.nan)

    # =====================================================================
    # 3. BET SIZING: CONTEXT-FIT vs HABITUAL FRACTIONS
    # Bots scale to pot/stack continuously; humans cluster on ½/¾/1x pot.
    # We compute bet/pot ratio and measure clustering on "round" pot fractions.
    # =====================================================================
    pot_fracs = []
    for hand in chunk:
        for a in _nonblind(hand):
            if not actor_filter(hand, a):
                continue
            if a.get("action_type") in {"bet", "raise"}:
                amt = a.get("amount")
                pot = a.get("pot_before")
                if amt and pot and pot > 0:
                    pot_fracs.append(amt / pot)
    pot_fracs = [f for f in pot_fracs if 0 < f < 5]   # drop degenerate
    if pot_fracs:
        pf = np.asarray(pot_fracs)
        out["potfrac_mean"] = round(float(pf.mean()), 4)
        out["potfrac_std"]  = round(float(pf.std()), 4)
        out["potfrac_cv"]   = round(float(pf.std() / (pf.mean() + 1e-9)), 4)
        # clustering on human round fractions {0.33,0.5,0.66,0.75,1.0}
        targets = np.array([0.33, 0.5, 0.66, 0.75, 1.0])
        dist = np.min(np.abs(pf[:, None] - targets[None, :]), axis=1)
        out["potfrac_round_rate"] = round(float(np.mean(dist < 0.05)), 4)  # human: high
        out["potfrac_dist_to_round_mean"] = round(float(dist.mean()), 4)   # bot: high
        # discretization: how many distinct sizes (bot: many; human: few)
        out["potfrac_n_unique"] = int(len(np.unique(np.round(pf, 2))))
        out["potfrac_entropy"] = _entropy(Counter(np.round(pf, 1)).values())
    else:
        for k in ("potfrac_mean", "potfrac_std", "potfrac_cv", "potfrac_round_rate",
                  "potfrac_dist_to_round_mean", "potfrac_n_unique", "potfrac_entropy"):
            out[k] = np.nan

    # =====================================================================
    # 4. ACTION-TYPE DIVERSITY & INEQUALITY
    # Bots often over-rely on a small action vocabulary in a balanced way;
    # humans skew. Gini of action-type usage captures the inequality shape.
    # =====================================================================
    at_counter = Counter()
    for hand in chunk:
        for a in _nonblind(hand):
            if actor_filter(hand, a):
                at_counter[a.get("action_type")] += 1
    out["action_type_gini"]    = _gini(at_counter.values())
    out["action_type_entropy"] = _entropy(at_counter.values())

    # =====================================================================
    # 5. STREET-PROGRESSION SHAPE
    # How aggression/continuation changes street to street. Bots have a stable
    # "shape" (e.g. cbet then give up at fixed rates); humans vary.
    # =====================================================================
    street_agg = {}
    for street in ("preflop", "flop", "turn", "river"):
        agg, n = 0, 0
        for hand in chunk:
            for a in _nonblind(hand):
                if a.get("street") == street and actor_filter(hand, a):
                    n += 1
                    if a.get("action_type") in AGG:
                        agg += 1
        street_agg[street] = agg / n if n else np.nan
    seq = [street_agg[s] for s in ("preflop", "flop", "turn", "river")]
    valid = [x for x in seq if not np.isnan(x)]
    if len(valid) >= 2:
        out["street_agg_slope"] = round(float(np.polyfit(range(len(valid)), valid, 1)[0]), 4)
        out["street_agg_dropoff"] = round(float(valid[0] - valid[-1]), 4)
    else:
        out["street_agg_slope"] = np.nan
        out["street_agg_dropoff"] = np.nan

    # =====================================================================
    # 6. DECISION-COMPLEXITY RESPONSE
    # Humans behave differently in multiway vs heads-up pots (cognitive load);
    # bots treat them more uniformly. Compare fold rate HU vs multiway.
    # =====================================================================
    fold_hu, tot_hu, fold_mw, tot_mw = 0, 0, 0, 0
    for hand in chunk:
        nplayers_acting = len({a.get("actor_seat") for a in _nonblind(hand)})
        for a in _nonblind(hand):
            if not actor_filter(hand, a):
                continue
            if nplayers_acting <= 2:
                tot_hu += 1; fold_hu += (a.get("action_type") == "fold")
            else:
                tot_mw += 1; fold_mw += (a.get("action_type") == "fold")
    if tot_hu >= 10 and tot_mw >= 10:
        out["fold_hu_vs_mw_diff"] = round(fold_hu / tot_hu - fold_mw / tot_mw, 4)
    else:
        out["fold_hu_vs_mw_diff"] = np.nan

    return out
