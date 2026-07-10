"""
features_tells.py — feature set v5: hero-centryczne "tells" (port z repo top-minera
vegasnc/Poker44-Miner, moduły bot_tells + gto_tells + advanced_tells, verbatim).

Trzy rodziny:
  - bot tells:      fold-to-aggression, AF per street, 3bet/VPIP ratio, WWSF,
                    cbet-fold, konsystencja sizingu, przejścia streetów
  - gto tells:      behavioral drift (połówki chunka), balans check/bet, probe bety,
                    stabilność PFR/VPIP, konsystencja odpowiedzi villainów
  - advanced tells: 4bet, SPR, multiway vs HU, entropia streetów, pozycja od buttona,
                    głębokość raise'ów

Wejście: chunk (lista rąk). Wyjście: dict cech (bez prefiksu — prefiks 'new4_'
nadaje rejestr features_registry.compute_features).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
from statistics import mean, pstdev
from typing import Any


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return default if denominator == 0 else numerator / denominator



# ======================================================================
# === bot_tells.py (port verbatim) ===
# ======================================================================

AGGRESSIVE = {"bet", "raise", "allin"}
STREETS = ("preflop", "flop", "turn", "river")


def extract_bot_tell_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    hero_seat = _hero_seat(chunk_group)

    features.update(_fold_to_aggression_features(chunk_group, hero_seat))
    features.update(_aggression_factor_features(chunk_group, hero_seat))
    features.update(_vpip_3bet_ratio_features(chunk_group, hero_seat))
    features.update(_postflop_surrender_features(chunk_group, hero_seat))
    features.update(_bet_size_consistency_features(chunk_group, hero_seat))
    features.update(_street_transition_features(chunk_group, hero_seat))
    return features


# ── Hero seat ──────────────────────────────────────────────────────────────

def _hero_seat(chunk_group: list[dict[str, Any]]) -> int | None:
    for hand in chunk_group:
        meta = hand.get("metadata", {})
        if isinstance(meta, dict) and meta.get("hero_seat") is not None:
            try:
                return int(meta["hero_seat"])
            except (TypeError, ValueError):
                pass
    return None


# ── 1. Fold-to-aggression by street ───────────────────────────────────────

def _fold_to_aggression_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Key bot tell: fold immediately when facing a bet/raise on flop/turn/river.
    Commercial bots (e.g. 3UpGaming) had WWSF ~15-17% — they almost never
    win when they see a flop, folding to any postflop aggression.
    """
    faced: dict[str, int] = {s: 0 for s in STREETS}
    folded: dict[str, int] = {s: 0 for s in STREETS}

    for hand in chunk_group:
        actions = hand.get("actions", [])
        by_street: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in actions:
            by_street[str(a.get("street", "unknown"))].append(a)

        for street in STREETS:
            street_actions = by_street[street]
            hero_acted = False
            villain_aggressed = False
            for i, a in enumerate(street_actions):
                seat = a.get("actor_seat")
                atype = str(a.get("action_type", ""))
                if seat != hero and atype in AGGRESSIVE:
                    villain_aggressed = True
                if seat == hero and villain_aggressed and not hero_acted:
                    faced[street] += 1
                    if atype == "fold":
                        folded[street] += 1
                    hero_acted = True

    features: dict[str, float] = {}
    for street in STREETS:
        features[f"bt_fold_to_{street}_aggression"] = safe_div(folded[street], faced[street])
        features[f"bt_faced_{street}_aggression_count"] = float(faced[street])

    # Overall postflop fold-to-aggression (flop+turn+river combined)
    post_faced = sum(faced[s] for s in ("flop", "turn", "river"))
    post_folded = sum(folded[s] for s in ("flop", "turn", "river"))
    features["bt_postflop_fold_to_aggression"] = safe_div(post_folded, post_faced)
    features["bt_postflop_faced_aggression"] = float(post_faced)
    return features


# ── 2. Aggression Factor (AF) by street ───────────────────────────────────

def _aggression_factor_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    AF = (bets + raises) / calls.
    Bot tell: high preflop AF (3-bets to steal) but very low postflop AF (~0.5).
    Normal human: more balanced AF across streets.
    """
    hero_agg: dict[str, int] = {s: 0 for s in STREETS}
    hero_calls: dict[str, int] = {s: 0 for s in STREETS}

    for hand in chunk_group:
        for a in hand.get("actions", []):
            if a.get("actor_seat") != hero:
                continue
            street = str(a.get("street", "unknown"))
            atype = str(a.get("action_type", ""))
            if atype in AGGRESSIVE:
                hero_agg[street] = hero_agg.get(street, 0) + 1
            elif atype == "call":
                hero_calls[street] = hero_calls.get(street, 0) + 1

    features: dict[str, float] = {}
    for street in STREETS:
        features[f"bt_hero_af_{street}"] = safe_div(hero_agg.get(street, 0), hero_calls.get(street, 0))

    preflop_af = safe_div(hero_agg.get("preflop", 0), hero_calls.get("preflop", 0))
    postflop_agg = sum(hero_agg.get(s, 0) for s in ("flop", "turn", "river"))
    postflop_calls = sum(hero_calls.get(s, 0) for s in ("flop", "turn", "river"))
    postflop_af = safe_div(postflop_agg, postflop_calls)

    features["bt_hero_af_preflop"] = preflop_af
    features["bt_hero_af_postflop"] = postflop_af
    # Mismatch: high preflop - low postflop is the bot tell
    features["bt_hero_af_mismatch"] = preflop_af - postflop_af
    features["bt_hero_af_ratio"] = safe_div(preflop_af, max(postflop_af, 0.01))
    return features


# ── 3. VPIP / 3bet ratio anomaly ──────────────────────────────────────────

def _vpip_3bet_ratio_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bot tell: VPIP=22 but 3bet=8 → 3bet/VPIP ratio ~0.36 (bots 3bet a huge
    fraction of their VPIP range). Normal humans: 3bet/VPIP ~0.10-0.20.
    Also: high 3bet with low postflop AF is a contradiction for humans.
    """
    vpip_hands = 0
    threebet_hands = 0
    open_raise_hands = 0
    hand_count = len(chunk_group)

    for hand in chunk_group:
        preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
        hero_pre = [a for a in preflop if a.get("actor_seat") == hero]

        if any(a.get("action_type") in {"call", "bet", "raise", "allin"} for a in hero_pre):
            vpip_hands += 1

        # 3-bet: hero raises after another player already raised
        raises_before_hero = 0
        for a in preflop:
            if a.get("actor_seat") != hero and a.get("action_type") in {"raise", "allin"}:
                raises_before_hero += 1
            if a.get("actor_seat") == hero and a.get("action_type") in {"raise", "allin"} and raises_before_hero >= 1:
                threebet_hands += 1
                break

        # Open raise: hero raises with no prior raise
        prior_raise = False
        for a in preflop:
            if a.get("actor_seat") != hero and a.get("action_type") in {"raise", "allin"}:
                prior_raise = True
            if a.get("actor_seat") == hero and a.get("action_type") in {"raise", "allin"} and not prior_raise:
                open_raise_hands += 1
                break

    vpip_rate = safe_div(vpip_hands, hand_count)
    threebet_rate = safe_div(threebet_hands, hand_count)
    open_raise_rate = safe_div(open_raise_hands, hand_count)

    return {
        "bt_hero_vpip": vpip_rate,
        "bt_hero_3bet_rate": threebet_rate,
        "bt_hero_open_raise_rate": open_raise_rate,
        # Key ratio: bots have abnormally high 3bet/VPIP
        "bt_hero_3bet_vpip_ratio": safe_div(threebet_rate, max(vpip_rate, 0.01)),
        # And high 3bet/open_raise ratio
        "bt_hero_3bet_open_ratio": safe_div(threebet_rate, max(open_raise_rate, 0.01)),
    }


# ── 4. Postflop surrender pattern (WWSF proxy) ────────────────────────────

def _postflop_surrender_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    WWSF (Won When Saw Flop): bots had 15-17%, humans typically 40-50%.
    Bots c-bet the flop but immediately fold to any counter-aggression,
    so they rarely win postflop pots they don't win on the flop c-bet.
    """
    saw_flop = 0
    won_postflop = 0
    cbet_then_folded = 0
    cbet_opportunities = 0
    check_fold_flop = 0
    check_flop_count = 0

    for hand in chunk_group:
        actions = hand.get("actions", [])
        by_street: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in actions:
            by_street[str(a.get("street", "unknown"))].append(a)

        flop_actions = by_street.get("flop", [])
        if not flop_actions:
            continue

        hero_on_flop = any(a.get("actor_seat") == hero for a in flop_actions)
        if not hero_on_flop:
            continue
        saw_flop += 1

        # Win: hand ended and hero is in winners
        outcome = hand.get("outcome", {})
        winners = outcome.get("winners") or []
        if hero in winners:
            won_postflop += 1

        # C-bet: hero raised preflop AND bets first on flop
        preflop_actions = by_street.get("preflop", [])
        hero_raised_preflop = any(
            a.get("actor_seat") == hero and a.get("action_type") in {"raise", "allin"}
            for a in preflop_actions
        )
        if hero_raised_preflop:
            cbet_opportunities += 1
            hero_flop_actions = [a for a in flop_actions if a.get("actor_seat") == hero]
            if hero_flop_actions and hero_flop_actions[0].get("action_type") in {"bet", "raise", "allin"}:
                # Hero c-bet — did they then fold to a raise?
                subsequent = [a for a in flop_actions if a.get("actor_seat") == hero and flop_actions.index(a) > flop_actions.index(hero_flop_actions[0])]
                if any(a.get("action_type") == "fold" for a in subsequent):
                    cbet_then_folded += 1

        # Check-fold on flop
        hero_flop = [a for a in flop_actions if a.get("actor_seat") == hero]
        if hero_flop and hero_flop[0].get("action_type") == "check":
            check_flop_count += 1
            if any(a.get("action_type") == "fold" for a in hero_flop[1:]):
                check_fold_flop += 1

    return {
        "bt_wwsf_proxy": safe_div(won_postflop, saw_flop),
        "bt_saw_flop_count": float(saw_flop),
        "bt_cbet_fold_rate": safe_div(cbet_then_folded, cbet_opportunities),
        "bt_cbet_opportunities": float(cbet_opportunities),
        "bt_check_fold_flop_rate": safe_div(check_fold_flop, check_flop_count),
    }


# ── 5. Bet size consistency (GTO snapping) ────────────────────────────────

def _bet_size_consistency_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bots snap to standard GTO fractions on every street with near-zero variance.
    Measure: std of hero's bet/pot ratios per street, and overall uniqueness.
    """
    STANDARD_FRACTIONS = [0.25, 0.33, 0.50, 0.67, 0.75, 1.00, 1.25, 1.50, 2.00]

    street_ratios: dict[str, list[float]] = {s: [] for s in STREETS}
    all_raise_to: list[float] = []

    for hand in chunk_group:
        for a in hand.get("actions", []):
            if a.get("actor_seat") != hero:
                continue
            if a.get("action_type") not in AGGRESSIVE:
                continue
            street = str(a.get("street", "unknown"))
            pot = numeric(a.get("pot_before"))
            amount = numeric(a.get("amount"))
            if pot > 0 and amount > 0 and a.get("action_type") != "allin":
                ratio = amount / pot
                if street in street_ratios:
                    street_ratios[street].append(ratio)
            rt = numeric(a.get("raise_to"))
            if rt > 0:
                all_raise_to.append(round(rt, 3))

    features: dict[str, float] = {}

    all_ratios: list[float] = []
    for street in STREETS:
        ratios = street_ratios[street]
        all_ratios.extend(ratios)
        if ratios:
            features[f"bt_hero_{street}_bet_ratio_std"] = float(pstdev(ratios)) if len(ratios) > 1 else 0.0
            features[f"bt_hero_{street}_bet_ratio_mean"] = float(mean(ratios))
            snapped = sum(
                1 for r in ratios
                if any(abs(r - f) <= 0.05 * f for f in STANDARD_FRACTIONS)
            )
            features[f"bt_hero_{street}_snap_rate"] = safe_div(snapped, len(ratios))
        else:
            features[f"bt_hero_{street}_bet_ratio_std"] = 0.0
            features[f"bt_hero_{street}_bet_ratio_mean"] = 0.0
            features[f"bt_hero_{street}_snap_rate"] = 0.0

    if all_ratios:
        features["bt_hero_overall_bet_ratio_std"] = float(pstdev(all_ratios)) if len(all_ratios) > 1 else 0.0
        features["bt_hero_overall_snap_rate"] = safe_div(
            sum(1 for r in all_ratios if any(abs(r - f) <= 0.05 * f for f in STANDARD_FRACTIONS)),
            len(all_ratios),
        )
    else:
        features["bt_hero_overall_bet_ratio_std"] = 0.0
        features["bt_hero_overall_snap_rate"] = 0.0

    # Unique raise_to amounts: bots reuse same sizes across hands
    features["bt_hero_unique_raise_to"] = float(len(set(all_raise_to)))
    features["bt_hero_raise_to_count"] = float(len(all_raise_to))
    features["bt_hero_raise_to_uniqueness"] = safe_div(len(set(all_raise_to)), len(all_raise_to))

    return features


# ── 6. Street transition aggression patterns ──────────────────────────────

def _street_transition_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bot pattern: aggressive preflop → passive/folding postflop.
    Measure hero's aggression transition across streets.
    """
    street_agg: dict[str, list[float]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        by_street: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in hand.get("actions", []):
            by_street[str(a.get("street", "unknown"))].append(a)

        for street in STREETS:
            hero_acts = [a for a in by_street[street] if a.get("actor_seat") == hero]
            if hero_acts:
                agg = sum(1 for a in hero_acts if a.get("action_type") in AGGRESSIVE)
                street_agg[street].append(safe_div(agg, len(hero_acts)))

    features: dict[str, float] = {}
    for street in STREETS:
        vals = street_agg[street]
        features[f"bt_hero_agg_rate_{street}"] = float(mean(vals)) if vals else 0.0

    pre = features.get("bt_hero_agg_rate_preflop", 0.0)
    flop = features.get("bt_hero_agg_rate_flop", 0.0)
    turn = features.get("bt_hero_agg_rate_turn", 0.0)
    river = features.get("bt_hero_agg_rate_river", 0.0)
    postflop = mean([flop, turn, river]) if any([flop, turn, river]) else 0.0

    # The key bot tell: high preflop, low postflop
    features["bt_preflop_to_postflop_agg_drop"] = max(0.0, pre - postflop)
    features["bt_agg_trend_flop_to_river"] = flop - river  # bots drop off quickly
    features["bt_postflop_agg_consistency"] = float(pstdev([flop, turn, river])) if all([flop, turn, river]) else 0.0

    return features


# ======================================================================
# === gto_tells.py (port verbatim) ===
# ======================================================================

VOLUNTARY_PREFLOP = {"call", "bet", "raise", "allin"}


def extract_gto_tell_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    hero = _hero_seat(chunk_group)
    features.update(_behavioral_drift_features(chunk_group, hero))
    features.update(_check_bet_balance_features(chunk_group, hero))
    features.update(_probe_bet_features(chunk_group, hero))
    features.update(_pfr_vpip_stability_features(chunk_group, hero))
    features.update(_villain_response_consistency_features(chunk_group, hero))
    return features



def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


# ── 1. First-half vs second-half behavioral drift ─────────────────────────

def _behavioral_drift_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bots are locked in pure GTO for <20 hands — no adaptation.
    Humans naturally adjust: they warm up, loosen up, adapt to table dynamics.
    Split chunk into first vs second half and measure behavioral differences.
    Large drift = human; near-zero drift = bot.
    """
    n = len(chunk_group)
    if n < 4:
        return {
            "gto_drift_vpip": 0.0, "gto_drift_pfr": 0.0, "gto_drift_agg": 0.0,
            "gto_drift_fold": 0.0, "gto_drift_bet_size": 0.0,
            "gto_drift_composite": 0.0,
        }

    mid = n // 2
    first_half = chunk_group[:mid]
    second_half = chunk_group[mid:]

    def half_stats(hands: list[dict]) -> dict[str, float]:
        vpip_hands, pfr_hands, hand_count = 0, 0, len(hands)
        agg_actions, total_actions, fold_actions = 0, 0, 0
        bet_ratios: list[float] = []
        for hand in hands:
            preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
            hero_pre = [a for a in preflop if a.get("actor_seat") == hero]
            if any(a.get("action_type") in VOLUNTARY_PREFLOP for a in hero_pre):
                vpip_hands += 1
            if any(a.get("action_type") in {"raise", "allin"} for a in hero_pre):
                pfr_hands += 1
            for a in hand.get("actions", []):
                if a.get("actor_seat") != hero:
                    continue
                total_actions += 1
                atype = a.get("action_type", "")
                if atype in AGGRESSIVE:
                    agg_actions += 1
                    pot = numeric(a.get("pot_before"))
                    amt = numeric(a.get("amount"))
                    if pot > 0 and amt > 0 and atype != "allin":
                        bet_ratios.append(amt / pot)
                elif atype == "fold":
                    fold_actions += 1
        return {
            "vpip": safe_div(vpip_hands, hand_count),
            "pfr": safe_div(pfr_hands, hand_count),
            "agg": safe_div(agg_actions, total_actions),
            "fold": safe_div(fold_actions, total_actions),
            "bet_size": float(mean(bet_ratios)) if bet_ratios else 0.0,
        }

    f = half_stats(first_half)
    s = half_stats(second_half)

    diffs = {k: abs(s[k] - f[k]) for k in f}
    composite = float(mean(diffs.values()))

    return {
        "gto_drift_vpip": diffs["vpip"],
        "gto_drift_pfr": diffs["pfr"],
        "gto_drift_agg": diffs["agg"],
        "gto_drift_fold": diffs["fold"],
        "gto_drift_bet_size": diffs["bet_size"],
        "gto_drift_composite": composite,
        # Direction of drift: positive = loosening (human tilt), negative = tightening
        "gto_drift_vpip_signed": s["vpip"] - f["vpip"],
        "gto_drift_agg_signed": s["agg"] - f["agg"],
    }


# ── 2. Check/bet balance per street ───────────────────────────────────────

def _check_bet_balance_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    GTO solvers prescribe specific check/bet frequency per street.
    Bots hit these frequencies precisely; humans deviate significantly.
    GTO typical: flop ~55% bet when in position, ~35% OOP.
    Key signal: hero's check/bet ratio deviates very little from a fixed value.
    Also measure balance between value bets and bluffs via bet-sizing consistency.
    """
    street_check: dict[str, int] = {s: 0 for s in STREETS}
    street_bet: dict[str, int] = {s: 0 for s in STREETS}
    # Track per-hand check/bet decisions to measure consistency
    per_hand_bet_rate: dict[str, list[float]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        by_street: dict[str, list] = defaultdict(list)
        for a in hand.get("actions", []):
            by_street[str(a.get("street",""))].append(a)

        for street in ("flop", "turn", "river"):
            hero_acts = [a for a in by_street[street] if a.get("actor_seat") == hero]
            if not hero_acts:
                continue
            n_bet = sum(1 for a in hero_acts if a.get("action_type") in AGGRESSIVE)
            n_check = sum(1 for a in hero_acts if a.get("action_type") == "check")
            street_bet[street] += n_bet
            street_check[street] += n_check
            total = n_bet + n_check
            if total > 0:
                per_hand_bet_rate[street].append(safe_div(n_bet, total))

    features: dict[str, float] = {}
    for street in ("flop", "turn", "river"):
        total = street_bet[street] + street_check[street]
        bet_rate = safe_div(street_bet[street], total)
        features[f"gto_bet_rate_{street}"] = bet_rate
        # Distance from 0.5 balance: pure GTO aims for a mix; measure deviation
        features[f"gto_bet_imbalance_{street}"] = abs(bet_rate - 0.5)
        # Consistency of bet/check decisions across hands (bots: near-zero std)
        rates = per_hand_bet_rate[street]
        features[f"gto_bet_rate_std_{street}"] = float(pstdev(rates)) if len(rates) > 1 else 0.0
        features[f"gto_bet_rate_cv_{street}"] = safe_div(
            float(pstdev(rates)) if len(rates) > 1 else 0.0,
            max(abs(float(mean(rates)) if rates else 0.0), 1e-9)
        )

    return features


# ── 3. Probe-bet (donk-bet) frequency when checked to ────────────────────

def _probe_bet_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    GTO probing: when villain checks to hero, the solver prescribes a specific
    bet frequency (often ~30-50% on each street). Bots hit this exactly.
    Humans probe inconsistently — sometimes never bet when checked to, sometimes always.
    Also: when villain bets and hero is last to act, hero's raise frequency.
    """
    checked_to_hero: dict[str, int] = {s: 0 for s in STREETS}
    hero_bet_after_check: dict[str, int] = {s: 0 for s in STREETS}
    villain_bet_hero_raised: dict[str, int] = {s: 0 for s in STREETS}
    villain_bet_hero_faced: dict[str, int] = {s: 0 for s in STREETS}

    # Per-hand rates for consistency measurement
    ph_probe_rates: dict[str, list[float]] = {s: [] for s in STREETS}
    ph_raise_rates: dict[str, list[float]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        by_street: dict[str, list] = defaultdict(list)
        for a in hand.get("actions", []):
            by_street[str(a.get("street",""))].append(a)

        for street in ("flop", "turn", "river"):
            acts = by_street[street]
            if not acts:
                continue

            hand_checked_to = 0
            hand_bet_after = 0
            hand_villain_bet = 0
            hand_raised = 0

            for i, a in enumerate(acts):
                seat = a.get("actor_seat")
                atype = a.get("action_type", "")

                # Was hero checked to? (villain checked, next meaningful action is hero's)
                if seat != hero and atype == "check":
                    # Check if hero acts next and what they do
                    for j in range(i + 1, len(acts)):
                        b = acts[j]
                        if b.get("actor_seat") == hero:
                            hand_checked_to += 1
                            checked_to_hero[street] += 1
                            if b.get("action_type") in AGGRESSIVE:
                                hand_bet_after += 1
                                hero_bet_after_check[street] += 1
                            break
                        break  # another non-hero acted first

                # Villain bet/raised → hero faces it
                if seat != hero and atype in AGGRESSIVE:
                    for j in range(i + 1, len(acts)):
                        b = acts[j]
                        if b.get("actor_seat") == hero:
                            hand_villain_bet += 1
                            villain_bet_hero_faced[street] += 1
                            if b.get("action_type") in AGGRESSIVE:
                                hand_raised += 1
                                villain_bet_hero_raised[street] += 1
                            break
                        break

            if hand_checked_to > 0:
                ph_probe_rates[street].append(safe_div(hand_bet_after, hand_checked_to))
            if hand_villain_bet > 0:
                ph_raise_rates[street].append(safe_div(hand_raised, hand_villain_bet))

    features: dict[str, float] = {}
    for street in ("flop", "turn", "river"):
        probe_rate = safe_div(hero_bet_after_check[street], checked_to_hero[street])
        raise_rate = safe_div(villain_bet_hero_raised[street], villain_bet_hero_faced[street])
        features[f"gto_probe_rate_{street}"] = probe_rate
        features[f"gto_raise_vs_bet_rate_{street}"] = raise_rate
        features[f"gto_checked_to_count_{street}"] = float(checked_to_hero[street])

        # Consistency (bots: very stable per-hand probe frequency)
        rates = ph_probe_rates[street]
        features[f"gto_probe_rate_std_{street}"] = float(pstdev(rates)) if len(rates) > 1 else 0.0

    return features


# ── 4. PFR/VPIP ratio stability across hands ──────────────────────────────

def _pfr_vpip_stability_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    "Too consistent numbers = suspicious" — rooms flag stable VPIP/PFR/3bet.
    Per-hand PFR/VPIP ratio variance is near-zero for GTO bots; humans show
    natural session-to-session and hand-to-hand variance.
    """
    per_hand_vpip: list[float] = []
    per_hand_pfr: list[float] = []
    per_hand_pfr_vpip_ratio: list[float] = []
    per_hand_3bet: list[float] = []

    for hand in chunk_group:
        preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
        hero_pre = [a for a in preflop if a.get("actor_seat") == hero]

        vpip = float(any(a.get("action_type") in VOLUNTARY_PREFLOP for a in hero_pre))
        pfr = float(any(a.get("action_type") in {"raise", "allin"} for a in hero_pre))

        # 3bet: hero raises after villain already raised
        prior_raise = False
        three_bet = 0.0
        for a in preflop:
            if a.get("actor_seat") != hero and a.get("action_type") in {"raise", "allin"}:
                prior_raise = True
            if a.get("actor_seat") == hero and a.get("action_type") in {"raise", "allin"} and prior_raise:
                three_bet = 1.0
                break

        per_hand_vpip.append(vpip)
        per_hand_pfr.append(pfr)
        per_hand_3bet.append(three_bet)
        # Ratio: meaningful only when vpip > 0
        if vpip > 0:
            per_hand_pfr_vpip_ratio.append(pfr / vpip)

    n = len(chunk_group)
    vpip_mean = float(mean(per_hand_vpip)) if per_hand_vpip else 0.0
    pfr_mean = float(mean(per_hand_pfr)) if per_hand_pfr else 0.0

    features: dict[str, float] = {
        "gto_vpip_mean": vpip_mean,
        "gto_pfr_mean": pfr_mean,
        "gto_3bet_mean": float(mean(per_hand_3bet)) if per_hand_3bet else 0.0,
        "gto_pfr_vpip_ratio": safe_div(pfr_mean, max(vpip_mean, 1e-9)),
        # Variance across hands (key bot signal: near-zero)
        "gto_vpip_std": float(pstdev(per_hand_vpip)) if len(per_hand_vpip) > 1 else 0.0,
        "gto_pfr_std": float(pstdev(per_hand_pfr)) if len(per_hand_pfr) > 1 else 0.0,
        "gto_pfr_vpip_ratio_std": float(pstdev(per_hand_pfr_vpip_ratio)) if len(per_hand_pfr_vpip_ratio) > 1 else 0.0,
        "gto_3bet_std": float(pstdev(per_hand_3bet)) if len(per_hand_3bet) > 1 else 0.0,
        # Coefficient of variation
        "gto_vpip_cv": safe_div(
            float(pstdev(per_hand_vpip)) if len(per_hand_vpip) > 1 else 0.0,
            max(vpip_mean, 1e-9)
        ),
        "gto_pfr_cv": safe_div(
            float(pstdev(per_hand_pfr)) if len(per_hand_pfr) > 1 else 0.0,
            max(pfr_mean, 1e-9)
        ),
    }
    return features


# ── 5. Villain aggression → hero response consistency ─────────────────────

def _villain_response_consistency_features(chunk_group: list[dict[str, Any]], hero: int | None) -> dict[str, float]:
    """
    Bots respond to the same situation the same way every time (deterministic GTO).
    When a villain bets on a specific street, a bot folds/calls/raises at fixed freqs.
    Measure: hero's response distribution when facing villain bets, per street.
    Low entropy in that distribution = bot.
    Also: does hero's response change based on pot size? (Humans: yes; bots: less so)
    """
    response_actions: dict[str, list[str]] = {s: [] for s in STREETS}
    response_small_pot: dict[str, list[str]] = {s: [] for s in STREETS}
    response_big_pot: dict[str, list[str]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        actions = hand.get("actions", [])
        meta = hand.get("metadata", {})
        bb = numeric(meta.get("bb")) or 0.02

        for i, a in enumerate(actions):
            if a.get("actor_seat") == hero:
                continue
            if a.get("action_type") not in AGGRESSIVE:
                continue
            street = str(a.get("street", ""))
            if street not in STREETS:
                continue

            pot = numeric(a.get("pot_before"))
            pot_in_bb = pot / bb if bb > 0 else 0

            # Next hero action
            for j in range(i + 1, len(actions)):
                b = actions[j]
                if b.get("street") != street:
                    break
                if b.get("actor_seat") == hero:
                    atype = str(b.get("action_type", ""))
                    response_actions[street].append(atype)
                    if pot_in_bb < 10:
                        response_small_pot[street].append(atype)
                    else:
                        response_big_pot[street].append(atype)
                    break
                break

    features: dict[str, float] = {}
    for street in STREETS:
        acts = response_actions[street]
        ent = _entropy(Counter(acts))
        features[f"gto_response_entropy_{street}"] = ent
        features[f"gto_response_count_{street}"] = float(len(acts))

        # Fold rate when facing villain bet per street
        features[f"gto_fold_rate_vs_bet_{street}"] = safe_div(
            acts.count("fold"), len(acts)
        )
        features[f"gto_call_rate_vs_bet_{street}"] = safe_div(
            acts.count("call"), len(acts)
        )
        features[f"gto_raise_rate_vs_bet_{street}"] = safe_div(
            sum(1 for a in acts if a in {"raise","bet","allin"}), len(acts)
        )

        # Pot-size sensitivity: does hero respond differently by pot size?
        small_ent = _entropy(Counter(response_small_pot[street]))
        big_ent = _entropy(Counter(response_big_pot[street]))
        # Bots: same entropy regardless of pot; humans: more cautious in big pots
        features[f"gto_response_pot_sensitivity_{street}"] = abs(big_ent - small_ent)

    # Overall response entropy across all postflop streets
    all_post = []
    for street in ("flop", "turn", "river"):
        all_post.extend(response_actions[street])
    features["gto_postflop_response_entropy"] = _entropy(Counter(all_post))

    return features


# ======================================================================
# === advanced_tells.py (port verbatim) ===
# ======================================================================



def extract_advanced_tell_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(_fourbet_features(chunk_group))
    features.update(_spr_features(chunk_group))
    features.update(_multiway_hu_features(chunk_group))
    features.update(_street_entropy_features(chunk_group))
    features.update(_villain_fold_features(chunk_group))
    features.update(_positional_features(chunk_group))
    features.update(_raise_depth_features(chunk_group))
    return features


# ── helpers ────────────────────────────────────────────────────────────────



def _fourbet_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Bots trained on GTO solvers have precise 4bet frequencies (~10-15% vs 3bets).
    Humans rarely 4-bet and do so inconsistently.
    Key bot tell: a bot will 4bet exactly the solver-optimal % of the time.
    """
    hero = _hero_seat(chunk_group)
    threebet_faced = 0    # times hero faces a 3bet (villain raised after hero's raise)
    fourbet_by_hero = 0
    fourbet_faced = 0     # times hero faces a 4bet
    fivebet_by_hero = 0
    total_hands = len(chunk_group)

    for hand in chunk_group:
        preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
        raise_number = 0  # running count of total raises so far
        hero_raised = False
        hero_raise_at = -1

        for a in preflop:
            atype = a.get("action_type", "")
            seat = a.get("actor_seat")

            if atype in AGGRESSIVE:
                raise_number += 1
                if seat == hero:
                    hero_raised = True
                    hero_raise_at = raise_number
                    if raise_number == 3:  # hero is 4-betting
                        fourbet_by_hero += 1
                    elif raise_number == 5:
                        fivebet_by_hero += 1
                else:
                    if raise_number == 3 and hero_raised and hero_raise_at == 2:
                        # hero raised (2bet or 3bet), villain now 4bets
                        threebet_faced += 1  # actually villain 3bet hero's open
                    if raise_number == 2 and hero_raised and hero_raise_at == 1:
                        threebet_faced += 1
                    if raise_number == 4 and hero_raise_at == 3:
                        fourbet_faced += 1

    return {
        "at_hero_4bet_rate": safe_div(fourbet_by_hero, max(threebet_faced, 1)),
        "at_hero_5bet_rate": safe_div(fivebet_by_hero, max(fourbet_faced, 1)),
        "at_hero_4bet_count": float(fourbet_by_hero),
        "at_hero_5bet_count": float(fivebet_by_hero),
        "at_3bet_faced_count": float(threebet_faced),
        "at_4bet_faced_count": float(fourbet_faced),
        # Normalised per hand
        "at_hero_4bet_per_hand": safe_div(fourbet_by_hero, total_hands),
        "at_hero_5bet_per_hand": safe_div(fivebet_by_hero, total_hands),
    }


# ── 2. SPR-dependent aggression ────────────────────────────────────────────

def _spr_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    SPR (Stack-to-Pot Ratio) dramatically changes optimal strategy.
    Humans play very differently at SPR<3 vs SPR>10; bots apply the same GTO
    template. Measure: hero's aggression rate stratified by SPR bucket.
    Also: variance of hero's aggression across SPR buckets (bots=near-zero).
    """
    hero = _hero_seat(chunk_group)
    # SPR buckets: shallow (<3), medium (3-10), deep (>10)
    bucket_agg: dict[str, list[float]] = {"shallow": [], "medium": [], "deep": []}
    bucket_call: dict[str, list[float]] = {"shallow": [], "medium": [], "deep": []}

    for hand in chunk_group:
        meta = hand.get("metadata", {})
        bb = numeric(meta.get("bb")) or 0.02
        players = hand.get("players", [])
        hero_player = next((p for p in players if p.get("seat") == hero), None)
        if not hero_player:
            continue

        hero_stack = numeric(hero_player.get("starting_stack"))
        if hero_stack <= 0:
            continue

        # Find the pot at the start of the flop to compute SPR
        actions = hand.get("actions", [])
        flop_actions = [a for a in actions if a.get("street") == "flop"]
        if not flop_actions:
            continue

        pot_at_flop = numeric(flop_actions[0].get("pot_before"))
        if pot_at_flop <= 0:
            continue

        spr = hero_stack / pot_at_flop
        bucket = "shallow" if spr < 3 else ("medium" if spr < 10 else "deep")

        hero_flop_acts = [a for a in flop_actions if a.get("actor_seat") == hero]
        if not hero_flop_acts:
            continue

        n_agg = sum(1 for a in hero_flop_acts if a.get("action_type") in AGGRESSIVE)
        n_call = sum(1 for a in hero_flop_acts if a.get("action_type") == "call")
        agg_rate = safe_div(n_agg, len(hero_flop_acts))
        call_rate = safe_div(n_call, len(hero_flop_acts))
        bucket_agg[bucket].append(agg_rate)
        bucket_call[bucket].append(call_rate)

    features: dict[str, float] = {}
    agg_means = []
    for bkt in ("shallow", "medium", "deep"):
        vals = bucket_agg[bkt]
        features[f"at_hero_agg_{bkt}"] = float(mean(vals)) if vals else 0.0
        features[f"at_hero_{bkt}_hand_count"] = float(len(vals))
        if vals:
            agg_means.append(float(mean(vals)))

    # Variance across SPR buckets: bots show near-zero variance (same play regardless of SPR)
    features["at_spr_agg_variance"] = float(pstdev(agg_means)) if len(agg_means) > 1 else 0.0
    features["at_spr_agg_range"] = (max(agg_means) - min(agg_means)) if len(agg_means) > 1 else 0.0

    return features


# ── 3. Multiway vs heads-up aggression ────────────────────────────────────

def _multiway_hu_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    GTO solvers are optimised for heads-up. In multiway pots, humans drastically
    tighten their range; bots may continue with HU frequencies or have poorly
    calibrated multiway play. Measure the difference in hero aggression HU vs MW.
    """
    hero = _hero_seat(chunk_group)
    hu_agg, hu_total = 0, 0
    mw_agg, mw_total = 0, 0

    for hand in chunk_group:
        actions = hand.get("actions", [])
        by_street: dict[str, list] = defaultdict(list)
        for a in actions:
            by_street[str(a.get("street", ""))].append(a)

        for street in ("flop", "turn", "river"):
            street_acts = by_street[street]
            if not street_acts:
                continue
            active_seats = {a.get("actor_seat") for a in street_acts
                            if a.get("action_type") not in {"fold"}}
            is_hu = len(active_seats) <= 2
            hero_acts = [a for a in street_acts if a.get("actor_seat") == hero]
            if not hero_acts:
                continue

            n_agg = sum(1 for a in hero_acts if a.get("action_type") in AGGRESSIVE)
            if is_hu:
                hu_agg += n_agg
                hu_total += len(hero_acts)
            else:
                mw_agg += n_agg
                mw_total += len(hero_acts)

    hu_rate = safe_div(hu_agg, hu_total)
    mw_rate = safe_div(mw_agg, mw_total)
    return {
        "at_hero_hu_postflop_agg": hu_rate,
        "at_hero_mw_postflop_agg": mw_rate,
        # Humans: big positive diff (less aggressive MW); bots: near-zero diff
        "at_hero_hu_mw_agg_diff": hu_rate - mw_rate,
        "at_hero_hu_hands": float(hu_total),
        "at_hero_mw_hands": float(mw_total),
        "at_hero_mw_ratio": safe_div(mw_total, max(hu_total + mw_total, 1)),
    }


# ── 4. Per-street action entropy ───────────────────────────────────────────

def _street_entropy_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Entropy of hero's action distribution per street.
    Bots: low entropy (nearly always choose the same action in a given spot).
    Humans: higher entropy (mix folds, calls, raises with more variety).
    Also measures entropy of BET SIZES (bots repeat same sizes).
    """
    hero = _hero_seat(chunk_group)
    street_actions: dict[str, list[str]] = {s: [] for s in STREETS}
    street_bet_sizes: dict[str, list[float]] = {s: [] for s in STREETS}

    for hand in chunk_group:
        for a in hand.get("actions", []):
            if a.get("actor_seat") != hero:
                continue
            street = str(a.get("street", "unknown"))
            if street not in STREETS:
                continue
            atype = str(a.get("action_type", ""))
            street_actions[street].append(atype)
            if atype in AGGRESSIVE:
                pot = numeric(a.get("pot_before"))
                amt = numeric(a.get("amount"))
                if pot > 0 and amt > 0:
                    street_bet_sizes[street].append(round(amt / pot, 2))

    features: dict[str, float] = {}
    all_entropy = []
    for street in STREETS:
        acts = street_actions[street]
        ent = _entropy(Counter(acts)) if acts else 0.0
        features[f"at_hero_action_entropy_{street}"] = ent
        features[f"at_hero_action_count_{street}"] = float(len(acts))
        if acts:
            all_entropy.append(ent)

        # Bet size entropy: bots snap to same sizes → very low entropy
        sizes = street_bet_sizes[street]
        if sizes:
            # Quantise to 0.05 buckets for entropy
            quantised = [round(s / 0.05) * 0.05 for s in sizes]
            features[f"at_hero_bet_size_entropy_{street}"] = _entropy(Counter(
                f"{q:.2f}" for q in quantised
            ))
        else:
            features[f"at_hero_bet_size_entropy_{street}"] = 0.0

    features["at_hero_action_entropy_mean"] = float(mean(all_entropy)) if all_entropy else 0.0
    features["at_hero_action_entropy_std"] = float(pstdev(all_entropy)) if len(all_entropy) > 1 else 0.0
    return features


# ── 5. Villain fold-to-hero-bet rate ──────────────────────────────────────

def _villain_fold_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Measure how often opponents fold to hero's bets/raises.
    Bots betting GTO fractions often get called more (opponents can't be exploited);
    human bluffers get different fold equity. Also captures whether hero is bluffing
    in spots where humans normally wouldn't.
    """
    hero = _hero_seat(chunk_group)
    hero_bet_then_villain_fold = 0
    hero_bet_then_villain_call = 0
    hero_bet_then_villain_raise = 0
    total_hero_bets_faced = 0

    for hand in chunk_group:
        actions = hand.get("actions", [])
        for i, a in enumerate(actions):
            if a.get("actor_seat") != hero:
                continue
            if a.get("action_type") not in AGGRESSIVE:
                continue
            # Look at the next action by a non-hero player on the same street
            street = a.get("street")
            for j in range(i + 1, len(actions)):
                b = actions[j]
                if b.get("street") != street:
                    break
                if b.get("actor_seat") == hero:
                    continue
                total_hero_bets_faced += 1
                btype = b.get("action_type", "")
                if btype == "fold":
                    hero_bet_then_villain_fold += 1
                elif btype == "call":
                    hero_bet_then_villain_call += 1
                elif btype in AGGRESSIVE:
                    hero_bet_then_villain_raise += 1
                break

    return {
        "at_villain_fold_to_hero_bet": safe_div(hero_bet_then_villain_fold, total_hero_bets_faced),
        "at_villain_call_to_hero_bet": safe_div(hero_bet_then_villain_call, total_hero_bets_faced),
        "at_villain_raise_to_hero_bet": safe_div(hero_bet_then_villain_raise, total_hero_bets_faced),
        "at_hero_bets_faced_by_villain": float(total_hero_bets_faced),
    }


# ── 6. Positional aggression differential ─────────────────────────────────

def _positional_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    In position (IP) players should be more aggressive postflop than OOP.
    Bots trained on GTO have very precise positional awareness; others may over- or
    under-adjust. Measure hero's postflop aggression when IP vs OOP.

    IP approximation: hero_seat is ≥ button_seat (acts after button in postflop)
    or hero is the button. Simplified: hero is IP when button_seat < hero_seat or
    hero_seat == button_seat.
    """
    hero = _hero_seat(chunk_group)
    ip_agg, ip_total = 0, 0
    oop_agg, oop_total = 0, 0

    for hand in chunk_group:
        meta = hand.get("metadata", {})
        button_seat = meta.get("button_seat")
        max_seats = int(meta.get("max_seats", 6))
        if button_seat is None or hero is None:
            continue

        # Determine if hero is IP: hero acts after all remaining players postflop.
        # In a 6-max game, postflop action starts left of button.
        # Hero is last to act (IP) if hero == button_seat or closest clockwise before btn.
        # Simplified: compute "distance from button" — smaller = closer to button = more IP.
        def seats_from_btn(seat: int) -> int:
            return (button_seat - seat) % max_seats

        hero_dist = seats_from_btn(hero)
        # hero_dist=0: hero is button (most IP postflop)
        # hero_dist=1: hero is cutoff
        is_ip = hero_dist <= 1  # button or cutoff approximation

        for a in hand.get("actions", []):
            if a.get("actor_seat") != hero:
                continue
            if a.get("street") not in ("flop", "turn", "river"):
                continue
            atype = a.get("action_type", "")
            if is_ip:
                ip_total += 1
                if atype in AGGRESSIVE:
                    ip_agg += 1
            else:
                oop_total += 1
                if atype in AGGRESSIVE:
                    oop_agg += 1

    ip_rate = safe_div(ip_agg, ip_total)
    oop_rate = safe_div(oop_agg, oop_total)

    return {
        "at_hero_ip_postflop_agg": ip_rate,
        "at_hero_oop_postflop_agg": oop_rate,
        # Humans: positively skewed (more aggressive IP); bots: very consistent or perfectly calibrated
        "at_hero_ip_oop_agg_diff": ip_rate - oop_rate,
        "at_hero_ip_hands": float(ip_total),
        "at_hero_oop_hands": float(oop_total),
        "at_hero_ip_ratio": safe_div(ip_total, max(ip_total + oop_total, 1)),
    }


# ── 7. Preflop raise sequence depth distribution ──────────────────────────

def _raise_depth_features(chunk_group: list[dict[str, Any]]) -> dict[str, float]:
    """
    Measure the distribution of preflop raise depths (2bet/3bet/4bet/5bet).
    Bots trigger 4bet/5bet wars at GTO-precise frequencies; humans are inconsistent.
    Also: bots always respond to given raise depths the same way (low entropy per depth).
    """
    hero = _hero_seat(chunk_group)
    depth_counts: Counter = Counter()  # depth at which the hand ended preflop
    hero_response_at_depth: dict[int, list[str]] = defaultdict(list)  # depth→hero's actions

    for hand in chunk_group:
        preflop = [a for a in hand.get("actions", []) if a.get("street") == "preflop"]
        raise_depth = 0
        for a in preflop:
            atype = a.get("action_type", "")
            if atype in AGGRESSIVE:
                raise_depth += 1
            if a.get("actor_seat") == hero and raise_depth > 0:
                hero_response_at_depth[raise_depth].append(atype)
        depth_counts[raise_depth] += 1

    total = sum(depth_counts.values()) or 1
    features: dict[str, float] = {
        "at_preflop_2bet_rate": depth_counts.get(1, 0) / total,
        "at_preflop_3bet_rate": depth_counts.get(2, 0) / total,
        "at_preflop_4bet_rate": depth_counts.get(3, 0) / total,
        "at_preflop_5bet_plus_rate": sum(v for k, v in depth_counts.items() if k >= 4) / total,
        "at_preflop_max_depth": float(max(depth_counts.keys(), default=0)),
        "at_preflop_depth_entropy": _entropy(Counter({str(k): v for k, v in depth_counts.items()})),
    }

    # Hero's action entropy at each raise depth (low = bot-like determinism)
    for depth in (1, 2, 3):
        acts = hero_response_at_depth[depth]
        features[f"at_hero_action_entropy_at_{depth}bet"] = _entropy(Counter(acts)) if acts else 0.0
        features[f"at_hero_count_at_{depth}bet"] = float(len(acts))

    return features


# ======================================================================
# Zbiorczy ekstraktor v5
# ======================================================================

def extract_chunk_features_v5(chunk):
    """Wszystkie trzy rodziny tells w jednym dict. None dla pustego chunka."""
    if not chunk:
        return None
    out = {}
    out.update(extract_bot_tell_features(chunk))
    out.update(extract_gto_tell_features(chunk))
    out.update(extract_advanced_tell_features(chunk))
    return out
