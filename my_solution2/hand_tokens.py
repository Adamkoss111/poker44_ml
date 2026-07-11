"""
hand_tokens.py — tokenizacja ręki pokerowej do modelu hand-level (bag-of-ngrams).

Token akcji = {street}{akcja}{kubelek sizingu wzgledem pota}:
  street: p/f/t/r (preflop/flop/turn/river)
  akcja:  F/X/C/B/R/A (fold/check/call/bet/raise/allin)
  sizing: 0 (brak kwoty), s/m/l/o (small <0.4 pota, medium <0.8, large <1.5, over)
Sizing POT-RELATIVE, nie BB — skala BB jest OOD miedzy benchmarkiem a live.

Dokument reki = unigramy + bigramy + trigramy sekwencji tokenow HERO
+ tokeny kontekstu: pozycja hero wzgl. buttona, dlugosc reki (kubelek),
liczba graczy, czy hero dotarl do showdown.
"""
from __future__ import annotations

_STREET = {"preflop": "p", "flop": "f", "turn": "t", "river": "r"}
_ACTION = {"fold": "F", "check": "X", "call": "C", "bet": "B",
           "raise": "R", "allin": "A", "all-in": "A", "all_in": "A"}


def _size_bucket(action: dict) -> str:
    amt = action.get("amount") or action.get("raise_to") or 0.0
    pot = action.get("pot_before") or 0.0
    if not amt or amt <= 0:
        return "0"
    if pot and pot > 0:
        frac = amt / pot
        if frac < 0.4:  return "s"
        if frac < 0.8:  return "m"
        if frac < 1.5:  return "l"
        return "o"
    return "m"


def tokenize_hand(hand: dict) -> str:
    """Reka -> dokument tokenow (string; slowa rozdzielone spacja)."""
    meta = hand.get("metadata") or {}
    hero = meta.get("hero_seat")
    actions = hand.get("actions") or []

    hero_seq = []
    n_actors = set()
    for a in actions:
        n_actors.add(a.get("actor_seat"))
        if a.get("actor_seat") != hero:
            continue
        st = _STREET.get(str(a.get("street", "")).lower(), "?")
        ac = _ACTION.get(str(a.get("action_type", "")).lower(), "?")
        if ac == "?":
            continue
        hero_seq.append(f"{st}{ac}{_size_bucket(a)}")

    toks = list(hero_seq)                                    # unigramy
    toks += [f"{a}_{b}" for a, b in zip(hero_seq, hero_seq[1:])]          # bigramy
    toks += [f"{a}_{b}_{c}" for a, b, c in zip(hero_seq, hero_seq[1:], hero_seq[2:])]  # trigramy

    # kontekst
    btn = meta.get("button_seat")
    if hero is not None and btn is not None:
        toks.append(f"pos{(int(hero) - int(btn)) % max(int(meta.get('max_seats') or 6), 2)}")
    ln = len(hero_seq)
    toks.append(f"len{min(ln, 8)}")
    toks.append(f"nseats{len(n_actors)}")
    return " ".join(toks) if toks else "empty"
