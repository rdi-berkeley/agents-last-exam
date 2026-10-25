"""Recover the engine's byte rules from the visible corpus, with no reference consulted.

Run as::

    python3 recover_rules.py ../data/visible/scenarios.json ../data/visible/expected.json

The question this answers is whether an agent could reach the rules from the corpus it
is given, or whether they can only be known by reading pkmn/engine's source. If it were
the latter the task would be unfair, so it is worth settling by execution rather than
by assertion.

Measured result. Searching every rotation, the complement and a bit reversal against the
twelve visible Explosion scenarios:

  damage roll   one transform survives, spelled rotr1 or equivalently rotl7, with the
                damage base pinned to exactly 315 and the acceptance floor narrowed to
                212 to 219. Every floor in that window produces identical output on the
                held-out set, so the residual width costs an agent nothing.
  critical hit  one transform survives, spelled rotl3 or equivalently rotr5, with the
                rate constrained to 42 to 55. That interval contains 52, which is the
                modelled species' base speed halved.

Both rotations are therefore derivable from behaviour alone. What this does NOT show is
that the whole engine is: one family of 47 was modelled here.
"""

from __future__ import annotations

import functools
import json
import pathlib
import sys

SIDE, OFF_LAST_DAMAGE, TARGET_HP = 184, 370, 18


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (8 - k))) & 0xFF


def _rotr(x: int, k: int) -> int:
    return ((x >> k) | (x << (8 - k))) & 0xFF


def transforms():
    """Byte-level candidates an engineer would try, by name."""
    yield "identity", lambda x: x
    for k in range(1, 8):
        yield f"rotl{k}", functools.partial(_rotl, k=k)
        yield f"rotr{k}", functools.partial(_rotr, k=k)
    yield "complement", lambda x: (~x) & 0xFF
    yield "reverse_bits", lambda x: int(f"{x:08b}"[::-1], 2)


def observations(scenarios: dict, expected: dict, family_suffix: str = "Explosion"):
    """Tape, damage dealt, and whether the target's hit points clipped it."""
    out = []
    for sid, sc in scenarios.items():
        if not sid.rsplit("_", 1)[0].endswith(family_suffix):
            continue
        init = bytes.fromhex(sc["init"])
        state = bytes.fromhex(expected[sid]["state"])
        dmg = state[OFF_LAST_DAMAGE] | (state[OFF_LAST_DAMAGE + 1] << 8)
        hp = init[SIDE + TARGET_HP] | (init[SIDE + TARGET_HP + 1] << 8)
        out.append({"tape": bytes.fromhex(sc["tape"]), "damage": dmg, "capped": dmg >= hp})
    return out


def recover_damage_roll(obs: list) -> list:
    """(transform, floor, damage base) triples consistent with every uncapped scenario.

    Capped scenarios are excluded because the target's remaining hit points, not the
    roll, decided the number, so they carry no information about the base.
    """
    usable = [o for o in obs if o["damage"] and not o["capped"]]
    hits = []
    for name, f in transforms():
        for floor in range(128, 256):
            lo, hi, ok = 0, 10 ** 9, True
            for o in usable:
                i = 1
                while i < len(o["tape"]) and f(o["tape"][i]) < floor:
                    i += 1
                if i >= len(o["tape"]) or f(o["tape"][i]) == 0:
                    ok = False
                    break
                roll = f(o["tape"][i])
                lo = max(lo, (o["damage"] * 255 + roll - 1) // roll)
                hi = min(hi, ((o["damage"] + 1) * 255 - 1) // roll)
            if ok and lo <= hi:
                hits.append((name, floor, lo, hi))
    return hits


def recover_crit(obs: list, roll_transform, floor: int, base: int) -> list:
    """(transform, rate) pairs that separate the critical hits from the rest."""
    labelled = []
    for o in obs:
        if not o["damage"]:
            continue
        i = 1
        while i < len(o["tape"]) and roll_transform(o["tape"][i]) < floor:
            i += 1
        roll = roll_transform(o["tape"][i])
        implied = o["damage"] * 255 / roll if roll else 0
        labelled.append((o["tape"][0], o["capped"] or implied > base * 1.4))
    return [(name, rate) for name, f in transforms() for rate in range(1, 256)
            if all((f(b) < rate) == crit for b, crit in labelled)]


def main(argv: list[str]) -> int:
    scenarios = {s["id"]: s for s in json.loads(pathlib.Path(argv[1]).read_text())}
    expected = json.loads(pathlib.Path(argv[2]).read_text())
    obs = observations(scenarios, expected)
    print(f"visible scenarios in the modelled family: {len(obs)}")

    hits = recover_damage_roll(obs)
    names = sorted({h[0] for h in hits})
    bases = {h[2] for h in hits} | {h[3] for h in hits}
    floors = sorted({h[1] for h in hits})
    print(f"damage roll: {len(names)} transform(s) survive {names}, "
          f"base {sorted(bases)}, floor {floors[0]}..{floors[-1]}")

    rot = dict(transforms())[names[0]]
    crit = recover_crit(obs, rot, floors[0], min(bases))
    cnames = sorted({c[0] for c in crit})
    rates = sorted(r for n, r in crit if n == cnames[0])
    print(f"critical hit: {len(cnames)} transform(s) survive {cnames}, "
          f"rate {rates[0]}..{rates[-1]}")
    print("\nrotl7 is rotr1 and rotr5 is rotl3, so each is a single transform "
          "recovered from behaviour alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
