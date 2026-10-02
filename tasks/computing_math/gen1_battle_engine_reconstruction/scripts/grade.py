"""Host-side scoring for computing_math/gen1_battle_engine_reconstruction.

No sandbox dependency, so it is unit testable and the held-out transcripts never
need to exist on the task VM.

Two numbers, because they answer different questions:

  full_pass    every held-out scenario reproduced exactly. An engine is
               bit-exact or it is not, and this is the headline.
  mechanics    mean over families of the fraction of that family's scenarios
               reproduced. One family is one move, so this reads as "how many of
               the mechanics did the submission get right". A submission with a
               single localized bug scores near 1.0 here and 0 on full_pass,
               which is the honest description of it.

The per-family breakdown is the diagnostic: the lowest families name the broken
mechanic. A corpus of random battles cannot do that.
"""

from __future__ import annotations

import re

UPDATE = re.compile(r"^u\d+ ")
STATE = re.compile(r"^state ([0-9a-f]+)\s*$")


def parse(text: str) -> dict:
    """Updates and final state from a submission's stdout; other lines ignored."""
    updates, state = [], None
    for line in (text or "").splitlines():
        if UPDATE.match(line):
            updates.append(line.rstrip())
        else:
            m = STATE.match(line.strip())
            if m:
                state = m.group(1)
    return {"updates": updates, "state": state}


def family_of(scenario_id: str) -> str:
    """Scenario ids are <side>_<species>_<move>_<tape index>."""
    return scenario_id.rsplit("_", 1)[0]


def score(results: dict, expected: dict) -> dict:
    """Score raw runner output against the held-out transcripts.

    ``results`` maps scenario id to {"rc": int, "stdout": str}. A missing entry
    is a failed run, never an error.
    """
    per_family: dict[str, list[bool]] = {}
    for sid, want in expected.items():
        got = parse((results.get(sid) or {}).get("stdout", ""))
        ok = got["updates"] == list(want["updates"]) and got["state"] == want["state"]
        per_family.setdefault(family_of(sid), []).append(ok)

    families = {f: sum(v) / len(v) for f, v in per_family.items()}
    exact = sum(sum(v) for v in per_family.values())
    total = sum(len(v) for v in per_family.values())
    return {
        "families": families,
        "broken": sorted((f for f, v in families.items() if v < 1.0),
                         key=lambda f: families[f]),
        "scenarios_exact": exact,
        "scenarios_total": total,
        "mechanics": sum(families.values()) / len(families) if families else 0.0,
        "full_pass": 1.0 if total and exact == total else 0.0,
    }
