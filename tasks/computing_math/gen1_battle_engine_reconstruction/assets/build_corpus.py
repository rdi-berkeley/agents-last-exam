"""Build the Gen-I scenario corpus, and refuse to ship one that is not isolated.

One family per move: the move's holder is put in the lead slot of a one-Pokemon
side, so the scenario exercises that move and nothing else. Each family is run
over several roll tapes.

    python3 build_corpus.py <out_dir> [tapes_per_family] [cap]

Isolation is structural, not asserted: with one Pokemon a side there is no
replacement to hand the move script to, which is exactly how families leaked into
each other at 6v6. A faint simply ends the battle.

What is asserted:

  distinct      two *different families* producing an identical transcript are
                not testing two different things. Duplicates inside one family
                are expected: some mechanics consume no RNG at all, so the roll
                tape cannot vary them, and those families collapse to a single
                scenario.
  non-trivial   a scenario shorter than MIN_UPDATES never exercised its move.

Faints and cap truncations are reported but allowed. A truncated run is a
stalemate, which is still a perfectly gradeable deterministic transcript.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

# Path to the oracle built from the pinned pkmn/engine commit; see NOTES.md.
# Deliberately not defaulted to an absolute path from the author's machine.
ORACLE = os.environ.get("GEN1_ORACLE", "")

# Roll tape length. 1024 is empirically sufficient for every scenario at cap 20
# (verified: no exhaustion panic and byte-identical transcripts against a 65536
# tape across 144 scenarios), and it keeps the shipped corpus small enough to
# stage inline.
TAPE_LEN = 1 << 10

TEAM1 = [("Electrode", ["Wrap", "Thrash", "Dig", "PinMissile"]),
         ("Tauros", ["DoubleKick", "Twineedle", "Slash", "HornDrill"]),
         ("Gengar", ["Swift", "JumpKick", "DoubleEdge", "MegaDrain"]),
         ("Scyther", ["DreamEater", "Explosion", "Substitute", "Reflect"]),
         ("Charizard", ["LightScreen", "Haze", "Mist", "Disable"]),
         ("Jynx", ["Mimic", "Conversion", "SeismicToss", "SuperFang"])]
TEAM2 = [("Dragonite", ["FocusEnergy", "Bide", "Rage", "LeechSeed"]),
         ("Exeggutor", ["SleepPowder", "ThunderWave", "Toxic", "ConfuseRay"]),
         ("Machamp", ["Blizzard", "FireBlast", "BodySlam", "Bite"]),
         ("Chansey", ["Psybeam", "Smog", "AuroraBeam", "Acid"]),
         ("Rhydon", ["BubbleBeam", "Psychic", "Recover", "HyperBeam"]),
         ("Slowbro", ["SwordsDance", "Amnesia", "Agility", "Screech"])]

# Battle layout, from the engine's own `dump layout`: sides at 0, Side is 184
# bytes, Pokemon is 24 bytes, hp is at +18 as a little-endian u16.
SIDE, MON, HP = 184, 24, 18


def run(tape: int, p1: str, p2: str, cap: int) -> dict:
    out = subprocess.run([ORACLE, str(tape), p1, p2, str(cap)],
                         capture_output=True, text=True, timeout=300, check=False)
    lines = out.stdout.splitlines()
    return {
        "init": next(ln for ln in lines if ln.startswith("init")).split()[1],
        "updates": [ln for ln in lines if ln.startswith("u")],
        "result": next(ln for ln in lines if ln.startswith("result")),
        "state": next(ln for ln in lines if ln.startswith("state")).split()[1],
    }


def alive(state_hex: str) -> bool:
    b = bytes.fromhex(state_hex)
    return all(int.from_bytes(b[s * SIDE + HP: s * SIDE + HP + 2], "little") > 0
               for s in (0, 1))


def main(argv: list[str]) -> int:
    if not ORACLE or not pathlib.Path(ORACLE).is_file():
        raise SystemExit(
            "set GEN1_ORACLE to the oracle built from the pinned pkmn/engine "
            "commit; see NOTES.md for the build recipe")
    out_dir = pathlib.Path(argv[1])
    per = int(argv[2]) if len(argv) > 2 else 6
    cap = int(argv[3]) if len(argv) > 3 else 20

    families = ([("p1", li, mi, sp, mv)
                 for li, (sp, mvs) in enumerate(TEAM1, 1)
                 for mi, mv in enumerate(mvs, 1)]
                + [("p2", li, mi, sp, mv)
                   for li, (sp, mvs) in enumerate(TEAM2, 1)
                   for mi, mv in enumerate(mvs, 1)])

    # u0 is always the switch-in, so a scenario needs at least one real turn.
    # Not more: Explosion is a self-KO and legitimately gets exactly one.
    MIN_UPDATES = 2
    scenarios, expected, faints, truncs, trivial = [], {}, [], [], []
    tape = 700000
    for side, li, mi, sp, mv in families:
        for k in range(per):
            tape += 7919
            p1 = f"{li}:{mi}" if side == "p1" else "1:3"
            p2 = f"{li}:{mi}" if side == "p2" else "1:3"
            r = run(tape, p1, p2, cap)
            sid = f"{side}_{sp}_{mv}_{k}"
            # `init` is setup, not answer: it is the starting battle state and it
            # ships with every scenario, visible and held out alike, so the teams
            # and their stats are recoverable. Only the updates and the final
            # state are graded.
            scenarios.append({"id": sid, "tape": tape, "p1": p1, "p2": p2,
                              "cap": cap, "targets": f"{sp}/{mv}",
                              "init": r["init"]})
            expected[sid] = {"updates": r["updates"], "state": r["state"]}
            if not alive(r["state"]):
                faints.append(sid)
            if "truncated" in r["result"]:
                truncs.append(sid)
            if len(r["updates"]) < MIN_UPDATES:
                trivial.append(sid)

    fam = lambda sid: sid.rsplit("_", 1)[0]
    seen, cross, within = {}, [], []
    for sid, tr in expected.items():
        k = (tuple(tr["updates"]), tr["state"])
        if k in seen:
            (within if fam(seen[k]) == fam(sid) else cross).append((seen[k], sid))
        else:
            seen[k] = sid
    keep = set(seen.values())
    scenarios = [sc for sc in scenarios if sc["id"] in keep]
    expected = {k: v for k, v in expected.items() if k in keep}
    invariant = sorted({fam(b) for _, b in within})
    print(f"families   : {len(families)}")
    print(f"scenarios  : {len(scenarios)}  ({per} tapes each, cap {cap})")
    print(f"faints     : {len(faints)}  (allowed: ends the battle, no replacement exists)")
    print(f"truncated  : {len(truncs)}  (allowed: stalemate, still deterministic)")
    print(f"trivial    : {len(trivial)}")
    print(f"tape-invariant families (collapsed to one scenario): {len(invariant)}")
    for f in invariant:
        print(f"    {f}")
    print(f"kept       : {len(scenarios)} scenarios after dedupe")
    print(f"cross-family collisions : {len(cross)}")
    if trivial:
        raise AssertionError(f"{len(trivial)} scenarios ran fewer than {MIN_UPDATES} "
                             f"updates: {trivial[:5]}")
    if cross:
        raise AssertionError(
            f"{len(cross)} transcripts are shared across different families, so "
            f"those mechanics are indistinguishable: {cross[:6]}")

    # Split by tape *within* each family, never by family. A family-level split
    # would grade mechanics the agent never saw an example of, and a move-specific
    # effect like Bide cannot be inferred from never observing it: that is
    # unsolvable rather than hard. Every mechanic is demonstrated; the held-out
    # tapes test whether the rule was implemented or the transcript memorised.
    # Hold out the last 30% of each family's tapes. Measured at 6 tapes a family
    # the visible set gave a median of 4 examples per mechanic, which is thin for
    # inferring a rule like a bit-rotated comparison or a rejection-sampling loop.
    HOLDOUT_FROM = max(1, int(per * 0.7))
    out_dir.mkdir(parents=True, exist_ok=True)
    M = (1 << 64) - 1
    for sc in scenarios:
        z = (sc["tape"] + 0x9E3779B97F4A7C15) & M
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        x = (z ^ (z >> 31)) | 1
        buf = bytearray()
        for _ in range(TAPE_LEN):
            x ^= (x << 13) & M; x &= M; x ^= x >> 7; x ^= (x << 17) & M; x &= M
            buf.append((x >> 24) & 0xFF)
        sc["tape_id"] = sc["tape"]
        sc["tape"] = bytes(buf).hex()
    for name, want in (("visible", True), ("holdout", False)):
        sub = [sc for sc in scenarios
               if (int(sc["id"].rsplit("_", 1)[1]) < HOLDOUT_FROM) == want]
        d = out_dir / name
        d.mkdir(exist_ok=True)
        (d / "scenarios.json").write_text(json.dumps(sub, indent=1) + "\n")
        (d / "expected.json").write_text(
            json.dumps({sc["id"]: expected[sc["id"]] for sc in sub}, indent=1) + "\n")

        print(f"  {name:<8} {len(sub):>4} scenarios, "
              f"{len({fam(sc['id']) for sc in sub}):>3} families")
    size = sum(f.stat().st_size for f in out_dir.glob("*.json"))
    print(f"\nwrote {out_dir}  ({size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
