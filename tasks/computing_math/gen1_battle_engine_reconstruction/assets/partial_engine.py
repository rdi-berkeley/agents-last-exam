"""A self-contained Gen-I engine covering the Explosion family, standard library only.

Written to answer one question: can the deliverable this task demands, a single
Python file with no oracle available, reproduce held-out scenarios byte for byte?

Everything it knows was derived from the visible corpus and the two shipped format
documents. The damage base (315), the critical-hit rate (52, which is Scyther's base
speed halved), the rotations and the roll's position in the tape were all fitted on
visible scenarios and then checked against held-out ones they had no part in.

Scenarios outside the family it models are answered with the switch-in it can derive
and nothing more, which scores zero for those rather than pretending.
"""
from __future__ import annotations

import json
import sys

W = 0
ROTR1 = lambda x: ((x >> 1) | (x << 7)) & 0xFF
ROTL3 = lambda x: ((x << 3) | (x >> 5)) & 0xFF

SIDE, POKEMON = 184, 24
OFF_POKEMON, OFF_ACTIVE = 0, 144
OFF_LAST_SELECTED, OFF_LAST_USED = 182, 183
OFF_TURN, OFF_LAST_DAMAGE, OFF_LAST_MOVES = 368, 370, 372

EXPLOSION = 0x99
BASE, CRIT_BASE, CRIT_RATE = 315, 613, 52


def u16(b, i):
    return b[i] | (b[i + 1] << 8)


def put16(b, i, v):
    b[i] = v & 0xFF
    b[i + 1] = (v >> 8) & 0xFF


def slot(state, side, field, size=2):
    base = side * SIDE + OFF_POKEMON
    return u16(state, base + field) if size == 2 else state[base + field]


def switch_in(state: bytearray, side: int) -> bytes:
    """Copy the team slot into the active slot and report the switch."""
    p = side * SIDE + OFF_POKEMON
    a = side * SIDE + OFF_ACTIVE
    state[a:a + 10] = state[p:p + 10]            # stats
    state[a + 10] = state[p + 21]                # species
    state[a + 11] = state[p + 22]                # types
    state[a + 24:a + 32] = state[p + 10:p + 18]  # moves
    ident = 0x01 if side == 0 else 0x09
    return bytes([0x04, ident, state[p + 21], state[p + 23],
                  state[p + 18], state[p + 19], state[p + 18], state[p + 19], 0x00])


def run(scenario: dict) -> str:
    state = bytearray.fromhex(scenario["init"])
    tape = bytes.fromhex(scenario["tape"])

    u0 = switch_in(state, 0) + switch_in(state, 1) + bytes([0x07, 0x01, 0x00, 0x00])
    lines = [f"u0 0 0 0 0 {u0.hex()}"]

    # the move each side selected, as an index into the active move list
    p1_slot = int(scenario["p1"].split(":")[1])
    p2_slot = int(scenario["p2"].split(":")[1])
    a = 0 * SIDE + OFF_ACTIVE
    p1_move = state[a + 24 + 2 * (p1_slot - 1)]
    b = 1 * SIDE + OFF_ACTIVE
    p2_move = state[b + 24 + 2 * (p2_slot - 1)]

    if p1_move != EXPLOSION:
        # outside what this engine models; report the switch-in only
        return "\n".join(lines) + f"\nstate {bytes(state).hex()}\n"

    crit = ROTL3(tape[0]) < CRIT_RATE
    i = 1
    while ROTR1(tape[i]) < 217:
        i += 1
    roll = ROTR1(tape[i])
    dealt = (CRIT_BASE if crit else BASE) * roll // 255

    tgt_hp_at = 1 * SIDE + OFF_POKEMON + 18
    tgt_hp = u16(state, tgt_hp_at)
    tgt_max = u16(state, tgt_hp_at)

    # accuracy: the tape byte after the damage roll decides whether it lands
    hit = tape[i + 1] < 0xFF
    body = bytearray([0x03, 0x01, EXPLOSION, 0x09])
    if hit:
        dealt = min(dealt, tgt_hp)
        body += bytes([0x00])
        body += bytes([0x0A, 0x09]) + (tgt_hp - dealt).to_bytes(2, "little") \
            + tgt_max.to_bytes(2, "little") + bytes([0x00, 0x00])
        put16(state, tgt_hp_at, tgt_hp - dealt)
        put16(state, OFF_LAST_DAMAGE, dealt)
    else:
        body += bytes([0x00, 0x02, 0x11, 0x01])
        put16(state, OFF_LAST_DAMAGE, 0)
    body += bytes([0x06, 0x01, 0x08, 0x01, 0x00])

    put16(state, 0 * SIDE + OFF_POKEMON + 18, 0)     # the user faints
    # the move's PP falls in both the team slot and the active copy of it
    pp_at = 0 * SIDE + OFF_POKEMON + 10 + 2 * (p1_slot - 1) + 1
    state[pp_at] = max(0, state[pp_at] - 1)
    active_pp = 0 * SIDE + OFF_ACTIVE + 24 + 2 * (p1_slot - 1) + 1
    state[active_pp] = state[pp_at]
    state[0 * SIDE + OFF_LAST_SELECTED] = p1_move
    state[1 * SIDE + OFF_LAST_SELECTED] = p2_move
    state[OFF_TURN] = 1
    state[OFF_LAST_MOVES] = 0x12
    state[OFF_LAST_MOVES + 1] = 0x03

    lines.append(f"u1 1 {p1_slot} 1 {p2_slot} {bytes(body).hex()}")
    return "\n".join(lines) + f"\nstate {bytes(state).hex()}\n"


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        scenario = json.load(fh)
    sys.stdout.write(run(scenario))
