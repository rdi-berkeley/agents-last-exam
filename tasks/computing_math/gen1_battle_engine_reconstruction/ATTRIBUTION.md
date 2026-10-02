# Third-party attribution

## pkmn/engine

The corpus shipped with this task, and the two format documents under
`data/`, are produced by executing [pkmn/engine](https://github.com/pkmn/engine)
built in cartridge-accurate mode. `assets/oracle_script.zig` imports it.

- Commit pinned: `78dc891c49788e6ec9007d0f02247d2e04a03d29`
- Build: Zig 0.16.0, `-Doptimize=ReleaseFast`, default (non-`showdown`) mode
- Upstream carries no releases or version tags; the commit is the only pin.

pkmn/engine is distributed under the MIT License. Its notice is reproduced
verbatim below, as that license requires.

```
Copyright (c) 2021-2026 pkmn contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## What this task does not contain

No Nintendo or Game Freak code, data, or assets. No ROM is required to build or
run anything here, and none is distributed.

pkmn/engine is an independent reimplementation that aims at cartridge accuracy.
It is not derived from the original game's code. The
[pret/pokered](https://github.com/pret/pokered) disassembly, which does
reconstruct the original, is deliberately **not** used: it carries no license
file and its build output is a copyrighted ROM.

The task's ground truth is therefore precisely "whatever the pinned pkmn/engine
binary does". That is a well-defined, deterministic, reproducible authority, and
it is what the task card claims. It is not a claim of fidelity to 1996 hardware.

## Species and move names

The roster uses species and move identifiers as spelled in pkmn/engine's own
`Species` and `Move` enumerations. They are factual identifiers for game
mechanics, used to name scenario families so a failing family is legible.
