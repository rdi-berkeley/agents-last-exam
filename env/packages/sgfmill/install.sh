#!/usr/bin/env bash
# sgfmill (pure-Python SGF library) for the go_game_reconstruction grader (verify_sgf.py).
# Installed into system python3 so the on-VM grader can import it.
set -euo pipefail
python3 -c "import sgfmill" 2>/dev/null || pip3 install --quiet --break-system-packages "sgfmill==1.1.1" 2>/dev/null || pip3 install --quiet "sgfmill==1.1.1"
python3 -c "import sgfmill; print('[pkg sgfmill] OK', sgfmill.__name__)"
