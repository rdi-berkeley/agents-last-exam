#!/usr/bin/env bash
# Runs inside the sandbox before the agent starts.
#
# This folder is copied in automatically; nothing declares it in task.yaml. Write the
# per-episode state your task needs here — anything random or time-dependent must be
# minted now rather than baked into an image ahead of time.
set -euo pipefail

printf 'world' > /ale/input/word.txt
