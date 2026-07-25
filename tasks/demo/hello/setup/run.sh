#!/usr/bin/env bash
# The declared assets are already in place by the time this runs, so there is nothing
# left to prepare: both variants read what was staged and write to /ale/output.
set -euo pipefail

mkdir -p /ale/output
