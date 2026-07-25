#!/usr/bin/env bash
# The bundle is already staged into /ale/input by the time this runs.
#
# The upstream data still describes the previous framework's absolute paths, so the
# copies staged here are left untouched and the task's own instruction carries the
# workspace paths instead. Rewriting shared data would break the framework that still
# reads it.
set -euo pipefail

test -f /ale/input/spec.json
