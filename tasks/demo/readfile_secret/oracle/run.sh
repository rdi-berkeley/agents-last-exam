#!/usr/bin/env bash
# What a correct agent does: read the file, copy the token out.
set -euo pipefail

cat /ale/input/secret.txt > /ale/output/answer.txt
