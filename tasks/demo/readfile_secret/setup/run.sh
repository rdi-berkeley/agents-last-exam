#!/usr/bin/env bash
# Mint a fresh token for this episode.
#
# Short and unguessable: eight hex characters are trivial to copy verbatim, while a long
# token is reproduced in shape but garbled by weaker models — which would conflate a
# copy-fidelity failure with the content-forwarding failure this task exists to catch.
#
# The directories are this task's to create: the framework builds only what the task
# declared, and this one generates its input rather than downloading it.
set -euo pipefail

mkdir -p /ale/input /ale/output
token="ALE-SECRET-$(python3 -c 'import secrets; print(secrets.token_hex(4))')-END"
printf '%s\n' "$token" > /ale/input/secret.txt
