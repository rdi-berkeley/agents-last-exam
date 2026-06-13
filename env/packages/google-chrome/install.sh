#!/usr/bin/env bash
# Google Chrome stable (browser wrapper tasks). Installs from Google's apt repo.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v google-chrome >/dev/null 2>&1; then
  apt-get update; apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /etc/apt/keyrings/google-chrome.gpg
  chmod a+r /etc/apt/keyrings/google-chrome.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
  apt-get update; apt-get install -y google-chrome-stable; rm -rf /var/lib/apt/lists/*
fi
command -v google-chrome >/dev/null 2>&1 || { echo "[pkg google-chrome] FATAL: missing" >&2; exit 1; }
echo "[pkg google-chrome] OK ($(google-chrome --version 2>/dev/null))"
