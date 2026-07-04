#!/usr/bin/env bash
# Sync ALE task data from gs://ale-data-public (GCS) to the Alibaba Cloud OSS
# task-data bucket. Mirror of scripts/aws/sync_task_data.sh.
#
# AWS streams cloud→cloud because gsutil speaks s3:// via a boto provider. gsutil
# has NO OSS backend and ossutil can't read gs://, so this stages to a local temp
# dir then `ossutil cp -r` into OSS. To keep local disk bounded on the full
# dataset (~260 GiB across domains), the default whole-bucket sync runs ONE
# top-level domain at a time (stage → push → wipe → next) rather than staging
# everything at once. For a bulk one-shot, Alibaba Data Online Migration
# (cross-cloud) is faster.
#
# The OSS bucket should be requester-pays + public so pullers (not the owner) pay
# egress — mirrors gcloud's ale-data-public. images/ is excluded (those are the
# multi-hundred-GB VM image exports, not task data — they become custom images
# via scripts/aliyun/import_images.sh).
#
# Auth: GCS reads use the host's gcloud login + a billing project (ale-data-public
# is requester-pays); OSS writes use ambient Alibaba creds (ossutil config / env).
#
# Usage:
#   ./sync_task_data.sh                              # gs://ale-data-public -> oss://ale-data-<acct>, all domains
#   ./sync_task_data.sh gs://src-subtree oss://dst   # one subtree (staged whole)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
J() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)"; }
ACCT=$(aliyun sts GetCallerIdentity 2>/dev/null | J "['AccountId']")
SRC="${1:-gs://ale-data-public}"
DST="${2:-oss://ale-data-$ACCT}"
BILLING="${ALE_GCS_BILLING_PROJECT:-agenthle-488519}"   # billed for requester-pays GCS reads
STAGE_ROOT="${ALE_STAGE_DIR:-$(mktemp -d)}"
trap 'rm -rf "$STAGE_ROOT"' EXIT

case "$SRC" in gs://*) ;; *) echo "src must be gs://..."; exit 2;; esac
case "$DST" in oss://*) ;; *) echo "dst must be oss://..."; exit 2;; esac

stage_push() {                       # $1 gs-subtree  $2 oss-subtree
  local stage; stage=$(mktemp -d "$STAGE_ROOT/XXXX")
  echo "  stage $1 -> $stage"
  gsutil -u "$BILLING" -m rsync -r "$1" "$stage"
  echo "  push  $stage -> $2"
  ossutil cp -r -f "$stage/" "$2/"
  rm -rf "$stage"
}

if [ "$SRC" = "gs://ale-data-public" ] && [ -z "${2:-}" ]; then
  # Whole-dataset default: one top-level domain at a time, excluding images/.
  echo "syncing all domains $SRC -> $DST (GCS billed to $BILLING; excluding images/)"
  for d in $(gsutil -u "$BILLING" ls "$SRC/" | sed -n 's#.*/\([^/]*\)/$#\1#p'); do
    [ "$d" = images ] && { echo "skip images/"; continue; }
    echo "=== domain: $d ==="
    stage_push "$SRC/$d" "$DST/$d"
  done
  echo "DONE: all domains synced to $DST"
else
  # Explicit subtree: stage + push directly.
  echo "syncing $SRC -> $DST (GCS billed to $BILLING)"
  stage_push "$SRC" "$DST"
  echo "DONE: $DST is in sync with $SRC"
fi
