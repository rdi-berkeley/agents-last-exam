#!/usr/bin/env bash
# Sync ALE task data from gs://ale-data-public (GCS) to the Alibaba Cloud OSS
# task-data bucket. The OSS bucket is requester-pays + public, so pullers (not
# the owner) pay egress — mirrors gcloud's ale-data-public.
#
# Unlike scripts/aws/sync_task_data.sh (which streams cloud→cloud because gsutil
# speaks s3:// via a boto provider), gsutil has NO OSS backend, so this stages a
# subtree to a local temp dir, then `ossutil cp -r` into OSS. Sync one domain at
# a time so the local staging stays bounded — the FULL dataset (~260 GiB) won't
# fit on disk; for a bulk one-shot copy use Alibaba's Data Online Migration
# (cross-cloud) service instead.
#
# Auth: GCS reads use the host's gcloud login + a billing project (ale-data-public
# is requester-pays); OSS writes use ambient Alibaba creds (ossutil config / env).
#
# Usage:  ./sync_task_data.sh <gs://src-subtree> <oss://dst-subtree>
#   e.g.  ./sync_task_data.sh gs://ale-data-public/demo oss://ale-data-<uid>/demo
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SRC="${1:?usage: sync_task_data.sh gs://src-subtree oss://dst-subtree}"
DST="${2:?usage: sync_task_data.sh gs://src-subtree oss://dst-subtree}"
BILLING="${ALE_GCS_BILLING_PROJECT:-agenthle-488519}"   # billed for requester-pays GCS reads

case "$SRC" in gs://*) ;; *) echo "src must be gs://..."; exit 2;; esac
case "$DST" in oss://*) ;; *) echo "dst must be oss://..."; exit 2;; esac

# Refuse the whole-bucket case: the local staging would blow up the disk.
[ "$SRC" = "gs://ale-data-public" ] && {
  echo "refusing to stage the full bucket locally — pass a domain subtree, or use"
  echo "Alibaba Data Online Migration for a bulk cross-cloud copy."; exit 2; }

STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
echo "staging $SRC -> $STAGE (requester-pays GCS billed to $BILLING)"
gsutil -u "$BILLING" -m rsync -r "$SRC" "$STAGE"
echo "pushing $STAGE -> $DST"
ossutil cp -r -f --enable-symlink-dir "$STAGE/" "$DST/"
echo "DONE: $DST is in sync with $SRC"
