#!/usr/bin/env bash
# Sync ALE task data from gs://ale-data-public (GCS) to the AWS S3 task-data
# bucket. Streams cloud→cloud via gsutil (no local staging — the full dataset is
# ~1.5 TiB and won't fit on disk). The S3 bucket is requester-pays + public, so
# pullers (not the owner) pay egress — mirrors gcloud's ale-data-public.
#
# Auth: GCS reads use the host's gcloud login + a billing project (ale-data-public
# is requester-pays); S3 writes use your AWS creds (~/.aws) bridged into a temp
# boto config so gsutil can address s3://.
#
# Usage:  ./sync_task_data.sh [gs://src] [s3://dst]
#   defaults: gs://ale-data-public  ->  s3://ale-data-<account>
#   sync one subtree:  ./sync_task_data.sh gs://ale-data-public/demo s3://ale-data-<acct>/demo
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ACCT=$(aws sts get-caller-identity --query Account --output text)
SRC="${1:-gs://ale-data-public}"
DST="${2:-s3://ale-data-$ACCT}"
BILLING="${ALE_GCS_BILLING_PROJECT:-agenthle-488519}"   # billed for requester-pays GCS reads

# Bridge AWS creds into a boto config so gsutil can write to s3://.
AKID=$(aws configure get aws_access_key_id)
ASK=$(aws configure get aws_secret_access_key)
BOTO=$(mktemp); trap 'rm -f "$BOTO"' EXIT
cat > "$BOTO" <<CFG
[Credentials]
aws_access_key_id = $AKID
aws_secret_access_key = $ASK
CFG

# Exclude images/ — that prefix holds the multi-hundred-GB VM image EXPORTS
# (qcow2/vmdk/tar.gz), which are NOT task data; they become AMIs via
# scripts/aws/import_images.sh. Task data is only the domain dirs (~260 GiB).
echo "syncing $SRC -> $DST (requester-pays GCS billed to $BILLING; excluding images/)"
BOTO_CONFIG="$BOTO" gsutil -u "$BILLING" -m rsync -r -x '^images/.*' "$SRC" "$DST"
echo "DONE: $DST is in sync with $SRC (task data only)"
