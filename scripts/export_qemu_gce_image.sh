#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_qemu_gce_image.sh IMAGE DESTINATION_URI [PROJECT] [STAGING_BUCKET]

Exports a validated QEMU guest image to a new, versioned gs://...qcow2 object.
The source image must be READY and carry these labels:
  ale-image-role=qemu-guest
  ale-validation=passed

The destination must not already exist. This script never overwrites a disk.
For a Requester Pays destination, provide a non-Requester Pays staging bucket.
EOF
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

image="$1"
destination_uri="$2"
project="${3:-${GCP_PROJECT:-}}"
staging_bucket="${4:-}"

if [[ -z "$project" ]]; then
  echo "PROJECT or GCP_PROJECT is required" >&2
  exit 2
fi
if [[ ! "$destination_uri" =~ ^gs://.+\.qcow2$ ]]; then
  echo "destination must be a gs:// URI ending in .qcow2" >&2
  exit 2
fi
if [[ "$destination_uri" =~ /ale-(ubuntu22|win10)\.qcow2$ ]]; then
  echo "destination must be versioned; refusing canonical object: $destination_uri" >&2
  exit 2
fi

object_exists() {
  local uri="$1"
  local output
  if output="$(gcloud storage objects describe \
    "$uri" \
    --billing-project="$project" 2>&1)"; then
    return 0
  fi
  if grep -qiE 'not found|status[=: ]+404|HTTPError 404' <<<"$output"; then
    return 1
  fi
  echo "cannot verify whether object exists: $uri" >&2
  echo "$output" >&2
  exit 1
}

if object_exists "$destination_uri"; then
  echo "destination already exists: $destination_uri" >&2
  exit 1
fi

export_uri="$destination_uri"
if [[ -n "$staging_bucket" ]]; then
  if [[ ! "$staging_bucket" =~ ^gs://[^/]+/?$ ]]; then
    echo "staging bucket must be a gs:// bucket URI without an object path" >&2
    exit 2
  fi
  export_uri="${staging_bucket%/}/${destination_uri##*/}"
  if object_exists "$export_uri"; then
    echo "staging object already exists: $export_uri" >&2
    exit 1
  fi
fi

image_json="$(gcloud compute images describe "$image" --project="$project" --format=json)"
IMAGE_JSON="$image_json" python3 - <<'PY'
import json
import os

image = json.loads(os.environ["IMAGE_JSON"])
if image.get("status") != "READY":
    raise SystemExit(f"source image is not READY: {image.get('status')}")
labels = image.get("labels") or {}
required = {
    "ale-image-role": "qemu-guest",
    "ale-validation": "passed",
}
missing = {
    key: expected
    for key, expected in required.items()
    if labels.get(key) != expected
}
if missing:
    raise SystemExit(f"source image is missing validation labels: {missing}")
PY

echo "exporting $image to $export_uri"
gcloud compute images export \
  --project="$project" \
  --billing-project="$project" \
  --image="$image" \
  --destination-uri="$export_uri" \
  --export-format=qcow2

if [[ "$export_uri" != "$destination_uri" ]]; then
  echo "copying $export_uri to $destination_uri"
  gcloud storage cp \
    "$export_uri" \
    "$destination_uri" \
    --billing-project="$project"
fi

gcloud storage objects describe \
  "$destination_uri" \
  --billing-project="$project" \
  --format="yaml(name,bucket,size,generation,createTime,updateTime,crc32c,md5Hash)"

if [[ "$export_uri" != "$destination_uri" ]]; then
  gcloud storage rm "$export_uri" --billing-project="$project"
fi
