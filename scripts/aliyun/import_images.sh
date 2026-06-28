#!/usr/bin/env bash
# Import ALE sandbox images from gs://ale-data-public/images/ into Alibaba Cloud
# as ECS custom images. One re-runnable script for the whole flow. Idempotent-ish:
# skips the OSS upload if the raw is already there.
#
#   ./import_images.sh ale-ubuntu22       # Linux: ImportImage, then bake aliyun CLI + ossutil
#   ./import_images.sh ale-win10          # Windows 10 desktop
#   ./import_images.sh ale-win-server     # Windows Server (GPU)
#   ./import_images.sh all                # all three
#
# Per image it: streams the GCS tar.gz (disk.raw) into the OSS import bucket
# (no local staging), turns it into a custom image via `aliyun ecs ImportImage`,
# tags it `ale:image-family=<name>` (the tag AliyunProvider resolves), and — for
# Linux — bakes the aliyun CLI + ossutil in (needed for oss:// data/output).
#
# Auth: ambient Alibaba credentials (`aliyun configure` / ALIBABA_CLOUD_ACCESS_KEY_*).
# GCS reads use the host's gcloud login + a billing project (requester-pays).
#
# Mirror of scripts/aws/import_images.sh. Aliyun differences:
#   • ImportImage validates the guest OS but (unlike AWS import-image) accepts
#     ALE's Ubuntu HWE kernel, so Linux goes straight through ImportImage.
#   • The import source is an OSS object (RAW), not an S3 object.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
R="${ALE_ALIYUN_REGION:-cn-hangzhou}"
UID_=$(aliyun sts GetCallerIdentity 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["AccountId"])')
BUCKET="ale-images-$UID_"
GCS=gs://ale-data-public/images
BILLING="${ALE_GCS_BILLING_PROJECT:-agenthle-488519}"   # billed for requester-pays GCS reads
SG_NAME=ale-sandbox
say() { printf '\n=== %s ===\n' "$*"; }
J() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)"; }

ensure_bucket() {                    # OSS import bucket exists (region-local)
  ossutil ls "oss://$BUCKET/" >/dev/null 2>&1 || \
    ossutil mb "oss://$BUCKET" --region "$R" >/dev/null
}

ensure_raw() {                       # $1 family → ensure oss://$BUCKET/images/$1.raw
  local key="images/$1.raw"
  if ossutil stat "oss://$BUCKET/$key" >/dev/null 2>&1; then echo "raw present: $1"; return; fi
  say "stream $GCS/$1.tar.gz -> oss://$BUCKET/$key"
  # gsutil can't write oss://, and ossutil can read stdin via `cp - oss://...`,
  # so we pipe GCS → tar → ossutil. The full raw never lands on local disk.
  gsutil -u "$BILLING" cat "$GCS/$1.tar.gz" | tar -xzO | ossutil cp - "oss://$BUCKET/$key"
}

poll_image() {                       # $1 image-id → wait until Available
  while :; do
    local st; st=$(aliyun ecs DescribeImages --RegionId "$R" --ImageId "$1" \
      | J "['Images']['Image'][0].get('Status','?')")
    echo "  $1 $st" >&2
    [ "$st" = Available ] && return
    case "$st" in CreateFailed|UnAvailable) echo "FAILED" >&2; return 1;; esac
    sleep 60
  done
}

tag_image() {                        # $1 image-id, $2 family
  aliyun ecs TagResources --RegionId "$R" --ResourceType image \
    --ResourceId.1 "$1" \
    --Tag.1.Key ale:image-family --Tag.1.Value "$2" \
    --Tag.2.Key Name --Tag.2.Value "$2" >/dev/null
  echo "tagged $1 (ale:image-family=$2)"
}

import_one() {                       # $1 family, $2 OSType (linux|windows), $3 Platform
  local fam=$1 ostype=$2 platform=$3
  ensure_bucket; ensure_raw "$fam"
  say "$fam: ImportImage (OSType=$ostype Platform=$platform)"
  local img
  img=$(aliyun ecs ImportImage --RegionId "$R" \
    --ImageName "$fam-$(date +%s)" --OSType "$ostype" --Platform "$platform" \
    --Architecture x86_64 --BootMode UEFI \
    --DiskDeviceMapping.1.OSSBucket "$BUCKET" \
    --DiskDeviceMapping.1.OSSObject "images/$fam.raw" \
    --DiskDeviceMapping.1.Format RAW \
    | J "['ImageId']")
  echo "import image: $img"
  poll_image "$img" || return 1
  tag_image "$img" "$fam"
  echo "DONE $fam -> $img"
}

case "${1:?usage: import_images.sh ale-ubuntu22|ale-win10|ale-win-server|all}" in
  ale-ubuntu22)   import_one ale-ubuntu22   linux   Ubuntu ;;
  ale-win10)      import_one ale-win10       windows Windows ;;
  ale-win-server) import_one ale-win-server  windows Windows ;;
  all)            import_one ale-ubuntu22 linux Ubuntu
                  import_one ale-win10 windows Windows
                  import_one ale-win-server windows Windows ;;
  *) echo "unknown image $1"; exit 2 ;;
esac
