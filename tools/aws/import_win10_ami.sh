#!/usr/bin/env bash
# Stream the ale-win10 GCE export (raw disk) from GCS into S3, then import it as
# an EC2 AMI via VM Import/Export.
#
# Windows specifics vs the ubuntu script:
#   • --platform Windows
#   • --license-type BYOL  (Windows 10 client is BYOL-only; AWS sells no Win10
#     license. Requires you to hold Microsoft VDA E3/E5 subscriptions, and the
#     resulting AMI must be launched on DEDICATED tenancy — see environment_aws
#     windows snapshot `tenancy: dedicated`.)
#   • --boot-mode uefi-preferred (GCE Windows images are typically UEFI)
#
# Driver caveat: AWS does NOT auto-inject ENA/NVMe drivers for Win10. If the
# launched instance won't boot or has no network/RDP, drivers are missing —
# install AWS NVMe + ENA drivers in the Windows image ON THE GCE SIDE and
# re-export, or debug live via the EC2 Serial Console (SAC).
set -euo pipefail
HERE="$(dirname "$0")"
# shellcheck disable=SC1091
source "${HERE}/.aws-env"

SRC="${ALE_WIN_IMAGE_SRC:-gs://ale-data-public/images/ale-win10.tar.gz}"
LOCAL_TARGZ="${ALE_WIN_LOCAL_TARGZ:-}"
KEY="images/ale-win10.raw"
S3URI="s3://${ALE_BUCKET}/${KEY}"
say() { printf '\n=== %s ===\n' "$*"; }

say "extract disk.raw -> ${S3URI}  (gunzip+untar -> S3 multipart, no local raw)"
if aws s3 ls "$S3URI" >/dev/null 2>&1; then
  echo "raw disk already in S3, skipping transfer"
else
  aws configure set default.s3.multipart_chunksize 512MB
  aws configure set default.s3.max_concurrent_requests 16
  if [ -n "$LOCAL_TARGZ" ] && [ -f "$LOCAL_TARGZ" ]; then
    echo "source: local $LOCAL_TARGZ"
    tar -xzO -f "$LOCAL_TARGZ" | aws s3 cp - "$S3URI" --expected-size 450000000000
  else
    echo "source: stream $SRC"
    gsutil cat "$SRC" | tar -xzO | aws s3 cp - "$S3URI" --expected-size 450000000000
  fi
fi

say "start import-image (Format=raw, Windows, BYOL)"
CONT=$(mktemp)
cat > "$CONT" <<JSON
[{"Description":"ALE win10 (from GCE export)","Format":"raw",
  "UserBucket":{"S3Bucket":"${ALE_BUCKET}","S3Key":"${KEY}"}}]
JSON
TASK=$(aws ec2 import-image --region "$AWS_REGION" \
  --description "ALE win10" --platform Windows --architecture x86_64 \
  --license-type BYOL --boot-mode uefi-preferred \
  --disk-containers "file://$CONT" \
  --query ImportTaskId --output text)
rm -f "$CONT"
echo "import task: $TASK"

say "poll until completed (~2-5h for ~400GB)"
while :; do
  read -r STATUS PROGRESS MSG IMAGE < <(aws ec2 describe-import-image-tasks \
    --region "$AWS_REGION" --import-task-ids "$TASK" \
    --query 'ImportImageTasks[0].[Status,Progress,StatusMessage,ImageId]' \
    --output text)
  printf '%s  status=%s progress=%s  %s\n' "$(date +%H:%M:%S)" \
    "$STATUS" "${PROGRESS:-?}" "${MSG:-}"
  case "$STATUS" in
    completed) echo "AMI: $IMAGE"; echo "$IMAGE" > "${HERE}/.win10-ami"; break ;;
    deleted|*[Ee]rror*) echo "import FAILED: $MSG"; exit 1 ;;
  esac
  sleep 60
done

say "DONE — set ALE_AWS_WIN_AMI=$(cat "${HERE}/.win10-ami") for runs (dedicated tenancy)"
