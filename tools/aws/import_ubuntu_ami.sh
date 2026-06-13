#!/usr/bin/env bash
# Stream the ale-ubuntu22 GCE export (raw disk) from GCS into S3, then turn it
# into an EC2 AMI via import-snapshot + register-image (NOT import-image).
#
# Why raw (not vmdk/qcow2): the exported VMDK is monolithicSparse and qcow2 is
# unsupported by AWS VM Import; only stream-optimized VMDK / raw / VHD work. The
# GCE default `.tar.gz` export is `disk.raw` gzipped in a tar, so we extract it
# on the fly and upload the raw disk.
#
# Why import-snapshot + register-image (not import-image): import-image rejects
# ALE's HWE kernel 6.5.0-15-generic ("Unsupported kernel version"). The snapshot
# path skips OS validation; register_from_snapshot.sh then registers an
# ENA/UEFI AMI. VERIFIED: the AMI boots on Nitro and cua-server answers on :5000.
#
# The pipe gsutil->tar->aws streams through this host's network with NO full
# local raw staging. multipart_chunksize is raised so the ~644GB stdin upload
# stays under the 10000-part S3 limit.
#
# Prereqs: source tools/aws/.aws-env (from bootstrap_account.sh); gsutil auth'd
# for gs://ale-data-public; aws CLI auth'd with ec2:ImportSnapshot + RegisterImage
# + iam:PassRole on role/vmimport + S3 write to $ALE_BUCKET.
set -euo pipefail
HERE="$(dirname "$0")"
# shellcheck disable=SC1091
source "${HERE}/.aws-env"

SRC="${ALE_IMAGE_SRC:-gs://ale-data-public/images/ale-ubuntu22.tar.gz}"
LOCAL_TARGZ="${ALE_LOCAL_TARGZ:-}"   # if set + exists, extract from here (robust)
KEY="images/ale-ubuntu22.raw"
S3URI="s3://${ALE_BUCKET}/${KEY}"
say() { printf '\n=== %s ===\n' "$*"; }

say "extract disk.raw -> ${S3URI}  (gunzip+untar -> S3 multipart, no local raw)"
if aws s3 ls "$S3URI" >/dev/null 2>&1; then
  echo "raw disk already in S3, skipping transfer"
else
  aws configure set default.s3.multipart_chunksize 512MB
  aws configure set default.s3.max_concurrent_requests 16
  # -xzO: extract member(s) to stdout. The GCE tarball holds a single disk.raw
  # (644 GB uncompressed). --expected-size keeps multipart under 10000 parts.
  if [ -n "$LOCAL_TARGZ" ] && [ -f "$LOCAL_TARGZ" ]; then
    echo "source: local $LOCAL_TARGZ"
    tar -xzO -f "$LOCAL_TARGZ" | aws s3 cp - "$S3URI" --expected-size 700000000000
  else
    echo "source: stream $SRC"
    gsutil cat "$SRC" | tar -xzO | aws s3 cp - "$S3URI" --expected-size 700000000000
  fi
fi

# NOTE: do NOT use `import-image` for the Ubuntu disk — it validates the guest
# kernel against a whitelist and rejects ALE's HWE kernel 6.5.0-15-generic
# ("Unsupported kernel version"). Use import-snapshot (no OS validation) +
# register-image instead. Verified: the resulting AMI boots on Nitro and brings
# up cua-server on :5000 (the 6.5 kernel ships nvme/ena modules).
say "start import-snapshot (Format=raw) — bypasses import-image OS validation"
CONT=$(mktemp)
cat > "$CONT" <<JSON
{"Description":"ALE ubuntu22 raw","Format":"raw","UserBucket":{"S3Bucket":"${ALE_BUCKET}","S3Key":"${KEY}"}}
JSON
TASK=$(aws ec2 import-snapshot --region "$AWS_REGION" \
  --description "ALE ubuntu22" \
  --disk-container "file://$CONT" \
  --query ImportTaskId --output text)
rm -f "$CONT"
echo "import-snapshot task: $TASK"

# poll the snapshot, then register an ENA/UEFI AMI from it.
"${HERE}/register_from_snapshot.sh" "$TASK" ale-ubuntu22 "${HERE}/.ubuntu-ami"

say "DONE — set ALE_AWS_UBUNTU_AMI=$(cat "${HERE}/.ubuntu-ami") for runs"
