#!/usr/bin/env bash
# Import ALE sandbox images from gs://ale-data-public/images/ into Alibaba Cloud
# as ECS custom images. One re-runnable script for the whole flow (mirror of
# scripts/aws/import_images.sh). Idempotent-ish: skips the OSS upload if the raw
# is already there.
#
#   ./import_images.sh ale-ubuntu22       # Linux: ImportImage, bake aliyun CLI + ossutil
#   ./import_images.sh ale-win10          # Windows 10 desktop
#   ./import_images.sh ale-win-server     # Windows Server (GPU)
#   ./import_images.sh all                # all three
#
# Per image it: streams the GCS tar.gz (disk.raw) into the OSS import bucket
# (no local staging), turns it into a custom image via `aliyun ecs ImportImage`,
# tags it `ale:image-family=<name>` (the tag AliyunProvider resolves). For Linux
# it then bakes the aliyun CLI + ossutil into the image (needed for oss:// data /
# output) by launching the imported image, installing through the in-guest cua
# server, and re-capturing with CreateImage — exactly as the AWS script bakes the
# aws CLI.
#
# Why per-OS paths (mirrors AWS): the Linux image needs the OSS CLIs baked in;
# Windows images already ship what they need, so they go import-and-tag only.
# Unlike AWS (whose import-image rejects ALE's Ubuntu HWE kernel, forcing
# import-snapshot+register), Alibaba ImportImage accepts the raw directly, so
# both OSes use ImportImage.
#
# Prerequisites:
#   • OSS service ACTIVATED on the account (console one-time "开通 OSS"); the raw
#     must live in OSS for ImportImage to read it.
#   • ambient Alibaba credentials (`aliyun configure` / ALIBABA_CLOUD_ACCESS_KEY_*).
#   • host gcloud login + a billing project for requester-pays GCS reads.
#   • a security group `ale-sandbox` with a vSwitch in its VPC (see the docs /
#     the live-setup the provider expects).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
say() { printf '\n=== %s ===\n' "$*"; }
J()  { python3 -c "import sys,json;print(json.load(sys.stdin)$1)"; }
R="${ALE_ALIYUN_REGION:-cn-hangzhou}"
ACCT=$(aliyun sts GetCallerIdentity 2>/dev/null | J "['AccountId']")
BUCKET="ale-images-$ACCT"
GCS=gs://ale-data-public/images
BILLING="${ALE_GCS_BILLING_PROJECT:-agenthle-488519}"   # requester-pays GCS reads
SG_NAME=ale-sandbox
BAKE_TYPE="${ALE_BAKE_INSTANCE_TYPE:-ecs.g7.2xlarge}"

ensure_bucket() {                    # OSS import bucket exists (region-local)
  ossutil ls "oss://$BUCKET/" >/dev/null 2>&1 || ossutil mb "oss://$BUCKET" --region "$R" >/dev/null
}

ensure_raw() {                       # $1 family → ensure oss://$BUCKET/images/$1.raw
  local key="images/$1.raw"
  if ossutil stat "oss://$BUCKET/$key" >/dev/null 2>&1; then echo "raw present: $1"; return; fi
  # ossutil (unlike `aws s3 cp -`) cannot read stdin — it stat()s the source —
  # so we can't pipe GCS→tar→OSS directly. Stage the decompressed raw to a local
  # scratch dir (needs ~170 GiB free), upload, then delete. ALE_IMG_SCRATCH
  # overrides the scratch location (default /var/tmp/aliimg).
  local scratch="${ALE_IMG_SCRATCH:-/var/tmp/aliimg}" raw
  mkdir -p "$scratch"; raw="$scratch/$1.raw"
  say "stage $GCS/$1.tar.gz -> $raw (decompress)"
  gsutil -u "$BILLING" cat "$GCS/$1.tar.gz" | tar -xzO > "$raw"
  [ -s "$raw" ] || { echo "staging produced an empty raw for $1" >&2; return 1; }
  say "upload $raw -> oss://$BUCKET/$key ($(stat -c %s "$raw") bytes)"
  ossutil cp -f "$raw" "oss://$BUCKET/$key"
  rm -f "$raw"
}

vsw_in_sg_vpc() {                    # echo a vSwitch id in the SG's VPC
  local vpc; vpc=$(aliyun ecs DescribeSecurityGroups --RegionId "$R" \
    --SecurityGroupName "$SG_NAME" | J "['SecurityGroups']['SecurityGroup'][0]['VpcId']")
  aliyun ecs DescribeVSwitches --RegionId "$R" --VpcId "$vpc" \
    | J "['VSwitches']['VSwitch'][0]['VSwitchId']"
}

sg_id() { aliyun ecs DescribeSecurityGroups --RegionId "$R" --SecurityGroupName "$SG_NAME" \
    | J "['SecurityGroups']['SecurityGroup'][0]['SecurityGroupId']"; }

wait_cua() {                         # $1 ip — wait up to 20 min for cua :5000
  local ip=$1
  for _ in $(seq 1 80); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://$ip:5000/status" 2>/dev/null)" = 200 ] \
      && { echo "cua ready at $ip"; return 0; }
    sleep 15
  done
  echo "cua never came up at $ip"; return 1
}

poll_image() {                       # $1 image-id → wait until Available
  while :; do
    local st; st=$(aliyun ecs DescribeImages --RegionId "$R" --ImageId "$1" \
      | J "['Images']['Image'][0]['Status']" 2>/dev/null || echo '?')
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

import_base() {                      # $1 family, $2 OSType, $3 Platform → echo ImageId
  local fam=$1 ostype=$2 platform=$3
  ensure_bucket; ensure_raw "$fam" >&2
  say "$fam: ImportImage (OSType=$ostype Platform=$platform)" >&2
  local img
  img=$(aliyun ecs ImportImage --RegionId "$R" \
    --ImageName "$fam-base-$(date +%s)" --OSType "$ostype" --Platform "$platform" \
    --Architecture x86_64 --BootMode UEFI \
    --DiskDeviceMapping.1.OSSBucket "$BUCKET" \
    --DiskDeviceMapping.1.OSSObject "images/$fam.raw" \
    --DiskDeviceMapping.1.Format RAW | J "['ImageId']")
  poll_image "$img" >&2 || { echo FAILED; return 1; }
  echo "$img"
}

import_linux() {                     # $1 family — import, bake aliyun CLI + ossutil
  local fam=$1 base bake_img sg subnet iid ip final
  base=$(import_base "$fam" linux Ubuntu); [ "$base" = FAILED ] && return 1
  say "$fam: bake aliyun CLI + ossutil (launch base, install via cua, CreateImage)"
  sg=$(sg_id); subnet=$(vsw_in_sg_vpc)
  iid=$(aliyun ecs RunInstances --RegionId "$R" --ImageId "$base" --InstanceType "$BAKE_TYPE" \
    --Amount 1 --VSwitchId "$subnet" --SecurityGroupId "$sg" --InternetMaxBandwidthOut 100 \
    --InstanceName ale-rebake --InstanceChargeType PostPaid --SystemDisk.Category cloud_essd \
    --Tag.1.Key purpose --Tag.1.Value ale-run | J "['InstanceIdSets']['InstanceIdSet'][0]")
  for _ in $(seq 1 40); do ip=$(aliyun ecs DescribeInstances --RegionId "$R" --InstanceIds "[\"$iid\"]" \
    | J "['Instances']['Instance'][0].get('PublicIpAddress',{}).get('IpAddress',['']) [0]" 2>/dev/null); \
    [ -n "$ip" ] && break; sleep 10; done
  wait_cua "$ip" || { aliyun ecs DeleteInstance --RegionId "$R" --InstanceId "$iid" --Force true >/dev/null; return 1; }
  # Install ossutil + aliyun CLI inside the guest, driven through the cua server
  # (same channel the AWS script uses to bake the aws CLI). 900s: cold install.
  local inst='curl -fsSL https://github.com/aliyun/ossutil/releases/download/v1.7.18/ossutil-v1.7.18-linux-amd64.zip -o /tmp/o.zip && cd /tmp && unzip -qo o.zip && sudo install -m755 ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil && curl -fsSL https://github.com/aliyun/aliyun-cli/releases/download/v3.4.2/aliyun-cli-linux-3.4.2-amd64.tgz -o /tmp/a.tgz && tar -xzf /tmp/a.tgz -C /tmp && sudo install -m755 /tmp/aliyun /usr/local/bin/aliyun && /usr/local/bin/ossutil --version && /usr/local/bin/aliyun version'
  curl -s --max-time 900 -X POST "http://$ip:5000/cmd" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys;print(json.dumps({"command":"run_command","params":{"command":sys.argv[1]}}))' "$inst")" \
    | tr '\r' '\n' | grep '^data:' | head -1 | sed 's/^data: //' | J "['stdout'][-60:]" || true
  final=$(aliyun ecs CreateImage --RegionId "$R" --InstanceId "$iid" \
    --ImageName "$fam-$(date +%s)" --Description "$fam + aliyun CLI + ossutil" | J "['ImageId']")
  poll_image "$final"
  tag_image "$final" "$fam"
  # retire base image + builder
  aliyun ecs DeleteImage --RegionId "$R" --ImageId "$base" 2>/dev/null || true
  aliyun ecs DeleteInstance --RegionId "$R" --InstanceId "$iid" --Force true >/dev/null
  echo "LINUX_DONE $fam -> $final"
}

import_windows() {                   # $1 family — import + tag (no bake needed)
  local fam=$1 img
  img=$(import_base "$fam" windows Windows); [ "$img" = FAILED ] && return 1
  tag_image "$img" "$fam"
  echo "WINDOWS_DONE $fam -> $img"
}

case "${1:?usage: import_images.sh ale-ubuntu22|ale-win10|ale-win-server|all}" in
  ale-ubuntu22)   import_linux ale-ubuntu22 ;;
  ale-win10)      import_windows ale-win10 ;;
  ale-win-server) import_windows ale-win-server ;;
  all)            import_linux ale-ubuntu22; import_windows ale-win10; import_windows ale-win-server ;;
  *) echo "unknown image $1"; exit 2 ;;
esac
