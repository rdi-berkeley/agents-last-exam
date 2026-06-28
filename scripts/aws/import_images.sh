#!/usr/bin/env bash
# Import ALE sandbox images from gs://ale-data-public/images/ into AWS as public
# AMIs. One re-runnable script for the whole flow (replaces the old per-image
# scripts). Idempotent-ish: skips the S3 transfer if the raw is already there.
#
#   ./import_images.sh ale-ubuntu22       # Linux: import-snapshot+register, bake aws CLI
#   ./import_images.sh ale-win10          # Windows 10 desktop: BYOL
#   ./import_images.sh ale-win-server     # Windows Server: license-included
#   ./import_images.sh all                # all three
#
# Per image it: streams the GCS tar.gz (disk.raw) into the S3 import bucket
# (no local staging), turns it into an AMI, tags it `ale:image-family=<name>`
# (the tag AwsProvider resolves), and makes it + its snapshot public in us-east-1.
#
# Why per-OS paths: import-image validates the guest OS and rejects ALE's Ubuntu
# HWE kernel, so Linux goes import-snapshot + register-image (no OS check) and
# then bakes the aws CLI in (needed for s3:// data/output). Windows goes
# import-image with the right --license-type (Win10 client = BYOL only;
# Win Server = AWS license-included so it runs on cheap shared tenancy).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
R=us-east-1
ACCT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=ale-images-$ACCT
GCS=gs://ale-data-public/images
SG_NAME=ale-sandbox
say() { printf '\n=== %s ===\n' "$*"; }
J() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)"; }

ensure_raw() {                       # $1 family → ensure s3://$BUCKET/images/$1.raw
  local key="images/$1.raw"
  if aws s3 ls "s3://$BUCKET/$key" >/dev/null 2>&1; then echo "raw present: $1"; return; fi
  say "stream $GCS/$1.tar.gz -> s3://$BUCKET/$key"
  aws configure set default.s3.multipart_chunksize 512MB
  gsutil cat "$GCS/$1.tar.gz" | tar -xzO | aws s3 cp - "s3://$BUCKET/$key"
}

subnet_in_sg_vpc() {                 # echo a subnet id in the SG's VPC
  local vpc; vpc=$(aws ec2 describe-security-groups --region $R \
    --filters Name=group-name,Values=$SG_NAME --query 'SecurityGroups[0].VpcId' --output text)
  aws ec2 describe-subnets --region $R \
    --filters "Name=vpc-id,Values=$vpc" "Name=tag:project,Values=ale" \
    --query 'Subnets[0].SubnetId' --output text
}

wait_cua() {                         # $1 ip — wait up to 20 min for cua :5000
  local ip=$1
  for _ in $(seq 1 80); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://$ip:5000/status" 2>/dev/null)" = 200 ] \
      && { echo "cua ready at $ip"; return 0; }
    sleep 15
  done
  echo "cua never came up at $ip"; return 1
}

publish() {                          # $1 ami, $2 family — tag + make public (+ snapshot)
  aws ec2 create-tags --region $R --resources "$1" \
    --tags Key=ale:image-family,Value=$2 Key=Name,Value=$2
  aws ec2 modify-image-attribute --region $R --image-id "$1" --launch-permission "Add=[{Group=all}]" || \
    echo "NOTE: could not make $1 public (license-included Windows may be account-private; others copy-image instead)"
  local snap; snap=$(aws ec2 describe-images --region $R --image-ids "$1" \
    --query 'Images[0].BlockDeviceMappings[0].Ebs.SnapshotId' --output text)
  aws ec2 modify-snapshot-attribute --region $R --snapshot-id "$snap" \
    --create-volume-permission "Add=[{Group=all}]" 2>/dev/null || true
  echo "published $1 ($2)"
}

poll_import_snapshot() {             # $1 task → echo SnapshotId
  while :; do
    read -r st pr snap < <(aws ec2 describe-import-snapshot-tasks --region $R --import-task-ids "$1" \
      --query 'ImportSnapshotTasks[0].SnapshotTaskDetail.[Status,Progress,SnapshotId]' --output text)
    printf '  snap %s %s%%\n' "$st" "${pr:-?}" >&2
    [ "$st" = completed ] && { echo "$snap"; return; }
    case "$st" in deleted|*[Ee]rror*) echo "FAILED" >&2; return 1;; esac
    sleep 60
  done
}

poll_import_image() {                # $1 task → echo ImageId
  while :; do
    read -r st pr img msg < <(aws ec2 describe-import-image-tasks --region $R --import-task-ids "$1" \
      --query 'ImportImageTasks[0].[Status,Progress,ImageId,StatusMessage]' --output text)
    printf '  image %s %s%% %s\n' "$st" "${pr:-?}" "${msg:-}" >&2
    [ "$st" = completed ] && { echo "$img"; return; }
    case "$st" in deleted|*[Ee]rror*) echo "FAILED" >&2; return 1;; esac
    sleep 60
  done
}

import_linux() {                     # $1 family (ale-ubuntu22)
  local fam=$1; ensure_raw "$fam"
  say "$fam: import-snapshot (bypasses import-image OS/kernel check)"
  local cont task snap ami0
  cont=$(mktemp); printf '{"Description":"%s","Format":"raw","UserBucket":{"S3Bucket":"%s","S3Key":"images/%s.raw"}}' "$fam" "$BUCKET" "$fam" > "$cont"
  task=$(aws ec2 import-snapshot --region $R --description "$fam" --disk-container "file://$cont" --query ImportTaskId --output text)
  rm -f "$cont"; echo "import-snapshot task: $task"
  snap=$(poll_import_snapshot "$task"); [ "$snap" = FAILED ] && return 1
  say "$fam: register base AMI from $snap"
  ami0=$(aws ec2 register-image --region $R --name "$fam-base-$(date +%s)" \
    --architecture x86_64 --root-device-name /dev/sda1 --virtualization-type hvm \
    --ena-support --boot-mode uefi \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={SnapshotId=$snap,VolumeType=gp3,DeleteOnTermination=true}" \
    --query ImageId --output text)
  say "$fam: bake aws CLI (launch base, install, create-image)"
  local subnet sg iid ip ami1
  subnet=$(subnet_in_sg_vpc)
  sg=$(aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=$SG_NAME --query 'SecurityGroups[0].GroupId' --output text)
  iid=$(aws ec2 run-instances --region $R --image-id "$ami0" --instance-type m6i.xlarge --count 1 \
    --subnet-id "$subnet" --security-group-ids "$sg" --associate-public-ip-address \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ale-rebake},{Key=purpose,Value=ale-run}]' \
    --query 'Instances[0].InstanceId' --output text)
  for _ in $(seq 1 40); do ip=$(aws ec2 describe-instances --region $R --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null); [ -n "$ip" ] && [ "$ip" != None ] && break; sleep 10; done
  wait_cua "$ip" || { aws ec2 terminate-instances --region $R --instance-ids "$iid" >/dev/null; return 1; }
  local inst='curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/a.zip && cd /tmp && unzip -q -o a.zip && sudo ./aws/install --update && rm -rf /tmp/aws /tmp/a.zip && /usr/local/bin/aws --version'
  # 900s: the aws CLI install (~100MB download + unzip + sudo install) can take
  # several minutes on a cold instance; too short and curl drops before the
  # single SSE `data:` reply, aborting the script (set -o pipefail) + leaking the builder.
  curl -s --max-time 900 -X POST "http://$ip:5000/cmd" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys;print(json.dumps({"command":"run_command","params":{"command":sys.argv[1]}}))' "$inst")" \
    | tr '\r' '\n' | grep '^data:' | head -1 | sed 's/^data: //' | J "['stdout'][-60:]"
  ami1=$(aws ec2 create-image --region $R --instance-id "$iid" --name "$fam-$(date +%s)" \
    --description "$fam + aws CLI (public)" --query ImageId --output text)
  poll_import_image_state "$ami1"
  publish "$ami1" "$fam"
  # retire base AMI + builder
  aws ec2 deregister-image --region $R --image-id "$ami0" 2>/dev/null || true
  aws ec2 delete-snapshot --region $R --snapshot-id "$snap" 2>/dev/null || true
  aws ec2 terminate-instances --region $R --instance-ids "$iid" >/dev/null
  echo "LINUX_DONE $fam -> $ami1"
}

poll_import_image_state() {          # $1 ami — wait until available
  while :; do
    local st; st=$(aws ec2 describe-images --region $R --image-ids "$1" --query 'Images[0].State' --output text)
    echo "  $1 $st" >&2; [ "$st" = available ] && return; [ "$st" = failed ] && return 1; sleep 30
  done
}

import_windows() {                   # $1 family, $2 license (BYOL|AWS)
  local fam=$1 lic=$2; ensure_raw "$fam"
  say "$fam: import-image (Windows, --license-type $lic)"
  local cont task ami
  cont=$(mktemp); printf '[{"Description":"%s","Format":"raw","UserBucket":{"S3Bucket":"%s","S3Key":"images/%s.raw"}}]' "$fam" "$BUCKET" "$fam" > "$cont"
  task=$(aws ec2 import-image --region $R --description "$fam" --platform Windows \
    --architecture x86_64 --license-type "$lic" --boot-mode uefi \
    --disk-containers "file://$cont" --query ImportTaskId --output text)
  rm -f "$cont"; echo "import-image task: $task"
  ami=$(poll_import_image "$task"); [ "$ami" = FAILED ] && return 1
  publish "$ami" "$fam"
  echo "WINDOWS_DONE $fam -> $ami"
}

case "${1:?usage: import_images.sh ale-ubuntu22|ale-win10|ale-win-server|all}" in
  ale-ubuntu22)   import_linux ale-ubuntu22 ;;
  ale-win10)      import_windows ale-win10 BYOL ;;
  ale-win-server) import_windows ale-win-server AWS ;;
  all)            import_linux ale-ubuntu22; import_windows ale-win10 BYOL; import_windows ale-win-server AWS ;;
  *) echo "unknown image $1"; exit 2 ;;
esac
