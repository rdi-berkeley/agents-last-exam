#!/usr/bin/env bash
# Poll an in-progress `ec2 import-snapshot` task, then `register-image` an AMI
# from the resulting EBS snapshot.
#
# Why this path instead of `import-image`: import-image validates the guest OS
# (kernel whitelist, Windows checks) and rejects images it doesn't recognise —
# e.g. ALE's Ubuntu HWE kernel 6.5.0-15-generic fails with "Unsupported kernel
# version". import-snapshot imports the raw disk as a plain snapshot with NO OS
# validation; register-image then builds a bootable AMI with the ENA + UEFI
# attributes Nitro needs. The guest kernel still has nvme/ena modules, so it
# boots; we reach it via cua-server, not AWS's blessed login path.
#
# Usage: register_from_snapshot.sh <import-snap-task-id> <ami-name> <out-file>
#   e.g. register_from_snapshot.sh import-snap-0abc ale-ubuntu22 tools/aws/.ubuntu-ami
set -euo pipefail
HERE="$(dirname "$0")"
# shellcheck disable=SC1091
source "${HERE}/.aws-env"

TASK="${1:?import-snap task id}"
NAME="${2:?ami name}"
OUT="${3:?output file for ami id}"
say() { printf '\n=== %s ===\n' "$*"; }

say "poll import-snapshot $TASK"
SNAP=""
while :; do
  read -r STATUS PROGRESS MSG SNAP < <(aws ec2 describe-import-snapshot-tasks \
    --region "$AWS_REGION" --import-task-ids "$TASK" \
    --query 'ImportSnapshotTasks[0].SnapshotTaskDetail.[Status,Progress,StatusMessage,SnapshotId]' \
    --output text)
  printf '%s  status=%s progress=%s  %s\n' "$(date +%H:%M:%S)" \
    "$STATUS" "${PROGRESS:-?}" "${MSG:-}"
  case "$STATUS" in
    completed) echo "snapshot: $SNAP"; break ;;
    deleted|*[Ee]rror*) echo "snapshot import FAILED: $MSG"; exit 1 ;;
  esac
  sleep 60
done

say "register-image from $SNAP (HVM, ENA, UEFI, gp3)"
# Root device /dev/sda1 is the AWS HVM convention; the snapshot is the boot vol.
AMI=$(aws ec2 register-image --region "$AWS_REGION" \
  --name "${NAME}-$(date +%s)" \
  --description "ALE ${NAME} (import-snapshot + register-image)" \
  --architecture x86_64 \
  --root-device-name /dev/sda1 \
  --virtualization-type hvm \
  --ena-support \
  --boot-mode uefi \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={SnapshotId=${SNAP},VolumeType=gp3,DeleteOnTermination=true}" \
  --query ImageId --output text)
echo "$AMI" > "$OUT"
say "DONE — AMI $AMI written to $OUT"
