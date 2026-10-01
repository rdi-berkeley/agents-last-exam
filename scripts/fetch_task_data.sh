#!/usr/bin/env bash
# Fetch a caller-pinned gated task archive for the local Docker provider.
set -euo pipefail
umask 077

REPO="agents-last-exam/agents-last-exam-data-archive"
FILE="ale-tasks-data.tar.gz"
REVISION=""
EXPECTED_SHA256=""
DEST=""
LEGACY=false

usage() {
  printf '%s\n' \
    'Usage: scripts/fetch_task_data.sh --revision COMMIT_SHA --sha256 ARCHIVE_SHA256 [--legacy] [DEST]' \
    '' \
    'Both pins are required: a 40-hex HF commit and the 64-hex archive SHA256.' \
    'No release revision/hash is built in or assumed by this script.' \
    'Default destination: task-data-v1.1; --legacy selects task-data instead.' \
    'Legacy also requires explicit historical revision/hash pins; no fallback to main.' \
    'DEST must be absent or empty, on ext2/ext3/ext4, xfs, or btrfs storage.' \
    'Requires gated HF access, huggingface-cli (or hf), and GNU/Linux core tools.'
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --revision|--sha256)
      (($# >= 2)) || fail "$1 requires a value"
      if [[ "$1" == --revision ]]; then
        [[ -z "$REVISION" ]] || fail 'Duplicate --revision'
        REVISION="$2"
      else
        [[ -z "$EXPECTED_SHA256" ]] || fail 'Duplicate --sha256'
        EXPECTED_SHA256="$2"
      fi
      shift 2
      ;;
    --legacy) LEGACY=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --)
      shift
      (($# == 1)) && [[ -z "$DEST" && -n "$1" ]] || fail 'Expected one destination after --'
      DEST="$1"
      shift
      ;;
    -*) fail "Unknown option: $1" ;;
    *)
      [[ -z "$DEST" && -n "$1" ]] || fail 'Only one nonempty destination is allowed'
      DEST="$1"
      shift
      ;;
  esac
done

[[ "$REVISION" =~ ^[[:xdigit:]]{40}$ ]] || fail '--revision must be an explicit 40-hex HF commit, not a branch/tag'
[[ "$EXPECTED_SHA256" =~ ^[[:xdigit:]]{64}$ ]] || fail '--sha256 must be an explicit 64-hex archive hash'
REVISION="${REVISION,,}"
EXPECTED_SHA256="${EXPECTED_SHA256,,}"
if [[ -z "$DEST" ]]; then
  if $LEGACY; then DEST=task-data; else DEST=task-data-v1.1; fi
fi
for tool in realpath findmnt df free pgrep stat find mktemp tar sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || fail "Required command not found: $tool"
done
if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
elif command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
else
  fail 'huggingface-cli or hf is required; install huggingface_hub and authenticate for gated access'
fi
[[ ! -L "$DEST" ]] || fail 'Destination must not be a symlink'
DEST="$(realpath -m -- "$DEST")"
PARENT="$(dirname -- "$DEST")"

check_destination() {
  local entries
  [[ ! -L "$DEST" ]] || fail 'Destination became a symlink'
  if [[ -e "$DEST" ]]; then
    [[ -d "$DEST" ]] || fail 'Destination exists and is not a directory'
    entries="$(find "$DEST" -mindepth 1 -maxdepth 1 -print -quit)" || fail 'Cannot inspect destination'
    [[ -z "$entries" ]] \
      || fail "Refusing nonempty destination (no old/new mixing): $DEST"
  fi
}

check_storage() {
  local anchor="$1" filesystem
  while [[ ! -e "$anchor" ]]; do anchor="$(dirname -- "$anchor")"; done
  filesystem="$(findmnt -n -o FSTYPE -T "$anchor")" || fail "Cannot inspect storage: $anchor"
  case "$filesystem" in
    ext2|ext3|ext4|xfs|btrfs) ;;
    *) fail "Refusing unverified or memory-backed filesystem $filesystem at $anchor" ;;
  esac
  findmnt -T "$anchor"
  df -h -- "$anchor"
}

check_destination
check_storage "$DEST"
check_storage "$PARENT"
free -h
pgrep -af 'qemu-system|qemu-kvm' || [[ "$?" == 1 ]]
mkdir -p -- "$PARENT"
check_storage "$PARENT"
if [[ -d "$DEST" ]]; then
  [[ "$(stat -c %d -- "$DEST")" == "$(stat -c %d -- "$PARENT")" ]] \
    || fail 'Destination and parent must share a filesystem for atomic installation'
fi
WORK="$(mktemp -d "$PARENT/.ale-fetch.XXXXXXXX")"
cleanup() {
  local status="$?"
  if ((status == 0)); then
    rm -rf -- "$WORK"
  else
    printf 'Fetch failed; destination not installed. Staged evidence retained at %s\n' "$WORK" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -- "$WORK/download" "$WORK/extracted" "$WORK/tmp"
printf '>> Downloading %s at immutable revision %s\n' "$REPO/$FILE" "$REVISION"
if ! TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp" \
  HF_HUB_CACHE="$WORK/hub-cache" HF_XET_CACHE="$WORK/xet-cache" \
  "$HF_CLI" download "$REPO" "$FILE" --repo-type dataset \
  --revision "$REVISION" --local-dir "$WORK/download" >"$WORK/download.log" 2>&1; then
  fail "Pinned download failed; no fallback attempted. See $WORK/download.log"
fi
ARCHIVE="$WORK/download/$FILE"
METADATA="$WORK/download/.cache/huggingface/download/$FILE.metadata"
[[ -f "$ARCHIVE" && ! -L "$ARCHIVE" ]] || fail 'Downloaded archive is missing or is a symlink'
[[ -f "$METADATA" && ! -L "$METADATA" ]] || fail 'HF local download revision metadata is missing or is a symlink'
IFS= read -r ACTUAL_REVISION < "$METADATA" || fail 'HF download revision metadata is invalid'
[[ "$ACTUAL_REVISION" == "$REVISION" ]] || fail 'HF download revision mismatch'
ACTUAL_SHA256="$(sha256sum < "$ARCHIVE")"
ACTUAL_SHA256="${ACTUAL_SHA256%% *}"
[[ "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]] || fail 'Archive SHA256 mismatch; nothing extracted'
printf '>> Verified revision %s and archive SHA256 %s\n' "$REVISION" "$ACTUAL_SHA256"
check_storage "$WORK"
TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp" TAR_OPTIONS= \
  tar --extract --gzip --file "$ARCHIVE" --directory "$WORK/extracted" \
  --no-same-owner --same-permissions
[[ -n "$(find "$WORK/extracted" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  || fail 'Archive contains no task data'
check_destination
mv -T -- "$WORK/extracted" "$DEST"
printf '>> Installed pinned data at %s\n' "$DEST"
printf '>> Select the matching Docker setup with task_data_source: local:%s\n' "$DEST"
