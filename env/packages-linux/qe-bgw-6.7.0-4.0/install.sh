#!/usr/bin/env bash
# Deterministic Quantum ESPRESSO 6.7MaX + BerkeleyGW 4.0 Linux x86_64 runtime.
set -euo pipefail

INSTALL_ROOT="${QE_BGW_ROOT:-/opt/qe-bgw-6.7.0-4.0}"
ARCHIVE_NAME="qe-bgw-6.7.0-4.0-linux-x86_64.tar.zst"
ARCHIVE_URL="https://github.com/rdi-berkeley/agents-last-exam/releases/download/runtime-qe-bgw-6.7.0-4.0/$ARCHIVE_NAME"
ARCHIVE_SHA256="d23a8b230214346b039ba9a12090136384c0c72d3fb459f505557ea82bfc0ea3"
ARCHIVE_SOURCE="${QE_BGW_ARCHIVE:-}"
ENV_PREFIX="$INSTALL_ROOT/envs/qe-bgw"
BGW_ROOT="$INSTALL_ROOT/src/BerkeleyGW-4.0"

required_paths() {
  printf '%s\n' \
    "$ENV_PREFIX/bin/pw.x" \
    "$ENV_PREFIX/bin/pw2bgw.x" \
    "$ENV_PREFIX/bin/bands.x" \
    "$ENV_PREFIX/bin/mpirun" \
    "$ENV_PREFIX/bin/python" \
    "$BGW_ROOT/Epsilon/epsilon.cplx.x" \
    "$BGW_ROOT/Sigma/sigma.cplx.x" \
    "$BGW_ROOT/BSE/kernel.cplx.x" \
    "$BGW_ROOT/BSE/absorption.cplx.x" \
    "$BGW_ROOT/BSE/inteqp.cplx.x"
}

verify_runtime() {
  local path missing verify_versions="${1:-1}"
  while IFS= read -r path; do
    test -x "$path" || { echo "[pkg qe-bgw] missing executable: $path" >&2; return 1; }
    missing="$(LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" ldd "$path" | awk '/not found/{print}')"
    test -z "$missing" || {
      printf '%s\n' "$missing" >&2
      echo "[pkg qe-bgw] unresolved shared libraries for $path" >&2
      return 1
    }
  done < <(required_paths)

  for path in \
    "$BGW_ROOT/Epsilon/epsilon.cplx.x" \
    "$BGW_ROOT/Sigma/sigma.cplx.x" \
    "$BGW_ROOT/BSE/kernel.cplx.x" \
    "$BGW_ROOT/BSE/absorption.cplx.x" \
    "$BGW_ROOT/BSE/inteqp.cplx.x"; do
    grep -a -q 'BerkeleyGW' "$path"
  done

  if [ "$verify_versions" -eq 0 ]; then
    return 0
  fi

  OPAL_PREFIX="$ENV_PREFIX" LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    "$ENV_PREFIX/bin/pw.x" -h >"$INSTALL_ROOT/.pw-version.txt" 2>&1 || true
  grep -q 'PWSCF v.6.7MaX' "$INSTALL_ROOT/.pw-version.txt" || {
    cat "$INSTALL_ROOT/.pw-version.txt" >&2
    echo "[pkg qe-bgw] QE version verification failed" >&2
    return 1
  }
  OPAL_PREFIX="$ENV_PREFIX" LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    "$ENV_PREFIX/bin/mpirun" --version >"$INSTALL_ROOT/.mpi-version.txt" 2>&1 || true
  grep -q 'mpirun (Open MPI) 4.1.6' "$INSTALL_ROOT/.mpi-version.txt" || {
    cat "$INSTALL_ROOT/.mpi-version.txt" >&2
    echo "[pkg qe-bgw] MPI version verification failed" >&2
    return 1
  }
  "$ENV_PREFIX/bin/python" --version >"$INSTALL_ROOT/.python-version.txt" 2>&1
  grep -q 'Python 3.11.15' "$INSTALL_ROOT/.python-version.txt" || {
    cat "$INSTALL_ROOT/.python-version.txt" >&2
    echo "[pkg qe-bgw] Python version verification failed" >&2
    return 1
  }
  rm -f "$INSTALL_ROOT/.pw-version.txt" "$INSTALL_ROOT/.mpi-version.txt" "$INSTALL_ROOT/.python-version.txt"
}

if [ -f "$INSTALL_ROOT/.ale-runtime-sha256" ] \
  && grep -qx "$ARCHIVE_SHA256" "$INSTALL_ROOT/.ale-runtime-sha256" \
  && verify_runtime; then
  echo "[pkg qe-bgw-6.7.0-4.0] verified existing runtime"
  exit 0
fi

missing_tools=()
command -v curl >/dev/null 2>&1 || missing_tools+=(curl)
command -v zstd >/dev/null 2>&1 || missing_tools+=(zstd)
if [ "${#missing_tools[@]}" -gt 0 ]; then
  command -v apt-get >/dev/null 2>&1 || {
    echo "[pkg qe-bgw] missing tools and apt-get is unavailable: ${missing_tools[*]}" >&2
    exit 1
  }
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl zstd
fi

mkdir -p "$(dirname "$INSTALL_ROOT")"
TMP_DIR="$(mktemp -d "$(dirname "$INSTALL_ROOT")/.qe-bgw-install.XXXXXX")"
STAGED_ROOT="$TMP_DIR/runtime"
BACKUP_ROOT="$TMP_DIR/previous-runtime"
SWITCHED=0
INSTALL_VERIFIED=0
HAD_PREVIOUS=0
cleanup() {
  if [ "$SWITCHED" -eq 1 ] && [ "$INSTALL_VERIFIED" -ne 1 ]; then
    rm -rf "$INSTALL_ROOT"
    if [ "$HAD_PREVIOUS" -eq 1 ] && [ -e "$BACKUP_ROOT" ]; then
      mv "$BACKUP_ROOT" "$INSTALL_ROOT"
    fi
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [ -n "$ARCHIVE_SOURCE" ]; then
  test -f "$ARCHIVE_SOURCE" || { echo "[pkg qe-bgw] archive not found: $ARCHIVE_SOURCE" >&2; exit 1; }
  cp "$ARCHIVE_SOURCE" "$TMP_DIR/$ARCHIVE_NAME"
else
  curl --fail --location --retry 5 --retry-all-errors \
    --output "$TMP_DIR/$ARCHIVE_NAME" "$ARCHIVE_URL"
fi
printf '%s  %s\n' "$ARCHIVE_SHA256" "$TMP_DIR/$ARCHIVE_NAME" | sha256sum -c -
mkdir -p "$STAGED_ROOT"
tar --use-compress-program=unzstd -xf "$TMP_DIR/$ARCHIVE_NAME" -C "$STAGED_ROOT"

printf '%s\n' "$ARCHIVE_SHA256" >"$STAGED_ROOT/.ale-runtime-sha256"

ENV_PREFIX="$STAGED_ROOT/envs/qe-bgw"
BGW_ROOT="$STAGED_ROOT/src/BerkeleyGW-4.0"
INSTALL_ROOT="$STAGED_ROOT" verify_runtime 0
rm -rf "$STAGED_ROOT"

if [ -e "$INSTALL_ROOT" ]; then
  mv "$INSTALL_ROOT" "$BACKUP_ROOT"
  HAD_PREVIOUS=1
fi
SWITCHED=1
mkdir -p "$INSTALL_ROOT"
tar --use-compress-program=unzstd -xf "$TMP_DIR/$ARCHIVE_NAME" -C "$INSTALL_ROOT"
printf '%s\n' "$ARCHIVE_SHA256" >"$INSTALL_ROOT/.ale-runtime-sha256"
ENV_PREFIX="$INSTALL_ROOT/envs/qe-bgw"
BGW_ROOT="$INSTALL_ROOT/src/BerkeleyGW-4.0"
verify_runtime
INSTALL_VERIFIED=1

echo "[pkg qe-bgw-6.7.0-4.0] OK"
