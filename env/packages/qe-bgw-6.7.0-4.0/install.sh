#!/usr/bin/env bash
# Quantum ESPRESSO 6.7 + BerkeleyGW 4.0 at /opt/qe-bgw-6.7.0-4.0 (envs/qe-bgw +
# src/BerkeleyGW-4.0). QE 6.7 + OpenMPI 4.1.6 + gfortran come from conda-forge.
# BerkeleyGW 4.0 source is REGISTRATION-GATED (berkeleygw.org) — provide the
# tarball at $BGW_TARBALL to complete the build; otherwise this exits with a
# clear message (env provisioning is otherwise correct).
set -euo pipefail
MM=/home/kasm-user/.local/bin/micromamba
[ -x "$MM" ] || { echo "[pkg qe-bgw] FATAL: micromamba required first" >&2; exit 1; }
P=/opt/qe-bgw-6.7.0-4.0; ENV="$P/envs/qe-bgw"
export MAMBA_ROOT_PREFIX=/home/kasm-user/.local/share/micromamba
if [ ! -x "$ENV/bin/pw.x" ]; then
  "$MM" create -y -p "$ENV" -c conda-forge "qe=6.7" "openmpi=4.1.6" gfortran fftw scalapack "python=3.11"
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
test -x "$ENV/bin/pw.x" || { echo "[pkg qe-bgw] FATAL: QE pw.x missing" >&2; exit 1; }
if [ ! -x "$P/src/BerkeleyGW-4.0/bin/epsilon.cplx.x" ]; then
  if [ -z "${BGW_TARBALL:-}" ] || [ ! -f "${BGW_TARBALL:-}" ]; then
    echo "[pkg qe-bgw] BLOCKED: BerkeleyGW 4.0 source is registration-gated (berkeleygw.org)." >&2
    echo "[pkg qe-bgw] QE 6.7 env is installed at $ENV. To finish, set BGW_TARBALL=/path/to/BerkeleyGW-4.0.tar.gz and re-run." >&2
    exit 3
  fi
  mkdir -p "$P/src"; tar -xzf "$BGW_TARBALL" -C "$P/src"
  ( cd "$P/src/BerkeleyGW-4.0" && cp config/sample_arch.mk.gnu arch.mk 2>/dev/null || true
    PATH="$ENV/bin:$PATH" make -j"$(nproc)" all-flavors )
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
echo "[pkg qe-bgw-6.7.0-4.0] OK"
