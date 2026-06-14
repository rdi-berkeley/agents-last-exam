#!/usr/bin/env bash
# Quantum ESPRESSO 6.7 + BerkeleyGW 4.0 at /opt/qe-bgw-6.7.0-4.0
#   envs/qe-bgw            : QE 6.7 + OpenMPI + gfortran + FFTW3 + ScaLAPACK (conda-forge)
#   src/BerkeleyGW-4.0     : BGW 4.0 built (complex flavor) against the conda toolchain.
# Build recipe mirrors the dev VM (dev-ubuntu22) ground truth: GNU compiler, -DMPI,
# -DUSESCALAPACK -DUSEFFTW3, mpifort/mpicxx/mpicc from the env, -O2.
# BGW 4.0 source: fetched from the public BerkeleyGW distribution mirror by default
# (browser UA required); override with BGW_TARBALL=/path/to/BerkeleyGW-4.0.tar.gz.
set -euo pipefail
MM=/home/kasm-user/.local/bin/micromamba
[ -x "$MM" ] || { echo "[pkg qe-bgw] FATAL: micromamba required first" >&2; exit 1; }
P=/opt/qe-bgw-6.7.0-4.0; ENV="$P/envs/qe-bgw"
export MAMBA_ROOT_PREFIX=/home/kasm-user/.local/share/micromamba

# 1) QE 6.7 conda env (matches dev VM: qe6.7 + openmpi + gfortran + fftw + scalapack)
if [ ! -x "$ENV/bin/pw.x" ]; then
  # gxx is required: BGW compiles C++ (wfn_utils.cpp) via mpicxx, which needs the conda
  # x86_64-conda-linux-gnu-c++. gfortran alone brings the Fortran+C compilers but not C++.
  "$MM" create -y -p "$ENV" -c conda-forge \
    "qe=6.7" "openmpi=4.1.6" gfortran gxx fftw scalapack "libblas=*=*openblas" "python=3.11"
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
test -x "$ENV/bin/pw.x" || { echo "[pkg qe-bgw] FATAL: QE pw.x missing" >&2; exit 1; }

# 2) BerkeleyGW 4.0 (complex flavor) built against the env toolchain
if [ ! -x "$P/src/BerkeleyGW-4.0/bin/epsilon.cplx.x" ]; then
  T="$(mktemp -d)"; TB="${BGW_TARBALL:-}"
  if [ -z "$TB" ] || [ ! -f "$TB" ]; then
    TB="$T/BerkeleyGW-4.0.tar.gz"
    # Public BerkeleyGW 4.0 source mirror (needs a browser User-Agent + redirect follow).
    curl -fsSL -A "Mozilla/5.0" -o "$TB" \
      "https://app.box.com/shared/static/22edl07muvhfnd900tnctsjjftbtcqc4.gz" \
      || { echo "[pkg qe-bgw] FATAL: could not fetch BGW source; set BGW_TARBALL to a local copy." >&2; exit 3; }
  fi
  mkdir -p "$P/src"; tar -xzf "$TB" -C "$P/src"
  # tarball may unpack as BerkeleyGW-4.0 or BerkeleyGW-<hash>; normalize the dir name.
  if [ ! -d "$P/src/BerkeleyGW-4.0" ]; then
    d="$(find "$P/src" -maxdepth 1 -type d -iname 'BerkeleyGW*' | head -1)"
    [ -n "$d" ] && mv "$d" "$P/src/BerkeleyGW-4.0"
  fi
  cd "$P/src/BerkeleyGW-4.0"
  MF="$ENV/bin/mpifort"; MC="$ENV/bin/mpicc"; MX="$ENV/bin/mpicxx"
  cat > arch.mk <<ARCH
COMPFLAG  = -DGNU
PARAFLAG  = -DMPI
MATHFLAG  = -DUSESCALAPACK -DUSEFFTW3
FCPP    = cpp -C -nostdinc
F90free = $MF -ffree-form -ffree-line-length-none -fallow-argument-mismatch
LINK    = $MF
FOPTS   = -O2
FNOOPTS = \$(FOPTS)
MOD_OPT = -J
INCFLAG = -I
C_PARAFLAG = -DPARA
CC_COMP = $MX -std=c++11
C_COMP  = $MC -std=c99
C_LINK  = $MX
C_OPTS  = -O2
REMOVE = /bin/rm -f
FFTWLIB      = -L$ENV/lib -lfftw3
FFTWINCLUDE  = $ENV/include
LAPACKLIB    = -L$ENV/lib -llapack -lblas
SCALAPACKLIB = -L$ENV/lib -lscalapack
ARCH
  cp flavor_cplx.mk flavor.mk
  PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}" make -j"$(nproc)" all
  chown -R 1000:0 "$P" 2>/dev/null || true
  rm -rf "$T"
fi
test -x "$P/src/BerkeleyGW-4.0/bin/epsilon.cplx.x" || { echo "[pkg qe-bgw] FATAL: BGW epsilon.cplx.x missing after build" >&2; exit 1; }
echo "[pkg qe-bgw-6.7.0-4.0] OK (QE 6.7 + BerkeleyGW 4.0 cplx)"
