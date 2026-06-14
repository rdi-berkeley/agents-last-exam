#!/usr/bin/env bash
# CellProfiler 4.2.8 conda-forge/bioconda env at /opt/cellprofiler-4.2.8.
# The task wrapper execs /opt/cellprofiler-4.2.8/bin/cellprofiler DIRECTLY (no conda
# activation), so the Java runtime must be discoverable without `micromamba run`:
#   - openjdk in the env (CellProfiler uses javabridge/bioformats -> needs a JVM),
#   - the env's `java` on PATH (javabridge does `type -p java`),
#   - libjvm.so registered with ldconfig (javabridge._javabridge links against it
#     at import time, before any activation could set LD_LIBRARY_PATH).
# Without these, `cellprofiler --version` aborts with "Error finding javahome" /
# "libjvm.so: cannot open shared object file" (a real break the old presence-only
# `test -x` verify hid).
set -euo pipefail
MM=/home/kasm-user/.local/bin/micromamba
[ -x "$MM" ] || { echo "[pkg cellprofiler-4.2.8] FATAL: micromamba required first" >&2; exit 1; }
P=/opt/cellprofiler-4.2.8
export MAMBA_ROOT_PREFIX=/home/kasm-user/.local/share/micromamba
if [ ! -x "$P/bin/cellprofiler" ]; then
  "$MM" create -y -p "$P" -c bioconda -c conda-forge cellprofiler=4.2.8.1 "openjdk=11"
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
# Ensure openjdk present even if the env predates this change.
if [ ! -x "$P/bin/java" ]; then "$MM" install -y -p "$P" -c conda-forge "openjdk=11"; chown -R 1000:0 "$P" 2>/dev/null || true; fi
# Make Java discoverable for direct (non-activated) invocation of bin/cellprofiler.
ln -sf "$P/bin/java" /usr/local/bin/java
JVMDIR="$(dirname "$(find "$P/lib" -name libjvm.so 2>/dev/null | head -1)")"
[ -n "$JVMDIR" ] || { echo "[pkg cellprofiler-4.2.8] FATAL: libjvm.so not found in env" >&2; exit 1; }
echo "$JVMDIR" > /etc/ld.so.conf.d/cellprofiler-jvm.conf; ldconfig
# Functional check: actually launch CellProfiler (proves Java bridge + bioformats load).
ver="$(QT_QPA_PLATFORM=offscreen "$P/bin/cellprofiler" --version 2>/dev/null | tail -1)"
echo "$ver" | grep -q "4.2.8" || { echo "[pkg cellprofiler-4.2.8] FATAL: cellprofiler --version failed (got: $ver)" >&2; exit 1; }
echo "[pkg cellprofiler-4.2.8] OK (CellProfiler $ver)"
