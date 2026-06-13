#!/usr/bin/env bash
# Verify — go_game: the sabaki AppImage extracts and its Electron binary has all
# shared libraries resolved (ldd clean). A full GUI launch needs the Kasm X
# display (present at runtime); here we prove the runtime libs are complete.
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/go_game_reconstruction_1/base}"
APP=$(ls "$CANON"/software/sabaki-*.AppImage 2>/dev/null | head -1)
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -n "$APP" || fail "sabaki AppImage not found"
chmod +x "$APP" 2>/dev/null || true
W="$(mktemp -d)"; cd "$W"
"$APP" --appimage-extract >/dev/null 2>&1 || fail "AppImage --appimage-extract failed (libfuse2?)"
BIN=$(ls squashfs-root/sabaki squashfs-root/AppRun 2>/dev/null | head -1)
test -n "$BIN" || BIN=$(find squashfs-root -maxdepth 1 -type f -executable | head -1)
MISS=$(ldd "$BIN" 2>/dev/null | grep -i "not found" || true)
# also check the bundled chrome-sandbox / electron core lib if present
ELE=$(find squashfs-root -maxdepth 2 -name 'sabaki' -type f 2>/dev/null | head -1)
[ -n "$ELE" ] && MISS="$MISS$(ldd "$ELE" 2>/dev/null | grep -i 'not found' || true)"
if [ -n "$MISS" ]; then echo "[verify] missing libs:"; echo "$MISS"; fail "Electron binary has unresolved shared libraries"; fi
echo "[verify] sabaki AppImage extracted; Electron binary shared libs all resolved"
echo "[verify] PASS"
