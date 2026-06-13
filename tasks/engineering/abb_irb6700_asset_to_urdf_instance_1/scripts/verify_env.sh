#!/usr/bin/env bash
# Verify — abb_irb6700: stdlib XML/CSV (URDF authoring); staged meshes/metadata present
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/engineering/abb_irb6700_asset_to_urdf_instance_1/base}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
/usr/bin/python - <<'PY' || fail "stdlib probe failed"
import sys, csv, xml.etree.ElementTree as ET
r=ET.Element("robot"); ET.SubElement(r,"link",{"name":"base"}); ET.tostring(r)
print("[verify] python", sys.version.split()[0], "xml/csv URDF authoring OK")
PY
test -d "$CANON/input/meshes" && echo "[verify] staged meshes/ present" || fail "input/meshes missing"
echo "[verify] PASS"
