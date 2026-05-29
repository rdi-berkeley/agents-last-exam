"""VM-side Civil 3D 2024 alignment verifier.

Runs on the Windows VM. Uses AutoCAD COM (AutoCAD.Application.24) to open the
agent's alignment.dwg and extract alignment data via AutoLISP commands sent
through SendCommand. Surface elevations are queried from the topo_surface.dwg.

Usage:
    python verify_alignment.py --alignment <path> --topo <path> --tsv <path>
                               --work-dir <temp dir for intermediate files>

stdout: single JSON object with extracted metrics.
stderr: diagnostic / progress messages.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time

try:
    import pythoncom
    import win32com.client
except ImportError:
    pass


def _log(msg: str) -> None:
    print(f"[verify] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# AutoCAD COM helpers
# ---------------------------------------------------------------------------

def _get_acad(max_wait=180):
    """Get or launch AutoCAD (Civil 3D host) via COM."""
    pythoncom.CoInitialize()

    # Try GetActiveObject first
    for progid in ("AutoCAD.Application.24", "AutoCAD.Application"):
        try:
            acad = win32com.client.GetActiveObject(progid)
            _log(f"connected to running AutoCAD via {progid}")
            return acad
        except Exception:
            continue

    # Launch via Dispatch
    _log("launching AutoCAD via Dispatch ...")
    acad = win32com.client.Dispatch("AutoCAD.Application.24")
    try:
        acad.Visible = True
    except AttributeError:
        pass

    for i in range(max_wait // 2):
        try:
            _ = acad.Documents
            _log("AutoCAD launched and ready")
            return acad
        except Exception:
            time.sleep(2)

    raise RuntimeError("AutoCAD failed to become ready")


def _retry_com(fn, retries=20, delay=5):
    """Retry a COM call that may fail with 'Call was rejected by callee'."""
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            code = getattr(e, 'args', [None])[0] if hasattr(e, 'args') else None
            if code in (-2147418111, -2147023179) and i < retries - 1:
                time.sleep(delay)
            else:
                raise


def _open_dwg(acad, dwg_path, wait_regen=30):
    """Open a DWG file in AutoCAD and wait for it to load.

    Returns the document object, or None on failure.
    """
    _log(f"opening {os.path.basename(dwg_path)} ...")

    doc = None
    for attempt in range(15):
        try:
            doc = acad.Documents.Open(dwg_path)
            break
        except Exception as e:
            _log(f"  open attempt {attempt+1}: {e}")
            time.sleep(10)

    if doc is None:
        return None

    # Wait for doc name to be accessible
    for i in range(90):
        try:
            name = doc.Name
            _log(f"  doc loaded: {name}")
            break
        except Exception:
            if i % 15 == 0:
                _log(f"  still loading... {i*2}s")
            time.sleep(2)
    else:
        _log("  doc never became accessible")
        return None

    # Wait for regeneration
    time.sleep(wait_regen)
    return doc


def _send_command(doc, cmd, retries=15):
    """Send a command to AutoCAD with retry logic."""
    for i in range(retries):
        try:
            doc.SendCommand(cmd)
            return True
        except Exception as e:
            if i < retries - 1:
                time.sleep(3)
    return False


def _extract_alignment_via_lisp(doc, output_file):
    """Extract alignment data using AutoLISP commands.

    Writes alignment info to output_file as JSON.
    Returns True on success.
    """
    if os.path.exists(output_file):
        os.remove(output_file)

    # Escape backslashes for LISP string
    lisp_out = output_file.replace("\\", "\\\\")

    # AutoLISP script to extract Civil 3D alignment data
    # Uses vlax functions to iterate objects and extract alignment properties
    lisp_code = f'''
(defun c:EXTRACT-ALIGNMENT-DATA ( / civapp civdoc aligns naligns i aln
  aname alength startsta endsta startpt endpt fp subents j subent
  subtype radius sublength curves spirals tangents profcount)
  (vl-load-com)
  (setq fp (open "{lisp_out}" "w"))
  (if (null fp)
    (progn (princ "\\nERROR: Cannot open output file") (princ))
    (progn
      (setq curves '() spirals '() tangents '())
      (setq aname "" alength 0.0 startsta 0.0 endsta 0.0)
      (setq startpt nil endpt nil profcount 0 naligns 0)

      ;; Try to get Civil 3D application
      (setq civapp nil)
      (if (vlax-method-applicable-p (vlax-get-acad-object) 'GetInterfaceObject)
        (progn
          (setq civapp
            (vl-catch-all-apply 'vla-GetInterfaceObject
              (list (vlax-get-acad-object) "AeccXUiLand.AeccApplication.13.6")))
          (if (vl-catch-all-error-p civapp) (setq civapp nil))))

      (if civapp
        (progn
          ;; Get civil document and alignments collection
          (setq civdoc (vlax-get-property civapp 'ActiveDocument))
          (setq aligns (vlax-get-property civdoc 'AlignmentsSiteless))
          (if (null aligns)
            (setq aligns (vl-catch-all-apply 'vlax-get-property
              (list civdoc 'Alignments))))
          (if (and aligns (not (vl-catch-all-error-p aligns)))
            (progn
              (setq naligns (vlax-get-property aligns 'Count))
              (if (> naligns 0)
                (progn
                  ;; Get first alignment
                  (setq aln (vlax-invoke-method aligns 'Item 0))
                  (setq aname (vlax-get-property aln 'Name))
                  (setq alength (vlax-get-property aln 'Length))
                  (setq startsta (vlax-get-property aln 'StartingStation))
                  (setq endsta (vlax-get-property aln 'EndingStation))

                  ;; Get start/end points
                  (setq startpt (vlax-invoke-method aln 'PointLocation startsta))
                  (setq endpt (vlax-invoke-method aln 'PointLocation endsta))

                  ;; Count sub-entities (curves, spirals, tangents)
                  (setq subents (vlax-get-property aln 'Entities))
                  (setq j 0)
                  (repeat (vlax-get-property subents 'Count)
                    (setq subent (vlax-invoke-method subents 'Item j))
                    (setq subtype (vlax-get-property subent 'Type))
                    (setq sublength (vlax-get-property subent 'Length))
                    (cond
                      ((= subtype 2) ;; Arc/Curve
                        (setq radius (vlax-get-property subent 'Radius))
                        (setq curves (cons (list radius sublength) curves)))
                      ((= subtype 3) ;; Spiral
                        (setq spirals (cons sublength spirals)))
                      ((= subtype 1) ;; Tangent/Line
                        (setq tangents (cons sublength tangents))))
                    (setq j (1+ j)))

                  ;; Count profiles
                  (setq profcount 0)
                  (setq profs (vl-catch-all-apply 'vlax-get-property
                    (list aln 'Profiles)))
                  (if (and profs (not (vl-catch-all-error-p profs)))
                    (setq profcount (vlax-get-property profs 'Count))))))))

      ;; Write JSON output
      (write-line "{{" fp)
      (write-line (strcat "  \\"alignment_count\\": " (itoa naligns) ",") fp)
      (write-line (strcat "  \\"name\\": \\"" aname "\\",") fp)
      (write-line (strcat "  \\"length\\": " (rtos alength 2 4) ",") fp)
      (write-line (strcat "  \\"start_station\\": " (rtos startsta 2 4) ",") fp)
      (write-line (strcat "  \\"end_station\\": " (rtos endsta 2 4) ",") fp)
      (if startpt
        (progn
          (write-line (strcat "  \\"start_x\\": " (rtos (car startpt) 2 4) ",") fp)
          (write-line (strcat "  \\"start_y\\": " (rtos (cadr startpt) 2 4) ",") fp))
        (progn
          (write-line "  \\"start_x\\": null," fp)
          (write-line "  \\"start_y\\": null," fp)))
      (if endpt
        (progn
          (write-line (strcat "  \\"end_x\\": " (rtos (car endpt) 2 4) ",") fp)
          (write-line (strcat "  \\"end_y\\": " (rtos (cadr endpt) 2 4) ",") fp))
        (progn
          (write-line "  \\"end_x\\": null," fp)
          (write-line "  \\"end_y\\": null," fp)))
      (write-line (strcat "  \\"n_curves\\": " (itoa (length curves)) ",") fp)
      (write-line (strcat "  \\"n_spirals\\": " (itoa (length spirals)) ",") fp)
      (write-line (strcat "  \\"n_tangents\\": " (itoa (length tangents)) ",") fp)
      (write-line (strcat "  \\"profile_count\\": " (itoa profcount) ",") fp)

      ;; Write curves array
      (write-line "  \\"curves\\": [" fp)
      (setq i 0)
      (foreach c (reverse curves)
        (if (> i 0) (write-line "," fp))
        (write-line (strcat "    {{\\"radius\\": " (rtos (car c) 2 4)
          ", \\"length\\": " (rtos (cadr c) 2 4) "}}") fp)
        (setq i (1+ i)))
      (write-line "  ]," fp)

      ;; Write spirals array
      (write-line "  \\"spirals\\": [" fp)
      (setq i 0)
      (foreach s (reverse spirals)
        (if (> i 0) (write-line "," fp))
        (write-line (strcat "    {{\\"length\\": " (rtos s 2 4) "}}") fp)
        (setq i (1+ i)))
      (write-line "  ]" fp)

      (write-line "}}" fp)
      (close fp)
      (princ (strcat "\\nAlignment data written to " "{lisp_out}"))
      (princ))))
'''

    # Write LISP file
    lisp_file = output_file.replace(".json", ".lsp")
    with open(lisp_file, "w", encoding="utf-8") as f:
        f.write(lisp_code)

    _log("loading LISP extraction script ...")
    lisp_file_escaped = lisp_file.replace("\\", "/")
    if not _send_command(doc, f'(load "{lisp_file_escaped}")\n'):
        _log("failed to load LISP script")
        return False

    time.sleep(5)

    _log("running EXTRACT-ALIGNMENT-DATA ...")
    if not _send_command(doc, "EXTRACT-ALIGNMENT-DATA\n"):
        _log("failed to run extraction command")
        return False

    # Wait for output file
    for i in range(30):
        if os.path.exists(output_file) and os.path.getsize(output_file) > 10:
            _log(f"extraction complete: {os.path.getsize(output_file)} bytes")
            return True
        time.sleep(2)

    _log("extraction timed out — no output file")
    return False


def _extract_surface_elevations_via_lisp(doc, xy_pairs, output_file):
    """Query surface Z elevations at XY points using AutoLISP.

    Returns True on success.
    """
    if os.path.exists(output_file):
        os.remove(output_file)

    lisp_out = output_file.replace("\\", "\\\\")

    # Build the XY pairs as LISP data
    xy_str = " ".join(f"({x:.4f} {y:.4f})" for x, y in xy_pairs)

    lisp_code = f'''
(defun c:QUERY-SURFACE-Z ( / civapp civdoc surfs surf fp pts x y z)
  (vl-load-com)
  (setq fp (open "{lisp_out}" "w"))
  (if (null fp)
    (progn (princ "\\nERROR: Cannot open output file") (princ))
    (progn
      (write-line "[" fp)
      (setq pts (list {xy_str}))
      (setq civapp nil)
      (if (vlax-method-applicable-p (vlax-get-acad-object) 'GetInterfaceObject)
        (progn
          (setq civapp
            (vl-catch-all-apply 'vla-GetInterfaceObject
              (list (vlax-get-acad-object) "AeccXUiLand.AeccApplication.13.6")))
          (if (vl-catch-all-error-p civapp) (setq civapp nil))))
      (if civapp
        (progn
          (setq civdoc (vlax-get-property civapp 'ActiveDocument))
          (setq surfs (vlax-get-property civdoc 'Surfaces))
          (if (and surfs (> (vlax-get-property surfs 'Count) 0))
            (progn
              (setq surf (vlax-invoke-method surfs 'Item 0))
              (setq i 0)
              (foreach pt pts
                (if (> i 0) (write-line "," fp))
                (setq x (car pt) y (cadr pt))
                (setq z (vl-catch-all-apply 'vlax-invoke-method
                  (list surf 'FindElevationAtXY x y)))
                (if (vl-catch-all-error-p z)
                  (write-line "  null" fp)
                  (write-line (strcat "  " (rtos z 2 6)) fp))
                (setq i (1+ i)))))))
      (write-line "]" fp)
      (close fp)
      (princ (strcat "\\nSurface elevations written to " "{lisp_out}"))
      (princ))))
'''

    lisp_file = output_file.replace(".json", ".lsp")
    with open(lisp_file, "w", encoding="utf-8") as f:
        f.write(lisp_code)

    _log("loading surface query LISP ...")
    lisp_file_escaped = lisp_file.replace("\\", "/")
    if not _send_command(doc, f'(load "{lisp_file_escaped}")\n'):
        return False

    time.sleep(5)

    _log("querying surface elevations ...")
    if not _send_command(doc, "QUERY-SURFACE-Z\n"):
        return False

    for i in range(60):
        if os.path.exists(output_file) and os.path.getsize(output_file) > 5:
            _log(f"surface query complete: {os.path.getsize(output_file)} bytes")
            return True
        time.sleep(2)

    _log("surface query timed out")
    return False


# ---------------------------------------------------------------------------
# TSV parser
# ---------------------------------------------------------------------------

def _parse_tsv(tsv_path: str):
    """Parse alignment_metrics.tsv."""
    with open(tsv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Civil 3D alignment verifier")
    parser.add_argument("--alignment", required=True, help="Path to alignment.dwg")
    parser.add_argument("--topo", required=True, help="Path to topo_surface.dwg")
    parser.add_argument("--tsv", required=True, help="Path to alignment_metrics.tsv")
    parser.add_argument("--work-dir", required=True, help="Temp dir for intermediate files")
    parser.add_argument("--tsv-only", action="store_true",
                        help="Skip AutoCAD COM; use TSV-based analysis only")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    result = {
        "alignment_dwg_exists": os.path.isfile(args.alignment),
        "tsv_exists": os.path.isfile(args.tsv),
        "alignment_info": None,
        "profile_info": None,
        "tsv_headers": [],
        "tsv_row_count": 0,
        "surface_elevations": [],
        "tsv_z_values": [],
        "_lisp_extraction_available": False,
        "error": None,
    }

    if not result["alignment_dwg_exists"]:
        result["error"] = "alignment.dwg not found"
        print(json.dumps(result))
        return

    if not result["tsv_exists"]:
        result["error"] = "alignment_metrics.tsv not found"
        print(json.dumps(result))
        return

    # Parse TSV
    try:
        headers, rows = _parse_tsv(args.tsv)
        result["tsv_headers"] = headers
        result["tsv_row_count"] = len(rows)
    except Exception as exc:
        result["error"] = f"TSV parse error: {exc}"
        print(json.dumps(result))
        return

    # TSV-only mode: skip AutoCAD COM entirely
    if args.tsv_only:
        _log("TSV-only mode: skipping AutoCAD COM")
        if rows:
            try:
                first = rows[0]
                last = rows[-1]
                total_length = float(last.get("Station", 0))
                n_curves, est_curves = _estimate_curves_from_tsv(rows)
                result["alignment_info"] = {
                    "alignment_count": 1,
                    "name": "tsv_analysis",
                    "length": total_length,
                    "start_x": float(first.get("X", 0)),
                    "start_y": float(first.get("Y", 0)),
                    "end_x": float(last.get("X", 0)),
                    "end_y": float(last.get("Y", 0)),
                    "n_curves": n_curves,
                    "n_spirals": 0,
                    "n_tangents": 0,
                    "profile_count": 1,
                    "curves": est_curves,
                    "spirals": [],
                    "_fallback": True,
                }
                result["profile_info"] = {"count": 1}
            except Exception as exc:
                _log(f"TSV analysis failed: {exc}")

        for row in rows:
            try:
                result["tsv_z_values"].append(float(row.get("Z", 0)))
            except (ValueError, TypeError):
                result["tsv_z_values"].append(None)

        print(json.dumps(result))
        return

    # Connect to AutoCAD
    try:
        acad = _get_acad()
    except Exception as exc:
        result["error"] = f"Cannot connect to AutoCAD: {exc}"
        print(json.dumps(result))
        return

    time.sleep(10)

    # Open alignment DWG
    doc = _open_dwg(acad, args.alignment, wait_regen=60)
    if doc is None:
        result["error"] = "Failed to open alignment.dwg"
        print(json.dumps(result))
        return

    # Extract alignment data via LISP
    align_json = os.path.join(args.work_dir, "alignment_data.json")
    if _extract_alignment_via_lisp(doc, align_json):
        try:
            with open(align_json, "r", encoding="utf-8") as f:
                ainfo = json.loads(f.read())
            result["alignment_info"] = ainfo
            if ainfo:
                result["profile_info"] = {"count": ainfo.get("profile_count", 0)}
        except Exception as exc:
            _log(f"Failed to parse alignment JSON: {exc}")
    else:
        _log("LISP extraction failed, trying TSV-based fallback...")
        # Fallback: extract basic info from TSV
        if rows:
            try:
                first = rows[0]
                last = rows[-1]
                total_length = float(last.get("Station", 0))
                n_curves, est_curves = _estimate_curves_from_tsv(rows)
                ainfo = {
                    "alignment_count": 1,
                    "name": "tsv_fallback",
                    "length": total_length,
                    "start_x": float(first.get("X", 0)),
                    "start_y": float(first.get("Y", 0)),
                    "end_x": float(last.get("X", 0)),
                    "end_y": float(last.get("Y", 0)),
                    "n_curves": n_curves,
                    "n_spirals": 0,
                    "n_tangents": 0,
                    "profile_count": 1,
                    "curves": est_curves,
                    "spirals": [],
                    "_fallback": True,
                }
                result["alignment_info"] = ainfo
                result["profile_info"] = {"count": 1}
            except Exception as exc:
                _log(f"TSV fallback failed: {exc}")

    # Close alignment doc
    try:
        _retry_com(lambda: doc.Close(False))
    except Exception:
        pass
    time.sleep(5)

    # Open topo surface to query elevations
    xy_pairs = []
    tsv_z_values = []
    for row in rows:
        try:
            x = float(row.get("X", 0))
            y = float(row.get("Y", 0))
            xy_pairs.append((x, y))
        except (ValueError, TypeError):
            xy_pairs.append((0.0, 0.0))
        try:
            tsv_z_values.append(float(row.get("Z", 0)))
        except (ValueError, TypeError):
            tsv_z_values.append(None)

    if xy_pairs:
        doc2 = _open_dwg(acad, args.topo, wait_regen=60)
        if doc2 is not None:
            surf_json = os.path.join(args.work_dir, "surface_elevations.json")
            if _extract_surface_elevations_via_lisp(doc2, xy_pairs, surf_json):
                try:
                    with open(surf_json, "r", encoding="utf-8") as f:
                        result["surface_elevations"] = json.loads(f.read())
                except Exception as exc:
                    _log(f"Failed to parse surface JSON: {exc}")
            try:
                _retry_com(lambda: doc2.Close(False))
            except Exception:
                pass

    result["tsv_z_values"] = tsv_z_values
    result["_lisp_extraction_available"] = bool(
        result["surface_elevations"] and
        result.get("alignment_info") and
        not result["alignment_info"].get("_fallback")
    )

    print(json.dumps(result))


def _estimate_curves_from_tsv(rows):
    """Estimate horizontal curves from TSV XY data.

    Returns (curve_count, curves_list) where each curve has estimated
    radius and arc length derived from heading change analysis.
    """
    if len(rows) < 3:
        return 0, []

    pts = []
    for r in rows:
        try:
            pts.append((float(r["X"]), float(r["Y"]), float(r.get("Station", 0))))
        except (ValueError, TypeError):
            continue

    if len(pts) < 3:
        return 0, []

    headings = []
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        headings.append(math.atan2(dy, dx))

    threshold = 0.02
    in_curve = False
    curves = []
    curve_start = 0

    for i in range(len(headings) - 1):
        dh = abs(headings[i + 1] - headings[i])
        if dh > math.pi:
            dh = 2 * math.pi - dh
        if dh > threshold:
            if not in_curve:
                in_curve = True
                curve_start = i
        else:
            if in_curve:
                in_curve = False
                curve_end = i
                total_dh = 0.0
                for j in range(curve_start, curve_end):
                    d = abs(headings[j + 1] - headings[j])
                    if d > math.pi:
                        d = 2 * math.pi - d
                    total_dh += d
                arc_len = pts[curve_end + 1][2] - pts[curve_start][2]
                if arc_len <= 0:
                    seg_count = curve_end - curve_start + 1
                    arc_len = seg_count * 20.0
                radius = arc_len / total_dh if total_dh > 1e-6 else 9999.0
                curves.append({"radius": round(radius, 2), "length": round(arc_len, 2)})

    if in_curve:
        curve_end = len(headings) - 1
        total_dh = 0.0
        for j in range(curve_start, curve_end):
            d = abs(headings[j + 1] - headings[j])
            if d > math.pi:
                d = 2 * math.pi - d
            total_dh += d
        arc_len = pts[min(curve_end + 1, len(pts) - 1)][2] - pts[curve_start][2]
        if arc_len <= 0:
            seg_count = curve_end - curve_start + 1
            arc_len = seg_count * 20.0
        radius = arc_len / total_dh if total_dh > 1e-6 else 9999.0
        curves.append({"radius": round(radius, 2), "length": round(arc_len, 2)})

    return len(curves), curves


if __name__ == "__main__":
    main()
