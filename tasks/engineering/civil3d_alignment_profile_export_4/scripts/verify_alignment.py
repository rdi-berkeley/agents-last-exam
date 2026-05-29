"""VM-side Civil 3D 2024 alignment verifier.

Runs on the Windows VM. Uses AutoCAD COM (AutoCAD.Application.24) to open the
agent's alignment.dwg, then runs a LISP script inside Civil 3D that accesses
the Civil 3D COM API (AeccXUiLand.AeccApplication) to extract alignment
geometry, profile data, and surface elevations.

The LISP-based approach is necessary because:
- Civil 3D COM (AeccXUiLand) is InprocServer32 and cannot be loaded from
  external Python.
- LANDXMLOUT shows a modal dialog that cannot be automated via SendCommand.
- The C3D COM API IS accessible from AutoLISP running inside acad.exe.

Usage:
    python verify_alignment.py --alignment <path> --topo <path> --tsv <path>
                               --work-dir <temp dir for scratch files>

stdout: single JSON object with extracted metrics.
stderr: diagnostic / progress messages.
"""

import argparse
import csv
import json
import os
import sys
import time

import win32con
import win32gui


def _log(msg: str) -> None:
    print(f"[verify] {msg}", file=sys.stderr, flush=True)


def _dismiss_dialogs():
    """Dismiss blocking startup dialogs (Security, Customize Workspace, etc.)."""
    def enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            lower = title.lower()
            if any(k in lower for k in [
                "security", "unsigned", "customize", "recovery",
                "workspace", "welcome", "let's get", "getting started",
                "new tab", "error", "warning",
            ]):
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        return True
    win32gui.EnumWindows(enum_cb, None)


EXTRACT_LISP = r"""
;;; extract_alignment_data.lsp
;;; Extracts alignment geometry, profile info, and surface elevations
;;; using the Civil 3D COM API accessible from inside acad.exe.
;;; Writes results as JSON to the specified output path.

(defun extract-alignment-data (output-path topo-dwg-path tsv-xy-path / fp c3d-app c3d-doc surfs surf-count
                                aligns align-count sites site-count
                                all-aligns target-align)

  (setq fp (open output-path "w"))

  ;; Get the Civil 3D application (versioned ProgID required on C3D 2024)
  (setq c3d-app (vl-catch-all-apply 'vlax-get-or-create-object
                  (list "AeccXUiLand.AeccApplication.13.6")))
  (if (null c3d-app)
    (setq c3d-app (vl-catch-all-apply 'vlax-get-or-create-object
                    (list "AeccXUiLand.AeccApplication"))))
  (if (or (vl-catch-all-error-p c3d-app) (null c3d-app))
    (progn
      (write-line (strcat "{\"error\":\"Cannot get C3D API: "
                          (if (vl-catch-all-error-p c3d-app)
                            (vl-catch-all-error-message c3d-app)
                            "nil") "\"}") fp)
      (close fp)
      (princ)
      (exit)
    )
  )

  (setq c3d-doc (vl-catch-all-apply 'vlax-get-property
                  (list c3d-app "ActiveDocument")))
  (if (or (vl-catch-all-error-p c3d-doc) (null c3d-doc))
    (progn
      (write-line (strcat "{\"error\":\"Cannot get C3D document: "
                          (if (vl-catch-all-error-p c3d-doc)
                            (vl-catch-all-error-message c3d-doc)
                            "nil") "\"}") fp)
      (close fp)
      (princ)
      (exit)
    )
  )

  ;; Collect all alignments from sites and siteless
  (setq all-aligns (list))

  ;; Siteless alignments
  (setq aligns (vl-catch-all-apply 'vlax-get-property
                 (list c3d-doc "AlignmentsSiteless")))
  (if (not (vl-catch-all-error-p aligns))
    (progn
      (setq align-count (vlax-get-property aligns "Count"))
      (if (> align-count 0)
        (vlax-for a aligns
          (setq all-aligns (cons (list a "siteless") all-aligns))
        )
      )
    )
  )

  ;; Site-based alignments
  (setq sites (vl-catch-all-apply 'vlax-get-property
                (list c3d-doc "Sites")))
  (if (not (vl-catch-all-error-p sites))
    (progn
      (setq site-count (vlax-get-property sites "Count"))
      (if (> site-count 0)
        (vlax-for site sites
          (setq site-name (vlax-get-property site "Name"))
          (setq site-aligns (vl-catch-all-apply 'vlax-get-property
                              (list site "Alignments")))
          (if (not (vl-catch-all-error-p site-aligns))
            (progn
              (setq sa-count (vlax-get-property site-aligns "Count"))
              (if (> sa-count 0)
                (vlax-for a site-aligns
                  (setq all-aligns (cons (list a site-name) all-aligns))
                )
              )
            )
          )
        )
      )
    )
  )

  ;; Build JSON for each alignment
  (write-line "{" fp)
  (write-line "\"all_alignments\":[" fp)

  (setq first-align T)
  (setq target-align nil)

  (foreach align-pair all-aligns
    (setq a (car align-pair))
    (setq a-site (cadr align-pair))

    (if (not first-align) (write-line "," fp))
    (setq first-align nil)

    (setq aname (vlax-get-property a "Name"))
    (setq alen (vlax-get-property a "Length"))

    ;; Start and end stations
    (setq sta-start (vl-catch-all-apply 'vlax-get-property (list a "StartingStation")))
    (if (vl-catch-all-error-p sta-start) (setq sta-start 0.0))
    (setq sta-end (vl-catch-all-apply 'vlax-get-property (list a "EndingStation")))
    (if (vl-catch-all-error-p sta-end) (setq sta-end 0.0))

    ;; Start and end points
    (setq sx nil sy nil ex nil ey nil)
    (setq sp (vl-catch-all-apply 'vlax-get-property (list a "StartPoint")))
    (if (not (vl-catch-all-error-p sp))
      (progn
        (setq sx (vlax-safearray-get-element (vlax-variant-value sp) 0))
        (setq sy (vlax-safearray-get-element (vlax-variant-value sp) 1))
      )
    )
    (setq ep (vl-catch-all-apply 'vlax-get-property (list a "EndPoint")))
    (if (not (vl-catch-all-error-p ep))
      (progn
        (setq ex (vlax-safearray-get-element (vlax-variant-value ep) 0))
        (setq ey (vlax-safearray-get-element (vlax-variant-value ep) 1))
      )
    )

    ;; Count sub-entities (curves, spirals, tangents)
    (setq n-curves 0 n-spirals 0 n-tangents 0)
    (setq curves-list (list) spirals-list (list))

    (setq entities (vl-catch-all-apply 'vlax-get-property (list a "Entities")))
    (if (not (vl-catch-all-error-p entities))
      (progn
        (setq ent-count (vlax-get-property entities "Count"))
        (setq eidx 0)
        (vlax-for ent entities
          (setq eidx (1+ eidx))
          (setq etype (vl-catch-all-apply 'vlax-get-property (list ent "Type")))
          (if (vl-catch-all-error-p etype) (setq etype -1))
          (setq ent-len (vl-catch-all-apply 'vlax-get-property (list ent "Length")))
          (if (vl-catch-all-error-p ent-len) (setq ent-len 0.0))

          ;; Type values: 1=Line/Tangent, 2=Arc/Curve, 3=Spiral
          (cond
            ((= etype 2) ; Arc/Curve
              (setq n-curves (1+ n-curves))
              (setq radius (vl-catch-all-apply 'vlax-get-property (list ent "Radius")))
              (if (vl-catch-all-error-p radius) (setq radius 0.0))
              (setq curves-list (cons (list n-curves radius ent-len) curves-list))
            )
            ((= etype 3) ; Spiral
              (setq n-spirals (1+ n-spirals))
              (setq spirals-list (cons (list n-spirals ent-len) spirals-list))
            )
            (T ; Line/Tangent or unknown
              (setq n-tangents (1+ n-tangents))
            )
          )
        )
      )
    )

    ;; Profile info
    (setq profile-count 0 profile-names (list))
    (setq profiles (vl-catch-all-apply 'vlax-invoke-method (list a "GetProfileIds")))
    (if (not (vl-catch-all-error-p profiles))
      (progn
        ;; profiles might be a safearray of ObjectIds
        (setq profile-count (vl-catch-all-apply 'vlax-safearray-get-u-bound
                              (list (vlax-variant-value profiles) 1)))
        (if (vl-catch-all-error-p profile-count)
          (setq profile-count 0)
          (setq profile-count (1+ profile-count))
        )
      )
    )
    ;; Alternative: try Profiles property
    (if (= profile-count 0)
      (progn
        (setq prof-coll (vl-catch-all-apply 'vlax-get-property (list a "Profiles")))
        (if (not (vl-catch-all-error-p prof-coll))
          (progn
            (setq profile-count (vlax-get-property prof-coll "Count"))
            (if (> profile-count 0)
              (vlax-for p prof-coll
                (setq pname (vl-catch-all-apply 'vlax-get-property (list p "Name")))
                (if (not (vl-catch-all-error-p pname))
                  (setq profile-names (cons pname profile-names))
                )
              )
            )
          )
        )
      )
    )

    ;; Write JSON for this alignment
    (write-line "{" fp)
    (write-line (strcat "\"name\":\"" aname "\",") fp)
    (write-line (strcat "\"site\":\"" a-site "\",") fp)
    (write-line (strcat "\"length\":" (rtos alen 2 4) ",") fp)
    (write-line (strcat "\"start_station\":" (rtos sta-start 2 4) ",") fp)
    (write-line (strcat "\"end_station\":" (rtos sta-end 2 4) ",") fp)
    (if sx
      (write-line (strcat "\"start_x\":" (rtos sx 2 4) ",") fp)
      (write-line "\"start_x\":null," fp)
    )
    (if sy
      (write-line (strcat "\"start_y\":" (rtos sy 2 4) ",") fp)
      (write-line "\"start_y\":null," fp)
    )
    (if ex
      (write-line (strcat "\"end_x\":" (rtos ex 2 4) ",") fp)
      (write-line "\"end_x\":null," fp)
    )
    (if ey
      (write-line (strcat "\"end_y\":" (rtos ey 2 4) ",") fp)
      (write-line "\"end_y\":null," fp)
    )
    (write-line (strcat "\"n_curves\":" (itoa n-curves) ",") fp)
    (write-line (strcat "\"n_spirals\":" (itoa n-spirals) ",") fp)
    (write-line (strcat "\"n_tangents\":" (itoa n-tangents) ",") fp)

    ;; Curves array
    (write-line "\"curves\":[" fp)
    (setq first-c T)
    (foreach c (reverse curves-list)
      (if (not first-c) (write-line "," fp))
      (setq first-c nil)
      (write-line (strcat "{\"index\":" (itoa (car c))
        ",\"radius\":" (rtos (cadr c) 2 4)
        ",\"length\":" (rtos (caddr c) 2 4) "}") fp)
    )
    (write-line "]," fp)

    ;; Spirals array
    (write-line "\"spirals\":[" fp)
    (setq first-s T)
    (foreach s (reverse spirals-list)
      (if (not first-s) (write-line "," fp))
      (setq first-s nil)
      (write-line (strcat "{\"index\":" (itoa (car s))
        ",\"length\":" (rtos (cadr s) 2 4) "}") fp)
    )
    (write-line "]," fp)

    ;; Profile info
    (write-line (strcat "\"profile_count\":" (itoa profile-count) ",") fp)
    (write-line "\"profile_names\":[" fp)
    (setq first-p T)
    (foreach pn (reverse profile-names)
      (if (not first-p) (write-line "," fp))
      (setq first-p nil)
      (write-line (strcat "\"" pn "\"") fp)
    )
    (write-line "]" fp)
    (write-line "}" fp)

    ;; Track target alignment
    (if (= aname "Road_01")
      (setq target-align a)
    )
  )

  (write-line "]," fp)

  ;; Surface elevations at TSV coordinates
  ;; Read XY pairs from the tsv-xy-path file (one "x,y" per line)
  (write-line "\"surface_elevations\":[" fp)

  (setq surfs (vl-catch-all-apply 'vlax-get-property (list c3d-doc "Surfaces")))
  (setq eg-surf nil)
  (if (not (vl-catch-all-error-p surfs))
    (progn
      (vlax-for s surfs
        (setq sname (vl-catch-all-apply 'vlax-get-property (list s "Name")))
        (if (and (not (vl-catch-all-error-p sname)) (= sname "EG"))
          (setq eg-surf s)
        )
      )
    )
  )

  (if (and eg-surf (findfile tsv-xy-path))
    (progn
      (setq xy-fp (open tsv-xy-path "r"))
      (setq first-elev T)
      (setq line (read-line xy-fp))
      (while line
        (if (not first-elev) (write-line "," fp))
        (setq first-elev nil)
        ;; Parse "x,y" from line
        (setq comma-pos (vl-string-search "," line))
        (if comma-pos
          (progn
            (setq px (atof (substr line 1 comma-pos)))
            (setq py (atof (substr line (+ comma-pos 2))))
            ;; Query surface elevation
            (setq elev (vl-catch-all-apply 'vlax-invoke-method
                         (list eg-surf "FindElevationAtXY" px py)))
            (if (vl-catch-all-error-p elev)
              (write-line "null" fp)
              (write-line (rtos elev 2 6) fp)
            )
          )
          (write-line "null" fp)
        )
        (setq line (read-line xy-fp))
      )
      (close xy-fp)
    )
  )

  (write-line "]" fp)
  (write-line "}" fp)
  (close fp)
  (princ (strcat "\nExtraction complete: " output-path "\n"))
  (princ)
)
"""


def _get_acad():
    """Get or launch AutoCAD (Civil 3D host) via COM."""
    import win32com.client

    _log("connecting to AutoCAD via COM ...")
    acad = win32com.client.Dispatch("AutoCAD.Application.24")
    time.sleep(3)

    _dismiss_dialogs()
    time.sleep(2)

    try:
        _ = acad.ActiveDocument
        _log("AutoCAD connected, document available")
    except Exception:
        _log("waiting for document ...")
        for _ in range(30):
            _dismiss_dialogs()
            time.sleep(3)
            try:
                _ = acad.ActiveDocument
                break
            except Exception:
                pass

    return acad


def _parse_tsv(tsv_path: str):
    """Parse alignment_profile.tsv."""
    with open(tsv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


def main():
    parser = argparse.ArgumentParser(description="Civil 3D alignment verifier")
    parser.add_argument("--alignment", required=True, help="Path to alignment.dwg")
    parser.add_argument("--topo", required=True, help="Path to topo_surface.dwg")
    parser.add_argument("--tsv", required=True, help="Path to alignment_profile.tsv")
    parser.add_argument("--work-dir", required=True, help="Temp dir for scratch files")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    result = {
        "alignment_dwg_exists": os.path.isfile(args.alignment),
        "tsv_exists": os.path.isfile(args.tsv),
        "all_alignments": [],
        "target_alignment": None,
        "profile_info": None,
        "tsv_headers": [],
        "tsv_row_count": 0,
        "surface_elevations": [],
        "error": None,
    }

    if not result["alignment_dwg_exists"]:
        result["error"] = "alignment.dwg not found"
        print(json.dumps(result))
        return

    if not result["tsv_exists"]:
        result["error"] = "alignment_profile.tsv not found"
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

    # Write XY coordinates file for surface elevation queries
    xy_file = os.path.join(args.work_dir, "tsv_xy.csv")
    with open(xy_file, "w") as f:
        for row in rows:
            try:
                x = float(row.get("x", row.get("X", 0)))
                y = float(row.get("y", row.get("Y", 0)))
                f.write(f"{x},{y}\n")
            except (ValueError, TypeError):
                f.write("0,0\n")

    # Connect to AutoCAD
    try:
        acad = _get_acad()
    except Exception as exc:
        result["error"] = f"Cannot connect to AutoCAD: {exc}"
        print(json.dumps(result))
        return

    # Dismiss any lingering dialogs
    for _ in range(5):
        _dismiss_dialogs()
        time.sleep(2)

    # Close any existing documents
    try:
        while acad.Documents.Count > 0:
            acad.ActiveDocument.Close(False)
            time.sleep(1)
    except Exception:
        pass
    time.sleep(2)

    _dismiss_dialogs()
    time.sleep(2)

    # Cancel any pending commands
    try:
        doc = acad.ActiveDocument
        doc.SendCommand("\x03\x03\x1b\x1b\n")
        time.sleep(3)
    except Exception:
        pass
    _dismiss_dialogs()
    time.sleep(2)

    # Open alignment DWG (with retry for "Call was rejected" errors)
    _log(f"opening {args.alignment} ...")
    open_err = None
    for attempt in range(5):
        try:
            acad.Documents.Open(args.alignment)
            time.sleep(15)
            open_err = None
            break
        except Exception as exc:
            open_err = exc
            _log(f"open attempt {attempt+1} failed: {exc}")
            _dismiss_dialogs()
            time.sleep(5)
    if open_err is not None:
        result["error"] = f"Cannot open alignment.dwg: {open_err}"
        print(json.dumps(result))
        return

    doc = acad.ActiveDocument
    _log(f"active doc: {doc.Name}")

    # If topo_surface.dwg is separate, also open it (for surface Z queries)
    topo_path = os.path.normpath(args.topo)
    align_path = os.path.normpath(args.alignment)
    if topo_path.lower() != align_path.lower() and os.path.isfile(topo_path):
        _log(f"opening {topo_path} for surface data ...")
        try:
            acad.Documents.Open(topo_path)
            time.sleep(10)
        except Exception:
            _log("topo open failed — surface Z queries may not work")

    # Switch back to alignment document
    for i in range(acad.Documents.Count):
        d = acad.Documents.Item(i)
        if os.path.normpath(d.FullName).lower() == align_path.lower():
            d.Activate()
            doc = d
            time.sleep(2)
            break

    # Write LISP extraction script
    lisp_file = os.path.join(args.work_dir, "extract.lsp")
    with open(lisp_file, "w", encoding="utf-8") as f:
        f.write(EXTRACT_LISP)

    # Prepare output file
    extract_out = os.path.join(args.work_dir, "alignment_data.json")
    if os.path.exists(extract_out):
        os.remove(extract_out)

    # Load and run LISP
    lisp_path = lisp_file.replace("\\", "/")
    out_path = extract_out.replace("\\", "/")
    topo_lisp = topo_path.replace("\\", "/")
    xy_lisp = xy_file.replace("\\", "/")

    _log("loading LISP extraction script ...")
    doc.SendCommand(f'(load "{lisp_path}")\n')
    time.sleep(3)

    _log("running extraction ...")
    doc.SendCommand(
        f'(extract-alignment-data "{out_path}" "{topo_lisp}" "{xy_lisp}")\n'
    )

    # Wait for extraction to complete
    for attempt in range(30):
        if os.path.exists(extract_out):
            time.sleep(2)
            break
        time.sleep(3)
    else:
        result["error"] = "LISP extraction timed out"
        print(json.dumps(result))
        return

    # Read extraction results
    try:
        with open(extract_out, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        result["error"] = f"Cannot parse extraction result: {exc}"
        print(json.dumps(result))
        return

    if data.get("error"):
        result["error"] = data["error"]
        print(json.dumps(result))
        return

    # Map extracted data to result format
    result["all_alignments"] = data.get("all_alignments", [])
    result["surface_elevations"] = data.get("surface_elevations", [])

    # Find Road_01
    for a in result["all_alignments"]:
        if a.get("name") == "Road_01":
            result["target_alignment"] = a
            result["profile_info"] = {
                "count": a.get("profile_count", 0),
                "names": a.get("profile_names", []),
            }
            break

    # Close documents
    try:
        while acad.Documents.Count > 0:
            acad.ActiveDocument.Close(False)
            time.sleep(1)
    except Exception:
        pass

    print(json.dumps(result))


if __name__ == "__main__":
    main()
