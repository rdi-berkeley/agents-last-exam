"""researcher_keynote_slide_deck_from_pdfs — GUI-first PowerPoint task."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback only
    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import EvaluationContext, llm_vision_binary_checklist_judge

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

POWERPOINT_EXE = r"C:\Program Files (x86)\Microsoft Office\root\Office16\POWERPNT.EXE"
LIBREOFFICE_EXE = r"C:\Program Files\LibreOffice\program\soffice.exe"
REMOTE_PY_TMP = r"C:\Users\User\AppData\Local\Temp\agenthle_researcher_keynote_inspect.py"
REMOTE_PDF_SUPPORT_TMP = r"C:\Users\User\AppData\Local\Temp\agenthle_researcher_keynote_pdf_support.py"
REMOTE_PDF_RENDER_TMP = r"C:\Users\User\AppData\Local\Temp\agenthle_researcher_keynote_pdf_render.py"
REMOTE_PDF_PAGES_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_researcher_keynote_pages"


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: float | None = None,
    check: bool = False,
) -> dict:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


async def _log_missing_path(
    session: cb.DesktopSession,
    path: str,
    *,
    tag: str,
    label: str,
) -> bool:
    if await session.exists(path):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


async def _file_size_bytes(session: cb.DesktopSession, path: str) -> int:
    result = await _run_command(
        session,
        f'powershell -NoProfile -Command "(Get-Item \'{path}\').Length"',
        timeout=60.0,
        check=False,
    )
    if result["return_code"] != 0:
        logger.error("Failed to stat %s: %s", path, result.get("stderr", "")[:400])
        return 0
    try:
        return int((result.get("stdout") or "").strip())
    except ValueError:
        logger.error("Unexpected size output for %s: %r", path, result.get("stdout"))
        return 0


async def _run_remote_python_script(
    session: cb.DesktopSession,
    *,
    script_path: str,
    script_text: str,
    args: list[str],
    timeout: float = 60.0,
) -> dict:
    await session.write_file(script_path, script_text)
    quoted_args = " ".join(json.dumps(arg) for arg in args)
    try:
        return await _run_command(
            session,
            f"python {json.dumps(script_path)} {quoted_args}",
            timeout=timeout,
            check=False,
        )
    finally:
        await _run_command(
            session,
            f'powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue \'{script_path}\'"',
            timeout=30.0,
            check=False,
        )


async def _remote_python_import_check(
    session: cb.DesktopSession,
    module_name: str,
) -> bool:
    result = await _run_command(
        session,
        f'python -c "import {module_name}"',
        timeout=60.0,
        check=False,
    )
    return result["return_code"] == 0


async def _inspect_pptx_structure(session: cb.DesktopSession, path: str) -> dict:
    py_script = r"""
import json,re,sys
from xml.etree import ElementTree as ET
from zipfile import BadZipFile,ZipFile
p=sys.argv[1]
R={"valid_zip":False,"required_parts_ok":False,"slide_count":0,"notes_count":0,"title_slide_text":"","content_slide_count":0,"visual_slides":0,"notes_compliant_slides":0,"text_density_compliant_slides":0,"paper_coverage_hits":{},"paper_claim_hits":{},"narrative_hits":{},"s9_reference_like_slides":[],"s9_noncompliant_slides":[],"errors":[]}
Q={"[Content_Types].xml","ppt/presentation.xml","docProps/core.xml"}
NS={"a":"http://schemas.openxmlformats.org/drawingml/2006/main","p":"http://schemas.openxmlformats.org/presentationml/2006/main"}
PK={"acl_2024":["winograd","ambiguity","pronoun disambiguation","stable diffusion","bee","flower"],"emnlp_2024":["offensive progressions","sensitivity testing","stop! benchmarking","severity","demographics","convenience store"],"naacl_2025":["time capsules","societal bias","books","fine-tuned llms","tracking societal bias"]}
CK=["dataset","benchmark","prompts","severity","demographics","books","fine-tuned","fine tuned","evaluation","metric","metrics","results","performance","precision","recall","f1","accuracy","bias","iou","intersection over union","hedges"]
NP=[r"\b\d+(?:\.\d+)?%\b",r"\b\d+(?:\.\d+)?\b"]
NK={"hook":["problem of progress","progress in ai","problem statement","motivating problem","motivation"],"overview":["research vision","brief history","overview","journey"],"methods":["method","formalization","dataset","evaluation metrics","approach"],"results":["results","interesting takeaways","error analysis","model specific insights"],"takeaways":["future directions","reflections","interesting takeaways"]}
RM=["http://","https://","www.","doi","arxiv","source:","retrieved from","according to"]
CP=[r"https?://",r"www\.",r"doi",r"arxiv",r"\[[0-9,\-\s]+\]",r"\([A-Z][A-Za-z]+(?:\s+et al\.)?\s+\d{4}\)",r"accessed\s+\d{4}",r"acl anthology"]
n=lambda s:" ".join(s.lower().split())
def ex(i,t,j):
 z=n(t+" "+j[:200]);return i==1 or any(k in z for k in["references","bibliography","thank you","questions?","questions","q&a"])
try:
 with ZipFile(p) as z:
  ns=z.namelist();R["valid_zip"]=True;miss=sorted(Q-set(ns));R["required_parts_ok"]=not miss
  if miss:R["errors"].append("missing_parts:"+",".join(miss))
  R["slide_count"]=sum(1 for x in ns if x.startswith("ppt/slides/slide") and x.endswith(".xml"))
  R["notes_count"]=sum(1 for x in ns if x.startswith("ppt/notesSlides/notesSlide") and x.endswith(".xml"))
  R["paper_coverage_hits"]={k:None for k in PK};R["paper_claim_hits"]={k:None for k in PK};R["narrative_hits"]={k:None for k in NK}
  ss=sorted([x for x in ns if re.match(r"^ppt/slides/slide\d+\.xml$",x)],key=lambda v:int(re.search(r"(\d+)",v).group(1)))
  nl={int(re.search(r"(\d+)",x).group(1)):x for x in ns if re.match(r"^ppt/notesSlides/notesSlide\d+\.xml$",x)}
  for i,sn in enumerate(ss,1):
   rt=ET.fromstring(z.read(sn));tx=[a.text or "" for a in rt.findall(".//a:t",NS)];j=" ".join(tx).strip();t=(tx[0] if tx else "").strip()
   if i==1:R["title_slide_text"]=j
   if ex(i,t,j):continue
   R["content_slide_count"]+=1
   if rt.findall(".//p:pic",NS):R["visual_slides"]+=1
   if len(" ".join(tx[1:]).split())<=75:R["text_density_compliant_slides"]+=1
   nt="";nn=nl.get(i)
   if nn:nt=" ".join((a.text or "") for a in ET.fromstring(z.read(nn)).findall(".//a:t",NS)).strip()
   if len(nt.split())>=20:R["notes_compliant_slides"]+=1
   c=n(j+" "+nt)
   for pn,ks in PK.items():
    pm=any(k in c for k in ks)
    if R["paper_coverage_hits"][pn] is None and pm:R["paper_coverage_hits"][pn]=i
    if R["paper_claim_hits"][pn] is None and pm and (any(k in c for k in CK) or any(re.search(pt,c,re.I) for pt in NP)):R["paper_claim_hits"][pn]=i
   for nm,ks in NK.items():
    if any(k in c for k in ks) and (nm=="takeaways" or R["narrative_hits"][nm] is None):R["narrative_hits"][nm]=i
   if any(m in c for m in RM):
    R["s9_reference_like_slides"].append(i)
    if not any(re.search(pt,c,re.I) for pt in CP):R["s9_noncompliant_slides"].append(i)
except FileNotFoundError:R["errors"].append("missing_file")
except BadZipFile:R["errors"].append("bad_zip")
except Exception as e:R["errors"].append(f"inspection_error:{type(e).__name__}:{e}")
print(json.dumps(R,separators=(',',':')))
"""
    result = await _run_remote_python_script(
        session,
        script_path=REMOTE_PY_TMP,
        script_text=py_script,
        args=[path],
        timeout=60.0,
    )
    if result["return_code"] != 0:
        logger.error("Failed to inspect PPTX structure at %s: %s", path, result.get("stderr", "")[:400])
        return {
            "valid_zip": False,
            "required_parts_ok": False,
            "slide_count": 0,
            "notes_count": 0,
            "errors": ["inspection_failed"],
        }
    try:
        return json.loads((result.get("stdout") or "").strip())
    except json.JSONDecodeError:
        logger.error("Unexpected PPTX inspection output for %s: %r", path, result.get("stdout"))
        return {
            "valid_zip": False,
            "required_parts_ok": False,
            "slide_count": 0,
            "notes_count": 0,
            "errors": ["invalid_inspection_json"],
        }


async def _inspect_pdf_grounding(
    session: cb.DesktopSession,
    *,
    pptx_path: str,
    pdf_paths: list[str],
) -> dict:
    py_script = r"""
import json,re,sys,fitz
from xml.etree import ElementTree as ET
from zipfile import BadZipFile,ZipFile
PPTX=sys.argv[1];PDFS=sys.argv[2:]
NS={"a":"http://schemas.openxmlformats.org/drawingml/2006/main"}
SW={"the","and","for","with","that","this","from","into","their","there","have","has","had","were","was","are","our","your","about","than","then","them","they","you","not","but","can","could","would","should","onto","over","under","between","while","during","also","using","used","use","show","shows","shown","these","those","more","most","less","least","each","such","many","some","very","much","well","paper","papers","slide","slides","research","researcher","researchers","result","results","method","methods","study","studies","finding","findings","large","language","models","model","data","task","tasks"}
PK={"acl_2024":["winograd","ambiguity","pronoun disambiguation","stable diffusion","bee","flower"],"emnlp_2024":["offensive progressions","sensitivity testing","stop! benchmarking","severity","demographics","convenience store"],"naacl_2025":["time capsules","societal bias","books","fine-tuned llms","tracking societal bias"]}
IX={"acl_2024":0,"emnlp_2024":1,"naacl_2025":2}
n=lambda s:" ".join(s.lower().split())
def ex(i,t,j):
 z=n(t+" "+j[:200]);return i==1 or any(k in z for k in["references","bibliography","thank you","questions?","questions","q&a"])
def tok(s):return [x for x in re.findall(r"[a-z0-9][a-z0-9\-']+",s.lower()) if len(x)>=3 and x not in SW]
def ng(ts,k):return {" ".join(ts[i:i+k]) for i in range(max(0,len(ts)-k+1)) if all(len(t)>=4 for t in ts[i:i+k])}
R={"paper_source_hits":{k:None for k in PK},"paper_source_details":{k:{} for k in PK},"errors":[]}
try:
 PT=[];PS=[]
 for p in PDFS:
  with fitz.open(p) as d:tx=n(" ".join(pg.get_text("text") for pg in d))
  PT.append(tx);PS.append(set(tok(tx)))
 with ZipFile(PPTX) as z:
  ns=z.namelist()
  ss=sorted([x for x in ns if re.match(r"^ppt/slides/slide\d+\.xml$",x)],key=lambda v:int(re.search(r"(\d+)",v).group(1)))
  nl={int(re.search(r"(\d+)",x).group(1)):x for x in ns if re.match(r"^ppt/notesSlides/notesSlide\d+\.xml$",x)}
  for i,sn in enumerate(ss,1):
   rt=ET.fromstring(z.read(sn));tx=[a.text or "" for a in rt.findall(".//a:t",NS)];j=" ".join(tx).strip();t=(tx[0] if tx else "").strip()
   if ex(i,t,j):continue
   nt="";nn=nl.get(i)
   if nn:nt=" ".join((a.text or "") for a in ET.fromstring(z.read(nn)).findall(".//a:t",NS)).strip()
   c=n(j+" "+nt);cts=tok(c)
   if len(cts)<6:continue
   tri=ng(cts,3);quad=ng(cts,4)
   for pn,ks in PK.items():
    if R["paper_source_hits"][pn] is not None or not any(k in c for k in ks):continue
    pi=IX[pn];pt=PT[pi];ps=PS[pi]
    ov=sorted({t for t in cts if len(t)>=5 and t in ps})[:12]
    th=sorted([x for x in tri if x in pt])[:4];qh=sorted([x for x in quad if x in pt])[:4]
    if len(ov)>=6 and (qh or th or len(ov)>=10):
     R["paper_source_hits"][pn]=i;R["paper_source_details"][pn]={"slide_index":i,"overlap_tokens_sample":ov,"tri_hits_sample":th,"quad_hits_sample":qh}
except FileNotFoundError as e:R["errors"].append(f"missing_file:{e}")
except BadZipFile:R["errors"].append("bad_zip")
except Exception as e:R["errors"].append(f"support_error:{type(e).__name__}:{e}")
print(json.dumps(R,separators=(',',':')))
"""
    result = await _run_remote_python_script(
        session,
        script_path=REMOTE_PDF_SUPPORT_TMP,
        script_text=py_script,
        args=[pptx_path, *pdf_paths],
        timeout=180.0,
    )
    if result["return_code"] != 0:
        logger.error("Failed to inspect PDF grounding for %s: %s", pptx_path, result.get("stderr", "")[:400])
        return {
            "paper_source_hits": {},
            "paper_source_details": {},
            "errors": ["support_failed"],
        }
    try:
        return json.loads((result.get("stdout") or "").strip())
    except json.JSONDecodeError:
        logger.error("Unexpected PDF grounding output for %s: %r", pptx_path, result.get("stdout"))
        return {
            "paper_source_hits": {},
            "paper_source_details": {},
            "errors": ["invalid_support_json"],
        }


async def _render_pdf_pages_remote(
    session: cb.DesktopSession,
    *,
    pdf_path: str,
    output_dir: str,
    dpi: int = 144,
) -> list[bytes]:
    """Render PDF pages to PNGs on the remote VM, then pull PNG bytes back locally."""
    py_script = r"""
import json, os, sys, fitz
pdf_path=sys.argv[1]
output_dir=sys.argv[2]
dpi=int(sys.argv[3])
fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)
os.makedirs(output_dir, exist_ok=True)
for name in os.listdir(output_dir):
    if name.lower().endswith(".png"):
        try:
            os.remove(os.path.join(output_dir, name))
        except OSError:
            pass
doc=fitz.open(pdf_path)
mat=fitz.Matrix(dpi/72, dpi/72)
paths=[]
try:
    for idx, page in enumerate(doc, start=1):
        out_path=os.path.join(output_dir, f"slide_{idx:03d}.png")
        page.get_pixmap(matrix=mat, alpha=False).save(out_path)
        paths.append(out_path)
finally:
    doc.close()
print(json.dumps({"page_paths": paths}, separators=(",", ":")))
"""
    result = await _run_remote_python_script(
        session,
        script_path=REMOTE_PDF_RENDER_TMP,
        script_text=py_script,
        args=[pdf_path, output_dir, str(dpi)],
        timeout=180.0,
    )
    if result["return_code"] != 0:
        logger.error(
            "Failed to render PDF pages remotely for %s: %s",
            pdf_path,
            result.get("stderr", "")[:400],
        )
        return []
    try:
        payload = json.loads((result.get("stdout") or "").strip())
    except json.JSONDecodeError:
        logger.error("Unexpected remote PDF render output for %s: %r", pdf_path, result.get("stdout"))
        return []
    page_paths = payload.get("page_paths", []) or []
    page_pngs: list[bytes] = []
    for page_path in page_paths:
        if not await session.exists(page_path):
            logger.error("Remote rendered page missing: %s", page_path)
            return []
        page_pngs.append(await session.read_bytes(page_path))
    return page_pngs


async def _judge_slide_pages(
    *,
    slide_pngs: list[bytes],
    tag: str,
    ctx: EvaluationContext,
) -> dict:
    """Judge slide pages with per-page binary checklists."""
    if not slide_pngs:
        return {
            "title_result": None,
            "content_results": [],
            "closing_result": None,
            "content_slide_count": 0,
            "coherence_ratio": 0.0,
            "density_ratio": 0.0,
            "visual_ratio": 0.0,
            "audience_ratio": 0.0,
            "strong_slide_ratio": 0.0,
        }

    total_pages = len(slide_pngs)
    title_result = await llm_vision_binary_checklist_judge(
        prompt_intro=(
            f"You are evaluating slide 1 of {total_pages} from an academic keynote deck. "
            "Judge only what is visibly present on this slide image."
        ),
        checklist_items=[
            ("is_title_slide", "Is this clearly a keynote-style title or opening slide?"),
            (
                "has_researcher_name",
                "Is the speaker name 'Ali Emami' visibly present or unmistakably readable on the slide?",
            ),
            (
                "has_affiliation",
                "Does the slide visibly show an affiliation line mentioning Emory University?",
            ),
            (
                "polished_opening",
                "Does the slide look polished and presentation-ready rather than like a rough draft or placeholder?",
            ),
        ],
        image_bytes=slide_pngs[0],
        max_tokens=256,
        eval_context=ctx,
        identifier="slide_01_title_checklist",
    )

    content_results: list[dict] = []
    for slide_index, slide_png in enumerate(slide_pngs[1:-1], start=2):
        content_results.append(
            await llm_vision_binary_checklist_judge(
                prompt_intro=(
                    f"You are evaluating slide {slide_index} of {total_pages} from a researcher keynote deck "
                    "intended for a general computer science audience. Judge only what is visibly present on this slide image."
                ),
                checklist_items=[
                    (
                        "clear_message",
                        "Does this slide appear to communicate one reasonably clear main idea with a readable heading or focal point?",
                    ),
                    (
                        "reasonable_density",
                        "Is the slide reasonably concise for a keynote talk, without looking overloaded by dense paragraphs or tiny unreadable text?",
                    ),
                    (
                        "visual_or_structure",
                        "Does the slide use a meaningful visual, chart, table, diagram, timeline, or other deliberate structure instead of being plain text only?",
                    ),
                    (
                        "coherent_non_placeholder",
                        "Does the slide look coherent and intentional, rather than broken, garbled, or placeholder content?",
                    ),
                    (
                        "general_audience_fit",
                        "Does this slide look appropriate for a polished keynote aimed at a broad computer science audience rather than a raw lab meeting slide?",
                    ),
                ],
                image_bytes=slide_png,
                max_tokens=320,
                eval_context=ctx,
                identifier=f"slide_{slide_index:02d}_content_checklist",
            )
        )

    closing_result = None
    if total_pages >= 2:
        closing_result = await llm_vision_binary_checklist_judge(
            prompt_intro=(
                f"You are evaluating the final slide ({total_pages} of {total_pages}) from an academic keynote deck. "
                "Judge only what is visibly present on this slide image."
            ),
            checklist_items=[
                (
                    "closing_or_takeaway",
                    "Does this look like a plausible closing, takeaway, thank-you, or Q&A slide for a keynote deck?",
                ),
                (
                    "polished_closing",
                    "Does the slide look polished and presentation-ready rather than unfinished?",
                ),
            ],
            image_bytes=slide_pngs[-1],
            max_tokens=192,
            eval_context=ctx,
            identifier=f"slide_{total_pages:02d}_closing_checklist",
        )

    content_slide_count = len(content_results)

    def _ratio(item_key: str) -> float:
        if content_slide_count == 0:
            return 0.0
        return sum(
            result.get("checklist_scores", {}).get(item_key, 0.0)
            for result in content_results
        ) / content_slide_count

    strong_slide_ratio = 0.0
    if content_slide_count:
        strong_slide_ratio = sum(
            1 for result in content_results if result.get("score", 0.0) >= 0.8
        ) / content_slide_count

    metrics = {
        "title_result": title_result,
        "content_results": content_results,
        "closing_result": closing_result,
        "content_slide_count": content_slide_count,
        "coherence_ratio": _ratio("coherent_non_placeholder"),
        "density_ratio": _ratio("reasonable_density"),
        "visual_ratio": _ratio("visual_or_structure"),
        "audience_ratio": _ratio("general_audience_fit"),
        "strong_slide_ratio": strong_slide_ratio,
    }
    logger.info("[%s] Page-image VLM metrics: %s", tag, metrics)
    return metrics


async def _powerpoint_openable(
    session: cb.DesktopSession,
    *,
    pptx_path: str,
) -> tuple[bool, dict]:
    command = (
        "powershell -NoProfile -Command "
        "\"$ErrorActionPreference='Stop'; "
        "$ppt = New-Object -ComObject PowerPoint.Application; "
        "$presentation = $ppt.Presentations.Open('%s',$false,$true,$false); "
        "$slides = $presentation.Slides.Count; "
        "$presentation.Close(); "
        "$ppt.Quit(); "
        "Write-Output $slides\""
    ) % pptx_path.replace("\\", "\\\\")
    result = await _run_command(
        session,
        command,
        timeout=180.0,
        check=False,
    )
    opened = result["return_code"] == 0 and (result.get("stdout") or "").strip().isdigit()
    return opened, result


@dataclass
class ResearcherKeynoteTaskConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = r"E:\agenthle"
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "researcher_keynote_slide_deck_from_pdfs"
    VARIANT_NAME: str = "base"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def output_submission_dir(self) -> str:
        return rf"{self.remote_output_dir}\submission"

    @property
    def output_pptx(self) -> str:
        return rf"{self.output_submission_dir}\final.pptx"

    @property
    def libreoffice_smoke_dir(self) -> str:
        return rf"{self.remote_output_dir}\libreoffice_smoke"

    @property
    def libreoffice_smoke_pdf(self) -> str:
        return rf"{self.libreoffice_smoke_dir}\final.pdf"

    @property
    def context_file(self) -> str:
        return rf"{self.input_dir}\context - Ali.txt"

    @property
    def reference_pptx(self) -> str:
        return rf"{self.reference_dir}\Research_Talk_Y - Ali.pptx"

    @property
    def pdf_inputs(self) -> list[str]:
        return [
            rf"{self.input_dir}\2024.acl-long.22 (1) - Ali.pdf",
            rf"{self.input_dir}\2024.emnlp-main.243.pdf",
            rf"{self.input_dir}\2025.naacl-long.118 - Ali.pdf",
        ]

    @property
    def task_description(self) -> str:
        pdf_lines = "\n".join(f"- `{path}`" for path in self.pdf_inputs)
        return f"""\
You are preparing a keynote slide deck in Microsoft PowerPoint for a researcher.

## Primary Software
- Use Microsoft PowerPoint as the primary authoring surface:
  `{POWERPOINT_EXE}`
- LibreOffice Impress may be used only as a secondary fallback for opening/render checks:
  `{LIBREOFFICE_EXE}`

## Input Files
{pdf_lines}
- Context and framing metadata: `{self.context_file}`

## Output Requirements
1. Create the keynote deck in the PowerPoint GUI on Windows.
2. Save the final deck exactly to:
   `{self.output_pptx}`
3. The saved file must be a non-empty `.pptx` that PowerPoint can open.

## Important Notes
- This benchmark is software-oriented and GUI-first.
- Do not treat Python slide libraries as the primary authoring workflow.
- Use the staged PDFs and context file to synthesize the deck content.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "context_file": self.context_file,
                "pdf_inputs": self.pdf_inputs,
                "reference_pptx": self.reference_pptx,
                "output_submission_dir": self.output_submission_dir,
                "output_pptx": self.output_pptx,
                "libreoffice_smoke_dir": self.libreoffice_smoke_dir,
                "libreoffice_smoke_pdf": self.libreoffice_smoke_pdf,
                "powerpoint_exe": POWERPOINT_EXE,
                "libreoffice_exe": LIBREOFFICE_EXE,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    cfg = ResearcherKeynoteTaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": "windows"},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _libreoffice_convertible(
    session: cb.DesktopSession,
    *,
    pptx_path: str,
    soffice_exe: str,
    outdir: str,
    expected_pdf: str,
) -> tuple[bool, dict]:
    await session.makedirs(outdir)
    await _run_command(
        session,
        f'powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue \'{expected_pdf}\'"',
        timeout=60.0,
        check=False,
    )
    result = await _run_command(
        session,
        f'"{soffice_exe}" --headless --convert-to pdf --outdir "{outdir}" "{pptx_path}"',
        timeout=180.0,
        check=False,
    )
    if result["return_code"] != 0:
        return False, result
    pdf_size = await _file_size_bytes(session, expected_pdf)
    return pdf_size > 0, result


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]
    output_pptx = meta["output_pptx"]
    libreoffice_exe = meta["libreoffice_exe"]
    required_name = "ali emami"
    required_affiliation = "emory university"

    if not await session.exists(output_pptx):
        logger.error("[%s] Missing agent deck at %s", tag, output_pptx)
        logger.warning(
            "[%s] Current evaluator requires a real authored PPTX deck and will not proceed "
            "to PPTX/PDF/page-image checks without it.",
            tag,
        )
        return [0.0]

    size_bytes = await _file_size_bytes(session, output_pptx)
    if size_bytes <= 0:
        logger.error("[%s] Agent deck exists but is empty: %s", tag, output_pptx)
        logger.warning(
            "[%s] Current evaluator requires a non-empty PPTX deck before structural, "
            "rendered-page, and VLM checklist checks can run.",
            tag,
        )
        return [0.0]

    inspection = await _inspect_pptx_structure(session, output_pptx)
    if not inspection.get("valid_zip"):
        logger.error("[%s] Agent deck is not a valid OOXML zip: %s | %s", tag, output_pptx, inspection)
        return [0.0]
    if not inspection.get("required_parts_ok"):
        logger.error("[%s] Agent deck is missing required PPTX parts: %s | %s", tag, output_pptx, inspection)
        return [0.0]

    slide_count = int(inspection.get("slide_count", 0))
    notes_count = int(inspection.get("notes_count", 0))
    title_slide_text = str(inspection.get("title_slide_text", "")).lower()
    content_slide_count = int(inspection.get("content_slide_count", 0))
    paper_coverage_hits = inspection.get("paper_coverage_hits", {}) or {}
    paper_claim_hits = inspection.get("paper_claim_hits", {}) or {}
    narrative_hits = inspection.get("narrative_hits", {}) or {}
    s9_reference_like_slides = inspection.get("s9_reference_like_slides", []) or []
    s9_noncompliant_slides = inspection.get("s9_noncompliant_slides", []) or []

    if slide_count < 11:
        logger.error("[%s] Agent deck has too few slides (%d < 11): %s", tag, slide_count, output_pptx)
        return [0.0]
    if notes_count < 1:
        logger.error("[%s] Agent deck has no notes slides: %s", tag, output_pptx)
        return [0.0]
    if required_name not in title_slide_text or required_affiliation not in title_slide_text:
        logger.error(
            "[%s] Title slide is missing canonical researcher metadata (%s / %s): %s",
            tag,
            required_name,
            required_affiliation,
            output_pptx,
        )
        return [0.0]
    if content_slide_count <= 0:
        logger.error("[%s] No content slides detected after title/reference exclusion: %s", tag, output_pptx)
        return [0.0]

    missing_papers = [paper for paper, hit in paper_coverage_hits.items() if not hit]
    if missing_papers:
        logger.error(
            "[%s] S3 paper coverage approximation failed; missing paper clusters %s: %s",
            tag,
            missing_papers,
            output_pptx,
        )
        return [0.0]

    missing_claim_papers = [paper for paper, hit in paper_claim_hits.items() if not hit]
    if missing_claim_papers:
        logger.error(
            "[%s] S8 approximation failed; missing claim-like paper coverage for %s: %s",
            tag,
            missing_claim_papers,
            output_pptx,
        )
        return [0.0]

    pdf_grounding = await _inspect_pdf_grounding(
        session,
        pptx_path=output_pptx,
        pdf_paths=meta["pdf_inputs"],
    )
    paper_source_hits = pdf_grounding.get("paper_source_hits", {}) or {}
    paper_source_details = pdf_grounding.get("paper_source_details", {}) or {}
    pdf_grounding_errors = pdf_grounding.get("errors", []) or []
    if pdf_grounding_errors:
        logger.error(
            "[%s] S8 source-grounded PDF support inspection failed for %s: %s",
            tag,
            output_pptx,
            pdf_grounding_errors,
        )
        return [0.0]

    missing_source_papers = [paper for paper, hit in paper_source_hits.items() if not hit]
    if missing_source_papers:
        logger.error(
            "[%s] S8 source-grounded support missing for paper clusters %s: %s | details=%s",
            tag,
            missing_source_papers,
            output_pptx,
            paper_source_details,
        )
        return [0.0]

    required_sections = ["hook", "overview", "methods", "results", "takeaways"]
    missing_sections = [name for name in required_sections if not narrative_hits.get(name)]
    if missing_sections:
        logger.error(
            "[%s] S4 narrative arc approximation missing sections %s: %s",
            tag,
            missing_sections,
            output_pptx,
        )
        return [0.0]

    hook_idx = int(narrative_hits["hook"])
    overview_idx = int(narrative_hits["overview"])
    methods_idx = int(narrative_hits["methods"])
    results_idx = int(narrative_hits["results"])
    takeaways_idx = int(narrative_hits["takeaways"])
    if not (hook_idx < takeaways_idx and overview_idx <= takeaways_idx and methods_idx <= results_idx):
        logger.error(
            "[%s] S4 narrative arc approximation order invalid (hook=%s overview=%s methods=%s results=%s takeaways=%s): %s",
            tag,
            hook_idx,
            overview_idx,
            methods_idx,
            results_idx,
            takeaways_idx,
            output_pptx,
        )
        return [0.0]

    if s9_noncompliant_slides:
        logger.error(
            "[%s] S9 approximation found reference-like slides without citation-like evidence %s: %s",
            tag,
            s9_noncompliant_slides,
            output_pptx,
        )
        return [0.0]

    if not await session.exists(libreoffice_exe):
        logger.error("[%s] LibreOffice executable missing for S10 approximation: %s", tag, libreoffice_exe)
        return [0.0]

    powerpoint_ok, powerpoint_result = await _powerpoint_openable(
        session,
        pptx_path=output_pptx,
    )
    if not powerpoint_ok:
        logger.warning(
            "[%s] PowerPoint COM open probe failed for %s; falling back to LibreOffice-only S10 gate on this VM: %s",
            tag,
            output_pptx,
            {
                "return_code": powerpoint_result.get("return_code"),
                "stdout": (powerpoint_result.get("stdout") or "")[:400],
                "stderr": (powerpoint_result.get("stderr") or "")[:400],
            },
        )

    libreoffice_ok, libreoffice_result = await _libreoffice_convertible(
        session,
        pptx_path=output_pptx,
        soffice_exe=libreoffice_exe,
        outdir=meta["libreoffice_smoke_dir"],
        expected_pdf=meta["libreoffice_smoke_pdf"],
    )
    if not libreoffice_ok:
        logger.error(
            "[%s] LibreOffice headless convert failed for %s: %s",
            tag,
            output_pptx,
            {
                "return_code": libreoffice_result.get("return_code"),
                "stdout": (libreoffice_result.get("stdout") or "")[:400],
                "stderr": (libreoffice_result.get("stderr") or "")[:400],
            },
        )
        return [0.0]

    libreoffice_pdf_size = await _file_size_bytes(session, meta["libreoffice_smoke_pdf"])
    if libreoffice_pdf_size < 1024:
        logger.error(
            "[%s] Converted PDF is unexpectedly small after LibreOffice export: %s",
            tag,
            meta["libreoffice_smoke_pdf"],
        )
        return [0.0]

    slide_pngs = await _render_pdf_pages_remote(
        session,
        pdf_path=meta["libreoffice_smoke_pdf"],
        output_dir=REMOTE_PDF_PAGES_DIR,
    )
    if not slide_pngs:
        logger.error(
            "[%s] Remote PDF page rendering failed or produced no slide images: %s",
            tag,
            meta["libreoffice_smoke_pdf"],
        )
        return [0.0]
    if len(slide_pngs) != slide_count:
        logger.warning(
            "[%s] Slide-count mismatch between PPTX XML (%d) and converted PDF pages (%d); continuing with PDF pages.",
            tag,
            slide_count,
            len(slide_pngs),
        )

    async with EvaluationContext(
        task_tag=tag,
        mode="custom",
        output_dir=None,
        target_path=output_pptx,
    ) as ctx:
        ctx.log_evaluation(
            identifier="gate_output_exists",
            score=1.0,
            file_size_bytes=size_bytes,
            pptx_slide_count=slide_count,
            pdf_page_count=len(slide_pngs),
        )
        page_metrics = await _judge_slide_pages(slide_pngs=slide_pngs, tag=tag, ctx=ctx)
        ctx.finalize(num_output_files=1)

    title_result = page_metrics["title_result"] or {}
    closing_result = page_metrics["closing_result"] or {}
    title_score = float(title_result.get("score", 0.0))
    closing_score = float(closing_result.get("score", 1.0)) if slide_pngs else 0.0
    density_ratio = float(page_metrics["density_ratio"])
    visual_ratio = float(page_metrics["visual_ratio"])
    coherence_ratio = float(page_metrics["coherence_ratio"])
    audience_ratio = float(page_metrics["audience_ratio"])
    strong_slide_ratio = float(page_metrics["strong_slide_ratio"])

    if title_score < 1.0:
        logger.error("[%s] Title slide VLM checklist failed: %s", tag, title_result)
        return [0.0]
    if coherence_ratio < 0.8:
        logger.error("[%s] Page-image coherence ratio too low (%.3f < 0.8): %s", tag, coherence_ratio, output_pptx)
        return [0.0]
    if density_ratio < 0.75:
        logger.error("[%s] Page-image density ratio too low (%.3f < 0.75): %s", tag, density_ratio, output_pptx)
        return [0.0]
    if visual_ratio < 0.5:
        logger.error("[%s] Page-image visual/structure ratio too low (%.3f < 0.5): %s", tag, visual_ratio, output_pptx)
        return [0.0]
    if audience_ratio < 0.75:
        logger.error("[%s] Page-image audience-fit ratio too low (%.3f < 0.75): %s", tag, audience_ratio, output_pptx)
        return [0.0]
    if strong_slide_ratio < 0.5:
        logger.error(
            "[%s] Too few strong content slides under VLM checklist (%.3f < 0.5): %s",
            tag,
            strong_slide_ratio,
            output_pptx,
        )
        return [0.0]
    if slide_pngs and closing_score < 0.5:
        logger.error("[%s] Final slide VLM checklist failed: %s", tag, closing_result)
        return [0.0]

    logger.info(
        "[%s] Found structurally valid PPTX output at %s (%d bytes, %d slides, %d notes slides, "
        "content=%d, page_density_ratio=%.3f, page_visual_ratio=%.3f, page_coherence_ratio=%.3f, "
        "page_audience_ratio=%.3f, page_strong_ratio=%.3f, title_vlm=%.3f, closing_vlm=%.3f, "
        "coverage=%s, claim_hits=%s, source_hits=%s, narrative=%s, s9_reference_like=%s, "
        "powerpoint_open=%s, libreoffice_convert=ok)",
        tag,
        output_pptx,
        size_bytes,
        slide_count,
        notes_count,
        content_slide_count,
        density_ratio,
        visual_ratio,
        coherence_ratio,
        audience_ratio,
        strong_slide_ratio,
        title_score,
        closing_score,
        paper_coverage_hits,
        paper_claim_hits,
        paper_source_hits,
        narrative_hits,
        s9_reference_like_slides,
        "ok" if powerpoint_ok else "probe_failed",
    )
    logger.warning(
        "[%s] Evaluator now gates on PPTX OOXML structure, title-slide researcher metadata, "
        "S3 paper coverage approximation, S4 narrative arc approximation, S8 claim-like paper-grounding plus "
        "PDF-source support checks via PyMuPDF, >=11 slide XML files, >=1 notes slide, LibreOffice headless convert success, "
        "and per-page VLM checklist review over rendered slide images for title quality, content coherence, "
        "text density, visual/structural support, audience fit, and closing-slide plausibility. "
        "A PowerPoint COM open probe is attempted as extra evidence, but on this VM it is not yet stable enough to serve as a hard gate.",
        tag,
    )
    return [1.0]
