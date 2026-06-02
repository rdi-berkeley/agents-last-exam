from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.utils.evaluation import EvaluationContext, collect_matching_files, llm_vision_judge

logger = logging.getLogger(__name__)

LOCAL_ROOT = Path(__file__).resolve().parent
VARIANTS_ROOT = LOCAL_ROOT / "variants"
FIXTURE_OUTPUT_DIRS = {"output_test_pos", "output_test_neg"}


def win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


EVAL_TMP_ROOT = win_join(r"C:\Users\User\AppData\Local\Temp\agenthle_eval", "odoo")


def remote_basename(path: str) -> str:
    return PureWindowsPath(path).name


def is_fixture_output_dir(path: str) -> bool:
    return remote_basename(path) in FIXTURE_OUTPUT_DIRS


def eval_work_dir(task_tag: str) -> str:
    return win_join(EVAL_TMP_ROOT, task_tag)


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    run_tag: str
    db_name: str
    variant_note: str


VARIANTS = [
    VariantSpec(
        variant_name="odoo_compact",
        run_tag="odoo_compact",
        db_name="odoo_compact",
        variant_note="Recovered compact legacy variant. The shared Odoo workflow is preserved, with this variant using its own DB and searchable run tag.",
    ),
    VariantSpec(
        variant_name="odoo_shifted",
        run_tag="odoo_shifted",
        db_name="odoo_shifted",
        variant_note="Recovered shifted legacy variant. The operational contract matches the recovered shared Odoo workflow while isolating all work in the shifted DB/tag namespace.",
    ),
    VariantSpec(
        variant_name="odoo_stress",
        run_tag="odoo_stress",
        db_name="odoo_stress",
        variant_note="Recovered stress legacy variant. It reuses the recovered end-to-end workflow but keeps its own DB/tag namespace for replay and grading isolation.",
    ),
    VariantSpec(
        variant_name="odoo_hard",
        run_tag="odoo_hard",
        db_name="odoo_hard",
        variant_note="Recovered hard legacy variant. This is the original end-to-end Odoo scenario that the recovered judger and SQL evidence checks were written for.",
    ),
]

VARIANTS_BY_NAME = {spec.variant_name: spec for spec in VARIANTS}


@dataclass
class OdooTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "business_finance"
    TASK_NAME: str = "odoo"
    VARIANT_NAME: str = ""
    RUN_TAG: str = ""
    DEFAULT_DB_NAME: str = ""
    VARIANT_NOTE: str = ""

    DEFAULT_ODOO_URL: str = "http://localhost:8069/web"
    DEFAULT_ODOO_SERVICE_NAME: str = "odoo-server-19.0"
    TEMPLATE_DB_NAME: str = "AgentService"
    DEFAULT_PSQL_PATH: str = r"C:\Program Files\PostgreSQL\17\bin\psql.exe"
    DEFAULT_DROPDB_PATH: str = r"C:\Program Files\PostgreSQL\17\bin\dropdb.exe"
    DEFAULT_CREATEDB_PATH: str = r"C:\Program Files\PostgreSQL\17\bin\createdb.exe"
    DEFAULT_PG_DUMP_PATH: str = r"C:\Program Files\PostgreSQL\17\bin\pg_dump.exe"
    DEFAULT_PG_RESTORE_PATH: str = r"C:\Program Files\PostgreSQL\17\bin\pg_restore.exe"
    DEFAULT_PSQL_ARGS: str = "-h 127.0.0.1 -p 5432 -U openpg"
    DEFAULT_PG_PASSWORD: str = "openpgpwd"
    DEFAULT_ODOO_PYTHON_PATH: str = r"C:\Program Files\Odoo 19.0.20260416\python\python.exe"
    DEFAULT_ODOO_ADMIN_LOGIN: str = "admin@example.com"
    DEFAULT_ODOO_ADMIN_PASSWORD: str = "admin"

    @property
    def variant_source_dir(self) -> Path:
        return VARIANTS_ROOT / self.VARIANT_NAME

    @property
    def prompt_path(self) -> str:
        return win_join(self.task_dir, "input", "prompt.txt")

    @property
    def launch_script_path(self) -> str:
        return win_join(self.software_dir, "launch_odoo.cmd")

    @property
    def task_description(self) -> str:
        # Only the published variant (odoo_compact) ships its prompt.txt with
        # the repo. Unpublished variants have no vendored prompt file, so fall
        # back to the variant note instead of letting a missing file crash the
        # whole module load (load() eagerly builds every variant's description).
        prompt_file = self.variant_source_dir / "input" / "prompt.txt"
        try:
            prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            prompt_text = self.VARIANT_NOTE
        out_dir = self.remote_output_dir
        return f"""\
Goal:
Complete the recovered end-to-end Odoo supply-chain workflow and leave the final business state inside the task database.

Variant:
- {self.VARIANT_NAME}
- {self.VARIANT_NOTE}

Recovered prompt:
{prompt_text}

Inputs / Environment:
- The benchmark data for this variant is staged under {self.task_dir}.
- The VM is expected to provide a working local Odoo web server at {self.DEFAULT_ODOO_URL}.
- If the Odoo service is available, use {self.launch_script_path} to open the correct database in a browser.
- Task setup resets database "{self.DEFAULT_DB_NAME}" from template "{self.TEMPLATE_DB_NAME}" when the PostgreSQL tools are available.
- You must work ONLY inside database "{self.DEFAULT_DB_NAME}".
- Use the run tag "{self.RUN_TAG}" in searchable fields such as Reference / Notes / Customer Ref / Vendor Ref on key documents you create manually.
- The agent-visible task prompt is staged at {self.prompt_path}.
- Odoo web login: {self.DEFAULT_ODOO_ADMIN_LOGIN} / {self.DEFAULT_ODOO_ADMIN_PASSWORD} (credentials are set by task setup).

Warehouses:
- Inbound warehouse code: "WH" (location tree rooted at "WH/Stock")
- Production & Shipping warehouse code: "My Co" (rooted at "My Co/Stock")

Output:
Save these files into {out_dir}
1. lc_split.png
2. invoices.png
3. return_credit.png
4. stock_wh.png
5. submission.txt
"""

    def to_metadata(self) -> dict:
        md = super().to_metadata()
        md.update(
            {
                "run_tag": self.RUN_TAG,
                "odoo_url": self.DEFAULT_ODOO_URL,
                "template_db": self.TEMPLATE_DB_NAME,
                "db_name": self.DEFAULT_DB_NAME,
                "psql_path": self.DEFAULT_PSQL_PATH,
                "dropdb_path": self.DEFAULT_DROPDB_PATH,
                "createdb_path": self.DEFAULT_CREATEDB_PATH,
                "pg_dump_path": self.DEFAULT_PG_DUMP_PATH,
                "pg_restore_path": self.DEFAULT_PG_RESTORE_PATH,
                "psql_args": self.DEFAULT_PSQL_ARGS,
                "pg_password": self.DEFAULT_PG_PASSWORD,
                "odoo_service_name": self.DEFAULT_ODOO_SERVICE_NAME,
                "odoo_python_path": self.DEFAULT_ODOO_PYTHON_PATH,
                "odoo_admin_login": self.DEFAULT_ODOO_ADMIN_LOGIN,
                "odoo_admin_password": self.DEFAULT_ODOO_ADMIN_PASSWORD,
                "variant_note": self.VARIANT_NOTE,
                "prompt_path": self.prompt_path,
                "launch_script_path": self.launch_script_path,
            }
        )
        return md


def cfg_for_variant(spec: VariantSpec) -> OdooTaskConfig:
    return OdooTaskConfig(
        VARIANT_NAME=spec.variant_name,
        RUN_TAG=spec.run_tag,
        DEFAULT_DB_NAME=spec.db_name,
        VARIANT_NOTE=spec.variant_note,
    )


def build_task(spec: VariantSpec) -> cb.Task:
    cfg = cfg_for_variant(spec)
    return cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
    )


async def _run_cmd(session: cb.DesktopSession, cmd: str, timeout: float = 240.0) -> dict:
    return await session.run_command(cmd, check=False)


async def _run_psql_json(
    session: cb.DesktopSession,
    psql_path: str,
    psql_args: str,
    db_name: str,
    sql_text: str,
    work_dir: str,
    pg_password: str | None,
):
    sql_path = win_join(work_dir, "autograde.sql")
    await session.write_file(sql_path, sql_text)

    if pg_password:
        cmd = (
            'powershell -NoProfile -Command '
            f'"$env:PGPASSWORD=\'{pg_password}\'; '
            f"& '{psql_path}' {psql_args} -d {db_name} -t -A -q -f \\\"{sql_path}\\\"\""
        )
    else:
        cmd = f'"{psql_path}" {psql_args} -d {db_name} -t -A -q -f "{sql_path}"'

    res = await session.run_command(cmd, check=False)
    if res.get("return_code", 1) != 0:
        raise RuntimeError(f"psql failed rc={res.get('return_code')}, stderr={res.get('stderr')}")

    raw = (res.get("stdout") or "").strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("psql returned empty output")
    return json.loads(lines[-1])


async def _reset_db(
    session: cb.DesktopSession,
    *,
    dropdb_path: str,
    createdb_path: str,
    pg_dump_path: str,
    pg_restore_path: str,
    psql_path: str,
    psql_args: str,
    pg_password: str,
    template_db: str,
    target_db: str,
    work_dir: str,
    odoo_service_name: str = "odoo-server-19.0",
):
    # Stop Odoo service so it does not hold connections during DB reset
    await _run_cmd(session, f'powershell -NoProfile -Command "Stop-Service -Name \'{odoo_service_name}\' -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 5"')

    try:
        terminate_sql = (
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname='{target_db}' AND pid<>pg_backend_pid();"
        )
        await _run_psql_json(session, psql_path, psql_args, "postgres", "SELECT json_build_object('ok', true)::text;", work_dir, pg_password)
        await _run_psql_json(
            session,
            psql_path,
            psql_args,
            "postgres",
            f"SELECT json_build_object('terminated', (SELECT COUNT(*) FROM ({terminate_sql}) t))::text;",
            work_dir,
            pg_password,
        )
    except Exception:
        pass

    cmd_drop = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f"& '{dropdb_path}' {psql_args} --if-exists {target_db}\""
    )
    drop_res = await _run_cmd(session, cmd_drop, timeout=240.0)
    if drop_res.get("return_code", 0) != 0:
        logger.warning("dropdb %s failed: %s", target_db, drop_res.get("stderr"))

    cmd_create_template = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f"& '{createdb_path}' {psql_args} -T {template_db} {target_db}\""
    )
    res = await _run_cmd(session, cmd_create_template, timeout=360.0)
    if res.get("return_code", 0) == 0:
        # Restart Odoo service after successful DB reset
        await _run_cmd(session, f'powershell -NoProfile -Command "Start-Service -Name \'{odoo_service_name}\' -ErrorAction SilentlyContinue"')
        return {"method": "createdb -T"}

    dump_path = win_join(work_dir, "template_dump.dump")
    cmd_dump = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f"& '{pg_dump_path}' {psql_args} -Fc -f \\\"{dump_path}\\\" {template_db}\""
    )
    res = await _run_cmd(session, cmd_dump, timeout=900.0)
    if res.get("return_code", 0) != 0:
        raise RuntimeError(f"pg_dump failed: {res.get('stderr')}")

    cmd_create_empty = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f"& '{createdb_path}' {psql_args} {target_db}\""
    )
    res = await _run_cmd(session, cmd_create_empty, timeout=240.0)
    if res.get("return_code", 0) != 0:
        raise RuntimeError(f"createdb failed: {res.get('stderr')}")

    cmd_restore = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f"& '{pg_restore_path}' {psql_args} -d {target_db} \\\"{dump_path}\\\"\""
    )
    res = await _run_cmd(session, cmd_restore, timeout=1200.0)
    if res.get("return_code", 0) != 0:
        await _run_cmd(session, 'powershell -NoProfile -Command "Start-Service -Name OdooServer -ErrorAction SilentlyContinue"')
        raise RuntimeError(f"pg_restore failed: {res.get('stderr')}")

    # Restart Odoo service after successful DB reset
    await _run_cmd(session, 'powershell -NoProfile -Command "Start-Service -Name OdooServer -ErrorAction SilentlyContinue"')
    return {"method": "pg_dump/pg_restore"}


async def _reset_admin_credentials(
    session: cb.DesktopSession,
    *,
    work_dir: str,
    db_name: str,
    pg_password: str,
    odoo_python_path: str,
    admin_login: str,
    admin_password: str,
) -> None:
    script = (
        "import psycopg2\n"
        "from passlib.hash import pbkdf2_sha512\n"
        f"pw_hash = pbkdf2_sha512.hash({admin_password!r})\n"
        f"conn = psycopg2.connect(host='127.0.0.1', port=5432, user='openpg', password={pg_password!r}, dbname={db_name!r})\n"
        "cur = conn.cursor()\n"
        f"cur.execute('UPDATE res_users SET login = %s, password = %s WHERE id = 2', ({admin_login!r}, pw_hash))\n"
        "conn.commit()\n"
        "cur.close()\n"
        "conn.close()\n"
        "print('OK')\n"
    )
    script_path = win_join(work_dir, "reset_admin.py")
    await session.write_file(script_path, script)
    res = await _run_cmd(session, f'"{odoo_python_path}" "{script_path}"', timeout=120.0)
    stdout = res.get("stdout") or ""
    if "OK" not in stdout:
        raise RuntimeError(f"admin credential reset failed: stdout={stdout}, stderr={res.get('stderr')}")


async def _write_report(session: cb.DesktopSession, *, out_dir: str, work_dir: str, report: dict) -> None:
    payload = json.dumps(report, indent=2)
    await session.write_file(win_join(work_dir, "autograde_report.json"), payload)
    if not is_fixture_output_dir(out_dir):
        await session.write_file(win_join(out_dir, "autograde_report.json"), payload)


async def start_variant_task(task_cfg, session: cb.DesktopSession) -> None:
    out_dir = task_cfg.metadata["remote_output_dir"]
    template_db = task_cfg.metadata["template_db"]
    db_name = task_cfg.metadata["db_name"]
    work_dir = eval_work_dir(task_cfg.metadata["variant_name"])

    await session.interface.create_dir(out_dir)
    await session.interface.create_dir(work_dir)

    try:
        info = await _reset_db(
            session,
            dropdb_path=task_cfg.metadata["dropdb_path"],
            createdb_path=task_cfg.metadata["createdb_path"],
            pg_dump_path=task_cfg.metadata["pg_dump_path"],
            pg_restore_path=task_cfg.metadata["pg_restore_path"],
            psql_path=task_cfg.metadata["psql_path"],
            psql_args=task_cfg.metadata["psql_args"],
            pg_password=task_cfg.metadata["pg_password"],
            template_db=template_db,
            target_db=db_name,
            work_dir=work_dir,
            odoo_service_name=task_cfg.metadata.get("odoo_service_name", "odoo-server-19.0"),
        )
        await session.write_file(win_join(work_dir, "RESET_OK.txt"), f"Reset OK. method={info.get('method')}\n")
        try:
            await _reset_admin_credentials(
                session,
                work_dir=work_dir,
                db_name=db_name,
                pg_password=task_cfg.metadata["pg_password"],
                odoo_python_path=task_cfg.metadata["odoo_python_path"],
                admin_login=task_cfg.metadata["odoo_admin_login"],
                admin_password=task_cfg.metadata["odoo_admin_password"],
            )
        except Exception as exc_cred:
            logger.warning("admin credential reset failed (non-fatal): %s", exc_cred)
    except Exception as exc:
        await session.write_file(win_join(work_dir, "RESET_FAILED.txt"), str(exc))

    await session.write_file(
        win_join(work_dir, "README_SETUP.txt"),
        "Setup only prepares the runtime output directory and attempts a DB reset. It does not launch Odoo or clear evaluator fixtures.\n",
    )


def _float_eq(a, b, tol=0.01) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


async def evaluate_variant_task(task_cfg, session: cb.DesktopSession) -> list[float]:
    out_dir = task_cfg.metadata["remote_output_dir"]
    ref_dir = task_cfg.metadata.get("reference_dir")
    db_name = task_cfg.metadata["db_name"]
    run_tag = task_cfg.metadata["run_tag"]
    work_dir = eval_work_dir(task_cfg.metadata["variant_name"])
    await session.interface.create_dir(work_dir)

    report = {
        "task_tag": task_cfg.metadata["variant_name"],
        "run_tag": run_tag,
        "db_name": db_name,
        "checks": {},
        "evidence": {},
        "screenshot_eval": {},
    }

    preflight_sql = r"""
SELECT json_build_object(
  'product_template', to_regclass('public.product_template'),
  'product_product', to_regclass('public.product_product'),
  'sale_order', to_regclass('public.sale_order'),
  'sale_order_line', to_regclass('public.sale_order_line'),
  'purchase_order', to_regclass('public.purchase_order'),
  'purchase_order_line', to_regclass('public.purchase_order_line'),
  'account_move', to_regclass('public.account_move'),
  'stock_quant', to_regclass('public.stock_quant'),
  'stock_lot', to_regclass('public.stock_lot'),
  'stock_scrap', to_regclass('public.stock_scrap'),
  'stock_location', to_regclass('public.stock_location'),
  'stock_warehouse', to_regclass('public.stock_warehouse'),
  'stock_picking', to_regclass('public.stock_picking'),
  'stock_picking_type', to_regclass('public.stock_picking_type'),
  'stock_move_line', to_regclass('public.stock_move_line'),
  'mrp_production', to_regclass('public.mrp_production'),
  'mrp_workorder', to_regclass('public.mrp_workorder'),
  'stock_landed_cost', to_regclass('public.stock_landed_cost'),
  'stock_valuation_adjustment_lines', to_regclass('public.stock_valuation_adjustment_lines'),
  'account_analytic_line', to_regclass('public.account_analytic_line'),
  'account_analytic_account', to_regclass('public.account_analytic_account')
)::text;
"""
    try:
        pre = await _run_psql_json(
            session,
            task_cfg.metadata["psql_path"],
            task_cfg.metadata["psql_args"],
            db_name,
            preflight_sql,
            work_dir,
            task_cfg.metadata["pg_password"],
        )
        report["evidence"]["preflight"] = pre
        needed = [
            "stock_landed_cost",
            "stock_valuation_adjustment_lines",
            "stock_warehouse",
            "stock_quant",
            "account_move",
            "sale_order",
            "purchase_order",
            "mrp_production",
            "stock_picking",
            "stock_move_line",
        ]
        ok = all(pre.get(item) is not None for item in needed)
        report["checks"]["db_preflight_ok"] = ok
        if not ok:
            await _write_report(session, out_dir=out_dir, work_dir=work_dir, report=report)
            return [0.0]
    except Exception as exc:
        report["checks"]["db_preflight_ok"] = False
        report["checks"]["db_preflight_error"] = str(exc)
        await _write_report(session, out_dir=out_dir, work_dir=work_dir, report=report)
        return [0.0]

    schema_sql = r'''
SELECT json_build_object(
  'sml_qty_col',
    CASE
      WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='stock_move_line' AND column_name='qty_done') THEN 'qty_done'
      WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='stock_move_line' AND column_name='quantity') THEN 'quantity'
      WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='stock_move_line' AND column_name='product_uom_qty') THEN 'product_uom_qty'
      ELSE NULL
    END,
  'pol_qty_col',
    CASE
      WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='purchase_order_line' AND column_name='product_qty') THEN 'product_qty'
      WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='purchase_order_line' AND column_name='product_uom_qty') THEN 'product_uom_qty'
      ELSE NULL
    END,
  'scrap_qty_col',
    CASE
      WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='stock_scrap' AND column_name='scrap_qty') THEN 'scrap_qty'
      WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='stock_scrap' AND column_name='product_qty') THEN 'product_qty'
      WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='stock_scrap' AND column_name='product_uom_qty') THEN 'product_uom_qty'
      ELSE NULL
    END
)::text;
'''
    try:
        schema = await _run_psql_json(
            session,
            task_cfg.metadata["psql_path"],
            task_cfg.metadata["psql_args"],
            db_name,
            schema_sql,
            work_dir,
            task_cfg.metadata["pg_password"],
        )
        report["evidence"]["schema"] = schema
    except Exception as exc:
        schema = {}
        report["evidence"]["schema_error"] = str(exc)

    sml_qty_col = schema.get("sml_qty_col") or "qty_done"
    if sml_qty_col not in ("qty_done", "quantity", "product_uom_qty"):
        sml_qty_col = "qty_done"
    pol_qty_col = schema.get("pol_qty_col") or "product_qty"
    if pol_qty_col not in ("product_qty", "product_uom_qty"):
        pol_qty_col = "product_qty"
    scrap_qty_col = schema.get("scrap_qty_col") or "scrap_qty"
    if scrap_qty_col not in ("scrap_qty", "product_qty", "product_uom_qty"):
        scrap_qty_col = "scrap_qty"

    sql = f"""
WITH
prod AS (
  SELECT pp.id AS product_id, pt.default_code
  FROM product_product pp
  JOIN product_template pt ON pt.id = pp.product_tmpl_id
  WHERE pt.default_code IN ('FP-1000','SA-200','SC-300','RM-010','RM-020','RM-030','RM-040','ACC-900')
),
eur AS (SELECT id AS currency_id FROM res_currency WHERE name='EUR' LIMIT 1),
lc AS (
  SELECT slc.id
  FROM stock_landed_cost slc
  JOIN stock_valuation_adjustment_lines sval ON sval.cost_id = slc.id
  JOIN prod p ON p.product_id = sval.product_id
  WHERE p.default_code IN ('RM-010','RM-020')
    AND (COALESCE(slc.name,'') ILIKE '%{run_tag}%' OR COALESCE(slc.description,'') ILIKE '%{run_tag}%')
  GROUP BY slc.id
  HAVING ABS(SUM(sval.additional_landed_cost) - 100.0) < 0.0001
  ORDER BY slc.id DESC
  LIMIT 1
),
lc_split AS (
  SELECT p.default_code, ROUND(SUM(sval.additional_landed_cost)::numeric, 2) AS add_cost
  FROM stock_valuation_adjustment_lines sval
  JOIN lc ON lc.id = sval.cost_id
  JOIN prod p ON p.product_id = sval.product_id
  GROUP BY p.default_code
),
lc_bill_paid AS (
  SELECT COUNT(*) AS cnt
  FROM stock_landed_cost slc
  JOIN lc ON lc.id = slc.id
  JOIN account_move am ON am.id = slc.vendor_bill_id
  WHERE am.move_type='in_invoice' AND am.state='posted' AND am.payment_state='paid'
),
so_candidate AS (
  SELECT so.id, so.name,
         SUM(CASE WHEN pt.default_code='FP-1000' THEN sol.product_uom_qty ELSE 0 END) AS fp_qty,
         SUM(CASE WHEN pt.default_code='ACC-900' THEN sol.product_uom_qty ELSE 0 END) AS acc_qty,
         BOOL_OR(
           COALESCE(so.client_order_ref,'') ILIKE '%{run_tag}%'
           OR COALESCE(so.origin,'') ILIKE '%{run_tag}%'
           OR COALESCE(so.note,'') ILIKE '%{run_tag}%'
           OR COALESCE(so.name,'') ILIKE '%{run_tag}%'
         ) AS has_tag,
         MAX(so.state) AS state
  FROM sale_order so
  JOIN sale_order_line sol ON sol.order_id = so.id
  JOIN product_product pp ON pp.id = sol.product_id
  JOIN product_template pt ON pt.id = pp.product_tmpl_id
  WHERE pt.default_code IN ('FP-1000','ACC-900')
  GROUP BY so.id, so.name
),
so AS (
  SELECT * FROM so_candidate WHERE has_tag ORDER BY id DESC LIMIT 1
),
dropship_done AS (
  SELECT COUNT(*) AS cnt
  FROM stock_move_line sml
  JOIN stock_picking sp ON sp.id = sml.picking_id
  JOIN stock_picking_type spt ON spt.id = sp.picking_type_id
  JOIN product_product pp ON pp.id = sml.product_id
  JOIN product_template pt ON pt.id = pp.product_tmpl_id
  WHERE spt.code='dropship' AND sp.state='done'
    AND pt.default_code='ACC-900'
  GROUP BY sp.id
  HAVING ABS(SUM(sml.{sml_qty_col}) - 3.0) < 0.0001
),
mo_sa_done AS (
  SELECT COUNT(*) AS cnt
  FROM mrp_production mp
  JOIN product_product pp ON pp.id = mp.product_id
  JOIN product_template pt ON pt.id = pp.product_tmpl_id
  WHERE pt.default_code='SA-200' AND mp.state='done' AND ABS(mp.product_qty - 3.0) < 0.0001
),
mo_fp_done AS (
  SELECT COUNT(*) AS cnt
  FROM mrp_production mp
  JOIN product_product pp ON pp.id = mp.product_id
  JOIN product_template pt ON pt.id = pp.product_tmpl_id
  WHERE pt.default_code='FP-1000' AND mp.state='done' AND ABS(mp.product_qty - 3.0) < 0.0001
),
workorders_done AS (
  SELECT COUNT(*) AS cnt
  FROM mrp_workorder wo
  WHERE wo.state='done'
),
paid_invoices AS (
  SELECT ROUND(am.amount_total::numeric, 2) AS amount_total
  FROM account_move am
  JOIN eur ON eur.currency_id = am.currency_id
  WHERE am.move_type = 'out_invoice'
    AND am.state = 'posted'
    AND am.payment_state = 'paid'
    AND (
      COALESCE(am.ref,'') ILIKE '%{run_tag}%'
      OR COALESCE(am.invoice_origin,'') ILIKE '%{run_tag}%'
      OR COALESCE(am.narration,'') ILIKE '%{run_tag}%'
    )
),
paid_credit AS (
  SELECT ROUND(am.amount_total::numeric, 2) AS amount_total
  FROM account_move am
  JOIN eur ON eur.currency_id = am.currency_id
  WHERE am.move_type = 'out_refund'
    AND am.state = 'posted'
    AND am.payment_state = 'paid'
    AND (
      COALESCE(am.ref,'') ILIKE '%{run_tag}%'
      OR COALESCE(am.invoice_origin,'') ILIKE '%{run_tag}%'
      OR COALESCE(am.narration,'') ILIKE '%{run_tag}%'
    )
),
paid_vendor_bills AS (
  SELECT COUNT(*) AS cnt
  FROM account_move am
  WHERE am.move_type='in_invoice'
    AND am.state='posted'
    AND am.payment_state='paid'
    AND (
      COALESCE(am.ref,'') ILIKE '%{run_tag}%'
      OR COALESCE(am.invoice_origin,'') ILIKE '%{run_tag}%'
      OR COALESCE(am.narration,'') ILIKE '%{run_tag}%'
    )
),
fp_lots AS (
  SELECT sl.name
  FROM stock_lot sl
  JOIN prod p ON p.product_id = sl.product_id
  WHERE p.default_code='FP-1000' AND sl.name IN ('SSK-0001','SSK-0002','SSK-0003')
),
sc_lots AS (
  SELECT sl.name
  FROM stock_lot sl
  JOIN prod p ON p.product_id = sl.product_id
  WHERE p.default_code='SC-300' AND sl.name IN ('CS-0001','CS-0002','CS-0003')
),
subcontract_sc_done AS (
  SELECT COUNT(DISTINCT sl.name) AS cnt
  FROM stock_move_line sml
  JOIN stock_picking sp ON sp.id = sml.picking_id
  JOIN stock_lot sl ON sl.id = sml.lot_id
  JOIN product_product pp ON pp.id = sml.product_id
  JOIN product_template pt ON pt.id = pp.product_tmpl_id
  WHERE sp.state='done' AND pt.default_code='SC-300' AND sl.name IN ('CS-0001','CS-0002','CS-0003')
),
wh_base AS (
  SELECT sw.id, sw.code, sw.lot_stock_id
  FROM stock_warehouse sw
  WHERE sw.code IN ('WH','My Co')
),
wh_locs AS (
  SELECT wb.code AS wh_code, loc.id AS location_id
  FROM wh_base wb
  JOIN stock_location loc
    ON loc.id = wb.lot_stock_id
    OR loc.parent_path LIKE CONCAT('%/', wb.lot_stock_id::text, '/%')
),
wh_qty AS (
  SELECT wl.wh_code, p.default_code AS product_code, ROUND(SUM(sq.quantity)::numeric, 6) AS qty
  FROM stock_quant sq
  JOIN wh_locs wl ON wl.location_id = sq.location_id
  JOIN prod p ON p.product_id = sq.product_id
  GROUP BY wl.wh_code, p.default_code
),
ssk2_wh AS (
  SELECT wl.wh_code, ROUND(SUM(sq.quantity)::numeric, 6) AS qty
  FROM stock_quant sq
  JOIN stock_lot sl ON sl.id = sq.lot_id
  JOIN wh_locs wl ON wl.location_id = sq.location_id
  WHERE sl.name='SSK-0002'
  GROUP BY wl.wh_code
),
scrap_rm030 AS (
  SELECT COUNT(*) AS cnt
  FROM stock_scrap ss
  JOIN prod p ON p.product_id = ss.product_id
  WHERE p.default_code='RM-030'
    AND ss.state='done'
    AND ABS(ss.{scrap_qty_col} - 1.0) < 0.0001
),
extra_po_rm030 AS (
  SELECT COUNT(*) AS cnt
  FROM purchase_order po
  JOIN purchase_order_line pol ON pol.order_id = po.id
  JOIN prod p ON p.product_id = pol.product_id
  WHERE p.default_code='RM-030'
    AND ABS(pol.{pol_qty_col} - 1.0) < 0.0001
    AND (
      COALESCE(po.partner_ref,'') ILIKE '%{run_tag}%'
      OR COALESCE(po.origin,'') ILIKE '%{run_tag}%'
      OR COALESCE(po.name,'') ILIKE '%{run_tag}%'
    )
),
  analytic_expense AS (
  SELECT COUNT(*) AS cnt
  FROM account_analytic_line aal
  JOIN account_analytic_account aaa ON aaa.id = aal.account_id
  WHERE aaa.name::text ILIKE '%AN-{run_tag}%'
    AND aal.name::text ILIKE '%Travel to Berlin%'
    AND ABS(aal.amount + 123.45) < 0.01
)
SELECT json_build_object(
  'lc_split', (SELECT COALESCE(json_object_agg(default_code, add_cost), '{{}}'::json) FROM lc_split),
  'lc_bill_paid_cnt', (SELECT cnt FROM lc_bill_paid),
  'so', (SELECT COALESCE(json_build_object('id', id, 'name', name, 'fp_qty', fp_qty, 'acc_qty', acc_qty, 'state', state), '{{}}'::json) FROM so),
  'dropship_done_cnt', (SELECT COALESCE(SUM(cnt), 0) FROM dropship_done),
  'mo_sa_done_cnt', (SELECT cnt FROM mo_sa_done),
  'mo_fp_done_cnt', (SELECT cnt FROM mo_fp_done),
  'workorders_done_cnt', (SELECT cnt FROM workorders_done),
  'paid_invoices', (SELECT COALESCE(json_agg(amount_total ORDER BY amount_total), '[]'::json) FROM paid_invoices),
  'paid_credit', (SELECT COALESCE(json_agg(amount_total ORDER BY amount_total), '[]'::json) FROM paid_credit),
  'paid_vendor_bills_cnt', (SELECT cnt FROM paid_vendor_bills),
  'fp_lots', (SELECT COALESCE(json_agg(name ORDER BY name), '[]'::json) FROM fp_lots),
  'sc_lots', (SELECT COALESCE(json_agg(name ORDER BY name), '[]'::json) FROM sc_lots),
  'subcontract_sc_distinct_lots', (SELECT cnt FROM subcontract_sc_done),
  'wh_qty', (SELECT COALESCE(json_agg(json_build_object('wh', wh_code, 'product', product_code, 'qty', qty) ORDER BY wh_code, product_code), '[]'::json) FROM wh_qty),
  'ssk2_wh', (SELECT COALESCE(json_agg(json_build_object('wh', wh_code, 'qty', qty) ORDER BY wh_code), '[]'::json) FROM ssk2_wh),
  'scrap_rm030_cnt', (SELECT cnt FROM scrap_rm030),
  'extra_po_rm030_cnt', (SELECT cnt FROM extra_po_rm030),
  'analytic_expense_cnt', (SELECT cnt FROM analytic_expense)
)::text;
"""
    try:
        evidence = await _run_psql_json(
            session,
            task_cfg.metadata["psql_path"],
            task_cfg.metadata["psql_args"],
            db_name,
            sql,
            work_dir,
            task_cfg.metadata["pg_password"],
        )
        report["evidence"].update(evidence)
        report["checks"]["db_evidence_collected"] = True
    except Exception as exc:
        report["checks"]["db_evidence_collected"] = False
        report["checks"]["db_evidence_error"] = str(exc)
        await _write_report(session, out_dir=out_dir, work_dir=work_dir, report=report)
        return [0.0]

    def wh_map(wh_qty_rows):
        by_wh = {}
        for row in wh_qty_rows:
            if isinstance(row, dict):
                by_wh.setdefault(row.get("wh"), {})
                by_wh[row.get("wh")][row.get("product")] = float(row.get("qty", 0) or 0)
        return by_wh

    lc_split = report["evidence"].get("lc_split") or {}
    inv = report["evidence"].get("paid_invoices") or []
    credit = report["evidence"].get("paid_credit") or []
    fp_lots = set(report["evidence"].get("fp_lots") or [])
    sc_lots = set(report["evidence"].get("sc_lots") or [])
    so = report["evidence"].get("so") or {}
    by_wh = wh_map(report["evidence"].get("wh_qty") or [])
    ssk2_map = {
        item.get("wh"): float(item.get("qty", 0) or 0)
        for item in (report["evidence"].get("ssk2_wh") or [])
        if isinstance(item, dict)
    }

    def wh_prod_qty(warehouse: str, product: str) -> float:
        return float(by_wh.get(warehouse, {}).get(product, 0) or 0)

    checks = {
        "landed_cost_rm010_60": _float_eq(lc_split.get("RM-010", 0), 60.00),
        "landed_cost_rm020_40": _float_eq(lc_split.get("RM-020", 0), 40.00),
        "landed_cost_vendor_bill_paid": int(report["evidence"].get("lc_bill_paid_cnt", 0) or 0) >= 1,
        "so_confirmed_exists": bool(so) and str(so.get("state", "")).lower() in ("sale", "done"),
        "so_lines_fp3_acc3": bool(so) and _float_eq(so.get("fp_qty", 0), 3.0) and _float_eq(so.get("acc_qty", 0), 3.0),
        "dropship_done_acc900_3": int(report["evidence"].get("dropship_done_cnt", 0) or 0) >= 1,
        "mo_sa200_done_qty3": int(report["evidence"].get("mo_sa_done_cnt", 0) or 0) >= 1,
        "mo_fp1000_done_qty3": int(report["evidence"].get("mo_fp_done_cnt", 0) or 0) >= 1,
        "workorders_done_at_least2": int(report["evidence"].get("workorders_done_cnt", 0) or 0) >= 2,
        "paid_invoice_460": any(_float_eq(item, 460.00) for item in inv),
        "paid_invoice_230": any(_float_eq(item, 230.00) for item in inv),
        "paid_creditnote_200": any(_float_eq(item, 200.00) for item in credit),
        "paid_vendor_bills_at_least6": int(report["evidence"].get("paid_vendor_bills_cnt", 0) or 0) >= 6,
        "fp_serials_all": fp_lots == {"SSK-0001", "SSK-0002", "SSK-0003"},
        "sc_serials_all": sc_lots == {"CS-0001", "CS-0002", "CS-0003"},
        "subcontract_sc300_received_3_serials": int(report["evidence"].get("subcontract_sc_distinct_lots", 0) or 0) >= 3,
        "scrap_rm030_qty1_done": int(report["evidence"].get("scrap_rm030_cnt", 0) or 0) >= 1,
        "extra_po_rm030_qty1_exists": int(report["evidence"].get("extra_po_rm030_cnt", 0) or 0) >= 1,
        "analytic_expense_travel_12345": int(report["evidence"].get("analytic_expense_cnt", 0) or 0) >= 1,
        "WH_RM010_zero": _float_eq(wh_prod_qty("WH", "RM-010"), 0.0),
        "WH_RM020_zero": _float_eq(wh_prod_qty("WH", "RM-020"), 0.0),
        "MyCo_RM010_4": _float_eq(wh_prod_qty("My Co", "RM-010"), 4.0),
        "MyCo_RM020_7": _float_eq(wh_prod_qty("My Co", "RM-020"), 7.0),
        "SSK0002_in_MyCo": float(ssk2_map.get("My Co", 0) or 0) >= 0.99,
        "SSK0002_not_in_WH": _float_eq(float(ssk2_map.get("WH", 0) or 0), 0.0),
    }

    report["score_items"] = [{"name": name, "passed": bool(passed), "weight": 1.0} for name, passed in checks.items()]
    report["checks"].update({name: bool(passed) for name, passed in checks.items()})
    total = len(checks)
    passed = sum(1 for value in checks.values() if value)
    report["final_score"] = round(passed / total, 4) if total > 0 else 0.0
    report["score_detail"] = {"passed": passed, "total": total}

    try:
        if ref_dir:
            output_files, reference_files = await collect_matching_files(session, out_dir, ref_dir)
            question_map = {
                "lc_split.png": "Does the candidate screenshot show Landed Cost allocation with RM-010 additional cost 60.00 and RM-020 additional cost 40.00?",
                "invoices.png": "Does the candidate screenshot show two paid customer invoices with totals 460 EUR and 230 EUR?",
                "return_credit.png": "Does the candidate screenshot show a return for serial SSK-0002 and a posted or paid credit note of 200 EUR?",
                "stock_wh.png": "Does the candidate screenshot show WH has RM-010=0, RM-020=0 and My Co has RM-010=4, RM-020=7?",
            }

            def prompt_with_question(question: str) -> str:
                return f"""You are evaluating an Odoo UI screenshot.

Compare two images:
1. First image: candidate screenshot from the agent run
2. Second image: reference screenshot for the correct result

Question: {question}

Answer with ONLY YES or NO.
"""

            async with EvaluationContext(
                task_tag=task_cfg.metadata["variant_name"],
                mode="custom",
                output_dir=None,
                target_path=out_dir,
                reference_path=ref_dir,
            ) as ctx:
                for name, question in question_map.items():
                    if name in output_files and name in reference_files:
                        target_bytes = await session.read_bytes(os.path.join(out_dir, name))
                        ref_bytes = await session.read_bytes(os.path.join(ref_dir, name))
                        res = await llm_vision_judge(
                            prompt=prompt_with_question(question),
                            image_bytes=target_bytes,
                            reference_image_bytes=ref_bytes,
                            return_details=True,
                            max_tokens=10,
                            eval_context=ctx,
                            identifier=f"{task_cfg.metadata['variant_name']}_{name}_content_check",
                        )
                        report["screenshot_eval"][name] = {
                            "passed": (res.get("score", 0.0) > 0.0),
                            "details": res,
                        }
    except Exception as exc:
        logger.warning("[%s] screenshot grading skipped: %s", task_cfg.metadata["variant_name"], exc)

    await _write_report(session, out_dir=out_dir, work_dir=work_dir, report=report)
    return [report["final_score"]]
