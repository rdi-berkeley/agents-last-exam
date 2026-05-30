"""
odoo_hard - End-to-End Verifiable Odoo Supply Chain Task (Supply Chain)

Updates in this version:
- Schema-robust grading: detect stock_move_line qty column (qty_done/quantity/product_uom_qty), purchase_order_line qty column, and stock_scrap qty column.

- Expanded objective grading to cover Sales, Dropship, Manufacturing (MO/WO), Subcontract receipt,
  Landed Cost vendor bill, and overall vendor bills paid.
- Strict warehouse distribution (Option 2) enforced:
    WH:    RM-010=0, RM-020=0
    My Co: RM-010=4, RM-020=7
  And return serial SSK-0002 must be in My Co, not in WH.
- Start() resets DB by recreating a fresh database (odoo_hard) each run.
  Preferred: createdb -T template_db; if that fails (template in use), fallback to pg_dump/pg_restore.

Notes:
- This code prioritizes setup convenience (password in metadata), not security hardening.
"""

import json
import logging
import os
from dataclasses import dataclass

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import collect_matching_files, llm_vision_judge, EvaluationContext

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig(GeneralTaskConfig):
    VARIANT_NAME: str = "base"
    DOMAIN_NAME: str = "business_finance"

    TASK_NAME: str = "odoo_hard"

    RUN_TAG: str = "odoo_hard"

    DEFAULT_ODOO_URL: str = "http://localhost:8069/web"
    TEMPLATE_DB_NAME: str = "AgentService"
    DEFAULT_DB_NAME: str = "odoo_hard"

    DEFAULT_PSQL_PATH: str = "psql"
    DEFAULT_DROPDB_PATH: str = "dropdb"
    DEFAULT_CREATEDB_PATH: str = "createdb"
    DEFAULT_PG_DUMP_PATH: str = "pg_dump"
    DEFAULT_PG_RESTORE_PATH: str = "pg_restore"

    DEFAULT_PSQL_ARGS: str = "-h 127.0.0.1 -p 5432 -U openpg"
    DEFAULT_PG_PASSWORD: str = os.environ.get("ODOO_PG_PASSWORD", "openpgpwd")

    @property
    def task_description(self) -> str:
        out_dir = self.remote_output_dir
        tag = self.RUN_TAG
        db_name = self.DEFAULT_DB_NAME

        return f"""
Goal:
Complete an end-to-end Odoo workflow that spans Sales, Purchase, Inventory, Manufacturing (2-level), Subcontracting,
Landed Costs, Invoicing/Accounting, and Expenses. All documents must be traceable and verifiable.

Inputs / Environment:
- Odoo Web URL will be opened automatically.
- A fresh database named "{db_name}" will be created at task start".
- You must work ONLY inside database "{db_name}".
- Use the run tag "{tag}" in a searchable field (Reference / Notes / Customer Ref / Vendor Ref) for EVERY key document.
- Use the existing warehouses:
  - Inbound warehouse: code "WH"  (stock location tree rooted at "WH/Stock")
  - Production & Shipping warehouse: code "My Co" (rooted at "My Co/Stock")

Required workflow (must be completed in order):
1) Global config (in "{db_name}"):
   - Company currency USD; enable EUR and CNY; lock FX rates:
       1 EUR = 1.20 USD; 1 CNY = 0.125 USD
   - Inventory valuation for stockable items: FIFO + Automated valuation
   - Warehouse routes/steps:
       - "WH": 2-step receipts (Vendors -> Input -> Stock)
       - "My Co": 3-step delivery (Pick -> Pack -> Ship)

2) Master data (exact codes):
   Partners:
     - Customer: CUST-EU01 Berlin Robotics GmbH (EUR)
     - Vendors: V-CN01 Shenzhen Metals (CNY), V-US01 ChipWorld (USD),
               V-US02 SensorCore (USD), V-SUB01 CalibrateLab (USD subcontractor),
               V-DS01 DropShipHub (USD dropship)
   Products:
     - FP-1000 Smart Sensor Kit (Stockable; MTO+Manufacture; Serial)
     - SA-200 Control Board (Stockable; Manufacture)
     - SC-300 Calibrated Sensor (Stockable; Subcontract; Serial)
     - RM-010 Aluminum Case (Buy; 96 CNY)
     - RM-020 PCB Blank (Buy; 64 CNY)
     - RM-030 Microcontroller (Buy; 20 USD)
     - RM-040 Sensor Core (Buy; 7 USD; supplied to subcontractor)
     - ACC-900 Power Adapter (Stockable; Dropship; Sale 30 EUR; Buy 10 USD)
     - FREIGHT-100 International Freight (Service; used as Landed Cost line; 100 USD)
   BoMs + operations:
     - FP-1000 BoM: SA-200×1, SC-300×1, RM-010×2; Assembly 15min @ 40 USD/hour
     - SA-200 BoM: RM-020×1, RM-030×1; SMT 30min @ 60 USD/hour
     - SC-300 Subcontract BoM: RM-040×1 supplied; subcontract fee 25 USD/unit

3) Sales:
   - Create an EUR pricelist: FP-1000=200 EUR, ACC-900=30 EUR
   - Create SO for FP-1000×3 and ACC-900×3; include tag "{tag}" in SO reference/notes; Confirm.
   - Confirm must trigger: FP-1000 via MTO->Manufacture, ACC-900 via Dropship.

4) Purchases:
   Create & confirm these POs (all must include tag "{tag}" in a searchable field):
   - PO-CN-001 (CNY): RM-010×10 @96; RM-020×10 @64
   - PO-US-001 (USD): RM-030×3 @20   (intentionally short; needed for later shortage)
   - PO-US-002 (USD): RM-040×5 @7
   - PO-SUB-001 (USD): SC-300×3 @25 (subcontract fee)
   - PO-DS-001 (USD): ACC-900×3 @10 (Dropship)

5) Inbound receiving + Landed Cost (must include tag "{tag}" in Landed Cost name/description):
   - Receive PO-CN-001 into warehouse "WH" using 2-step receipts.
   - Create one Landed Cost record for the PO-CN receipt:
       Cost line: FREIGHT-100 = 100 USD
       Split method: By Current Cost
   - Validate Landed Cost.
   - Complete WH/Input -> WH/Stock transfer.
   - Transfer RM-010 and RM-020 from WH warehouse to "My Co" warehouse (internal transfer done).

6) Subcontracting:
   - Ensure RM-040 is received and transferred to "My Co" warehouse.
   - Resupply subcontractor with RM-040×3 (done).
   - Receive SC-300×3; assign serials: CS-0001, CS-0002, CS-0003.

7) Manufacturing (2-level) + mandatory shortage recovery:
   - Produce SA-200×3 in "My Co" warehouse using SMT operation.
   - During SA production, scrap RM-030 qty 1 (done) causing shortage.
   - Resolve shortage by creating an extra PO for RM-030×1 (must include tag "{tag}"), receive it and transfer to "My Co",
     then finish SA-200×3.
   - Produce FP-1000×3 using Assembly operation; assign serials: SSK-0001, SSK-0002, SSK-0003.

8) Delivery (from "My Co" using 3-step delivery):
   - Partially deliver FP-1000: ship 2 units first (SSK-0001, SSK-0002), generate backorder for 1.
   - Deliver the backorder 1 unit (SSK-0003).
   - ACC-900 must be delivered via Dropship (vendor -> customer).

9) Invoicing & payments (EUR, invoice on delivered quantities):
   - After first delivery: create/post invoice #1 total 460 EUR; register payment to Paid.
   - After second delivery: create/post invoice #2 total 230 EUR; register payment to Paid.

10) Vendor bills:
   - Create/post vendor bills for all POs + freight; pay/reconcile.

11) Expenses:
   - Create analytic account: AN-odoo_hard
   - Create expense: "Travel to Berlin" 123.45 USD, linked to AN-odoo_hard
   - Submit -> Approve -> Post -> Pay

12) Mandatory return/refund:
- Return serial SSK-0002 back into STOCK of warehouse "My Co".
- Create/post a credit note for 200 EUR (FP only) and mark it paid.

Output (what to submit for full credit):
Save these files into: {out_dir}
1) lc_split.png       - Screenshot showing Landed Cost allocation
2) invoices.png       - Screenshot showing two PAID customer invoices
3) return_credit.png  - Screenshot showing return of SSK-0002 AND paid credit note
4) stock_wh.png       - Screenshot showing warehouse on-hand distribution:
                        WH: RM-010=0 and RM-020=0
                        My Co: RM-010=4 and RM-020=7
5) submission.txt     - A short text file listing the key document numbers you created (SO/POs/MOs/Invoices/Credit Note/Landed Cost)
                        and confirming you used tag "{tag}" everywhere.

Verification:
The task is considered successful if the workflow is completed exactly as specified and the required output files are present and consistent with the final expected business state. The {db_name} database must contain all the expected documents with the run tag "{tag}" in searchable fields, and the screenshots must match the expected states.
"""

    def to_metadata(self) -> dict:
        md = super().to_metadata()
        md.update({
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
        })
        return md


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


async def _run_cmd(session: cb.DesktopSession, cmd: str, timeout: float = 240.0) -> dict:
    return await session.run_command(cmd, timeout=timeout, check=False)


async def _run_psql_json(session: cb.DesktopSession, psql_path: str, psql_args: str, db_name: str, sql_text: str, work_dir: str, pg_password: str | None):
    sql_path = os.path.join(work_dir, "autograde.sql")
    await session.write_file(sql_path, sql_text)

    if pg_password:
        cmd = (
            'powershell -NoProfile -Command '
            f'"$env:PGPASSWORD=\'{pg_password}\'; '
            f'& {psql_path} {psql_args} -d {db_name} -t -A -q -f \\"{sql_path}\\""'
        )
    else:
        cmd = f'{psql_path} {psql_args} -d {db_name} -t -A -q -f "{sql_path}"'

    res = await session.run_command(cmd, timeout=300.0, check=False)
    if res.get("return_code", 1) != 0:
        raise RuntimeError(f"psql failed rc={res.get('return_code')}, stderr={res.get('stderr')}")

    raw = (res.get("stdout") or "").strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("psql returned empty output")
    return json.loads(lines[-1])


async def _reset_db(session: cb.DesktopSession, *, dropdb_path: str, createdb_path: str,
                    pg_dump_path: str, pg_restore_path: str,
                    psql_path: str, psql_args: str, pg_password: str,
                    template_db: str, target_db: str, work_dir: str):
    # Terminate sessions to target_db (best-effort)
    try:
        terminate_sql = f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='{target_db}' AND pid<>pg_backend_pid();"
        await _run_psql_json(session, psql_path, psql_args, "postgres", "SELECT json_build_object('ok', true)::text;", work_dir, pg_password)
        await _run_psql_json(session, psql_path, psql_args, "postgres", f"SELECT json_build_object('terminated', (SELECT COUNT(*) FROM ({terminate_sql}) t))::text;", work_dir, pg_password)
    except Exception as e:
        logger.warning(f"Failed to terminate connections to DB '{target_db}': {e}")

    cmd_drop = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f'& {dropdb_path} {psql_args} --if-exists {target_db}"'
    )
    await _run_cmd(session, cmd_drop, timeout=240.0)

    # Try fast clone
    cmd_create_template = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f'& {createdb_path} {psql_args} -T {template_db} {target_db}"'
    )
    res = await _run_cmd(session, cmd_create_template, timeout=360.0)
    if res.get("return_code", 0) == 0:
        return {"method": "createdb -T"}

    # Fallback: pg_dump/pg_restore
    dump_path = os.path.join(work_dir, "template_dump.dump")
    cmd_dump = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f'& {pg_dump_path} {psql_args} -Fc -f \\"{dump_path}\\" {template_db}"'
    )
    res = await _run_cmd(session, cmd_dump, timeout=900.0)
    if res.get("return_code", 0) != 0:
        raise RuntimeError(f"pg_dump failed: {res.get('stderr')}")

    cmd_create_empty = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f'& {createdb_path} {psql_args} {target_db}"'
    )
    res = await _run_cmd(session, cmd_create_empty, timeout=240.0)
    if res.get("return_code", 0) != 0:
        raise RuntimeError(f"createdb failed: {res.get('stderr')}")

    cmd_restore = (
        'powershell -NoProfile -Command '
        f'"$env:PGPASSWORD=\'{pg_password}\'; '
        f'& {pg_restore_path} {psql_args} -d {target_db} \\"{dump_path}\\""'
    )
    res = await _run_cmd(session, cmd_restore, timeout=1200.0)
    if res.get("return_code", 0) != 0:
        raise RuntimeError(f"pg_restore failed: {res.get('stderr')}")

    return {"method": "pg_dump/pg_restore"}


class _OdooHardSetup(BaseTaskSetup):
    """Per-run Postgres DB reset from template + odoo URL open + README.

    Shape B: DB state mutates across runs; Stage 1 cannot keep the DB
    clean and the agent has no Postgres admin access.
    """

    async def setup(self, task_cfg, session: cb.DesktopSession) -> None:
        out_dir = task_cfg.metadata["remote_output_dir"]
        odoo_url = task_cfg.metadata.get("odoo_url", config.DEFAULT_ODOO_URL)
        template_db = task_cfg.metadata.get("template_db", config.TEMPLATE_DB_NAME)
        db_name = task_cfg.metadata.get("db_name", config.DEFAULT_DB_NAME)

        psql_path = task_cfg.metadata.get("psql_path", config.DEFAULT_PSQL_PATH)
        dropdb_path = task_cfg.metadata.get("dropdb_path", config.DEFAULT_DROPDB_PATH)
        createdb_path = task_cfg.metadata.get("createdb_path", config.DEFAULT_CREATEDB_PATH)
        pg_dump_path = task_cfg.metadata.get("pg_dump_path", config.DEFAULT_PG_DUMP_PATH)
        pg_restore_path = task_cfg.metadata.get("pg_restore_path", config.DEFAULT_PG_RESTORE_PATH)
        psql_args = task_cfg.metadata.get("psql_args", config.DEFAULT_PSQL_ARGS)
        pg_password = task_cfg.metadata.get("pg_password", config.DEFAULT_PG_PASSWORD)

        try:
            await session.remove_file(out_dir)
        except Exception:
            logger.warning(f"Failed to remove existing output directory '{out_dir}', it may not exist or be a file. Attempting to continue.")
        await session.makedirs(out_dir)

        try:
            info = await _reset_db(
                session,
                dropdb_path=dropdb_path,
                createdb_path=createdb_path,
                pg_dump_path=pg_dump_path,
                pg_restore_path=pg_restore_path,
                psql_path=psql_path,
                psql_args=psql_args,
                pg_password=pg_password,
                template_db=template_db,
                target_db=db_name,
                work_dir=out_dir,
            )
            await session.write_file(os.path.join(out_dir, "RESET_OK.txt"), f"Reset OK. method={info.get('method')}\n")
        except Exception as e:
            await session.write_file(os.path.join(out_dir, "RESET_FAILED.txt"), str(e))

        db_url = f"{odoo_url}?db={db_name}"
        try:
            await session.run_file(db_url)
        except Exception as e:
            logger.warning(f"[odoo_hard] failed to open odoo url: {e}")

        try:
            await session.write_file(
                os.path.join(out_dir, "README_AUTOGRADE.txt"),
                "Submit: lc_split.png, invoices.png, return_credit.png, stock_wh.png, submission.txt\n"
            )
        except Exception as e:
            logger.warning(f"Failed to write README_AUTOGRADE.txt: {e}")


_setup = _OdooHardSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _float_eq(a, b, tol=0.01) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    out_dir = task_cfg.metadata["remote_output_dir"]
    ref_dir = task_cfg.metadata.get("reference_dir")

    psql_path = task_cfg.metadata.get("psql_path", config.DEFAULT_PSQL_PATH)
    psql_args = task_cfg.metadata.get("psql_args", config.DEFAULT_PSQL_ARGS)
    db_name = task_cfg.metadata.get("db_name", config.DEFAULT_DB_NAME)
    pg_password = task_cfg.metadata.get("pg_password", config.DEFAULT_PG_PASSWORD)

    tag = config.RUN_TAG

    report = {
        "variant_name": config.VARIANT_NAME,
        "run_tag": tag,
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
        pre = await _run_psql_json(session, psql_path, psql_args, db_name, preflight_sql, out_dir, pg_password)
        report["evidence"]["preflight"] = pre
        needed = [
            "stock_landed_cost", "stock_valuation_adjustment_lines",
            "stock_warehouse", "stock_quant", "account_move",
            "sale_order", "purchase_order", "mrp_production",
            "stock_picking", "stock_move_line"
        ]
        ok = all(pre.get(x) is not None for x in needed)
        report["checks"]["db_preflight_ok"] = ok
        if not ok:
            await session.write_file(os.path.join(out_dir, "autograde_report.json"), json.dumps(report, indent=2))
            return [0.0]
    except Exception as e:
        report["checks"]["db_preflight_ok"] = False
        report["checks"]["db_preflight_error"] = str(e)
        await session.write_file(os.path.join(out_dir, "autograde_report.json"), json.dumps(report, indent=2))
        return [0.0]

    # Detect schema variations (column names differ across Odoo versions/customizations)
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
        schema = await _run_psql_json(session, psql_path, psql_args, db_name, schema_sql, out_dir, pg_password)
        report["evidence"]["schema"] = schema
    except Exception as e:
        schema = {}
        report["evidence"]["schema_error"] = str(e)

    sml_qty_col = schema.get("sml_qty_col") or "qty_done"
    if sml_qty_col not in ("qty_done", "quantity", "product_uom_qty"):
        sml_qty_col = "qty_done"

    pol_qty_col = schema.get("pol_qty_col") or "product_qty"
    if pol_qty_col not in ("product_qty", "product_uom_qty"):
        pol_qty_col = "product_qty"

    scrap_qty_col = schema.get("scrap_qty_col") or "scrap_qty"
    if scrap_qty_col not in ("scrap_qty", "product_qty", "product_uom_qty"):
        scrap_qty_col = "scrap_qty"

    logger.info(f"[schema] stock_move_line quantity column: {sml_qty_col}")
    logger.info(f"[schema] purchase_order_line qty column: {pol_qty_col}")
    logger.info(f"[schema] stock_scrap qty column: {scrap_qty_col}")


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
    AND (COALESCE(slc.name,'') ILIKE '%{tag}%' OR COALESCE(slc.description,'') ILIKE '%{tag}%')
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
           COALESCE(so.client_order_ref,'') ILIKE '%{tag}%'
           OR COALESCE(so.origin,'') ILIKE '%{tag}%'
           OR COALESCE(so.note,'') ILIKE '%{tag}%'
           OR COALESCE(so.name,'') ILIKE '%{tag}%'
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
  SELECT *
  FROM so_candidate
  WHERE has_tag
  ORDER BY id DESC
  LIMIT 1
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
      COALESCE(am.ref,'') ILIKE '%{tag}%'
      OR COALESCE(am.invoice_origin,'') ILIKE '%{tag}%'
      OR COALESCE(am.narration,'') ILIKE '%{tag}%'
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
      COALESCE(am.ref,'') ILIKE '%{tag}%'
      OR COALESCE(am.invoice_origin,'') ILIKE '%{tag}%'
      OR COALESCE(am.narration,'') ILIKE '%{tag}%'
    )
),

paid_vendor_bills AS (
  SELECT COUNT(*) AS cnt
  FROM account_move am
  WHERE am.move_type='in_invoice'
    AND am.state='posted'
    AND am.payment_state='paid'
    AND (
      COALESCE(am.ref,'') ILIKE '%{tag}%'
      OR COALESCE(am.invoice_origin,'') ILIKE '%{tag}%'
      OR COALESCE(am.narration,'') ILIKE '%{tag}%'
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
  SELECT
    wl.wh_code,
    p.default_code AS product_code,
    ROUND(SUM(sq.quantity)::numeric, 6) AS qty
  FROM stock_quant sq
  JOIN wh_locs wl ON wl.location_id = sq.location_id
  JOIN prod p ON p.product_id = sq.product_id
  GROUP BY wl.wh_code, p.default_code
),
ssk2_wh AS (
  SELECT
    wl.wh_code,
    ROUND(SUM(sq.quantity)::numeric, 6) AS qty
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
      COALESCE(po.partner_ref,'') ILIKE '%{tag}%'
      OR COALESCE(po.origin,'') ILIKE '%{tag}%'
      OR COALESCE(po.name,'') ILIKE '%{tag}%'
    )
),

analytic_expense AS (
  SELECT COUNT(*) AS cnt
  FROM account_analytic_line aal
  JOIN account_analytic_account aaa ON aaa.id = aal.account_id
  WHERE aaa.name::text ILIKE '%AN-odoo_hard%'
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
        evidence = await _run_psql_json(session, psql_path, psql_args, db_name, sql, out_dir, pg_password)
        report["evidence"].update(evidence)
        report["checks"]["db_evidence_collected"] = True
    except Exception as e:
        report["checks"]["db_evidence_collected"] = False
        report["checks"]["db_evidence_error"] = str(e)
        await session.write_file(os.path.join(out_dir, "autograde_report.json"), json.dumps(report, indent=2))
        return [0.0]

    def wh_map(wh_qty_rows):
        by_wh = {}
        for r in wh_qty_rows:
            if isinstance(r, dict):
                by_wh.setdefault(r.get("wh"), {})
                by_wh[r.get("wh")][r.get("product")] = float(r.get("qty", 0) or 0)
        return by_wh

    lc_split = report["evidence"].get("lc_split") or {}
    inv = report["evidence"].get("paid_invoices") or []
    credit = report["evidence"].get("paid_credit") or []
    fp_lots = set(report["evidence"].get("fp_lots") or [])
    sc_lots = set(report["evidence"].get("sc_lots") or [])
    so = report["evidence"].get("so") or {}
    by_wh = wh_map(report["evidence"].get("wh_qty") or [])
    ssk2_map = {x.get("wh"): float(x.get("qty", 0) or 0) for x in (report["evidence"].get("ssk2_wh") or []) if isinstance(x, dict)}

    def wh_prod_qty(wh, prod):
        return float(by_wh.get(wh, {}).get(prod, 0) or 0)

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

        "paid_invoice_460": any(_float_eq(x, 460.00) for x in inv),
        "paid_invoice_230": any(_float_eq(x, 230.00) for x in inv),
        "paid_creditnote_200": any(_float_eq(x, 200.00) for x in credit),

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

    # Detailed score items (each check is one objective question; missing => 0 for that item)
    score_items = []
    for name, passed in checks.items():
        score_items.append({"name": name, "passed": bool(passed), "weight": 1.0})
        logger.info(f"[odoo_hard][score_item] {name} => {'PASS' if passed else 'FAIL'}")
    report["score_items"] = score_items

    report["checks"].update({k: bool(v) for k, v in checks.items()})

    n = len(checks)
    passed = sum(1 for v in checks.values() if v)
    report["final_score"] = 0.0 if n == 0 else max(0.0, min(1.0, passed / n))
    logger.info(f"[odoo_hard][score_summary] passed={passed}/{n} final_score={report['final_score']:.4f}")
    report["score_detail"] = {"passed": passed, "total": n}

    # Optional screenshot VLM (diagnostic only)
    try:
        if ref_dir:
            output_files, reference_files = await collect_matching_files(session, out_dir, ref_dir)
            question_map = {
                "lc_split.png": "Does the candidate screenshot show Landed Cost allocation with RM-010 additional cost 60.00 and RM-020 additional cost 40.00?",
                "invoices.png": "Does the candidate screenshot show two paid customer invoices with totals 460 EUR and 230 EUR?",
                "return_credit.png": "Does the candidate screenshot show a return for serial SSK-0002 AND a posted/paid credit note of 200 EUR?",
                "stock_wh.png": "Does the candidate screenshot show WH has RM-010=0, RM-020=0 AND My Co has RM-010=4, RM-020=7?",
            }

            def prompt_with_question(question: str) -> str:
                return f"""You are evaluating an Odoo UI screenshot.

Compare two images:
1) First image: candidate screenshot from the agent's run
2) Second image: reference screenshot for the correct result

Question: {question}

Answer with ONLY "YES" or "NO".
"""

            async with EvaluationContext(
                task_tag=config.VARIANT_NAME,
                mode="custom",
                output_dir=None,
                target_path=out_dir,
                reference_path=ref_dir
            ) as ctx:
                for fname, q in question_map.items():
                    if fname in output_files and fname in reference_files:
                        target_bytes = await session.read_bytes(os.path.join(out_dir, fname))
                        ref_bytes = await session.read_bytes(os.path.join(ref_dir, fname))
                        res = await llm_vision_judge(
                            prompt=prompt_with_question(q),
                            image_bytes=target_bytes,
                            reference_image_bytes=ref_bytes,
                            return_details=True,
                            max_tokens=10,
                            eval_context=ctx,
                            identifier=f"{fname}_content_check"
                        )
                        report["screenshot_eval"][fname] = {"passed": (res.get("score", 0.0) > 0.0), "details": res}
    except Exception as e:
        logger.warning(f"[odoo_hard] screenshot VLM grading skipped/failed: {e}")

    try:
        await session.write_file(os.path.join(out_dir, "autograde_report.json"), json.dumps(report, indent=2))
    except Exception as e:
        logger.warning(f"Failed to write autograde_report.json: {e}")

    return [report["final_score"]]
