"""Active Analog Circuit Design Task - Hardware Design Benchmark."""

import ntpath
import logging
from dataclasses import dataclass

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.engineering.Analog_Active.eval import (
    check_capacitor_placement,
    check_feedback_resistor_value,
    check_rload_value,
    check_tran_directive,
    check_voltage_source_value,
    analyze_raw_output,
)

logger = logging.getLogger(__name__)


# ── Variant Definitions ──

VARIANT_SPECS = {
    "Analog_Active_v1": {
        "input_voltage": 5.0,
        "output_v_target": 1.5,
        "load_i_target": 10.0,
        "feedback_r_target": 6650.0,
        "load_r_target": 0.15,
    },
    "Analog_Active_v2": {
        "input_voltage": 3.3,
        "output_v_target": 1.0,
        "load_i_target": 8.0,
        "feedback_r_target": 15000.0,
        "load_r_target": 0.125,
    },
    "Analog_Active_v3": {
        "input_voltage": 5.0,
        "output_v_target": 2.5,
        "load_i_target": 6.0,
        "feedback_r_target": 3090.0,
        "load_r_target": 0.4167,
    },
    "Analog_Active_v4": {
        "input_voltage": 3.3,
        "output_v_target": 1.2,
        "load_i_target": 10.0,
        "feedback_r_target": 10000.0,
        "load_r_target": 0.12,
    },
    "Analog_Active_v5": {
        "input_voltage": 5.0,
        "output_v_target": 3.3,
        "load_i_target": 8.0,
        "feedback_r_target": 2210.0,
        "load_r_target": 0.4125,
    },
}

VARIANTS = list(VARIANT_SPECS.keys())


@dataclass
class TaskConfig(GeneralTaskConfig):
    VARIANT_NAME: str = "Analog_Active_v1"
    DOMAIN_NAME: str = "engineering"

    TASK_NAME: str = "Analog_Active"
    OS_TYPE: str = "windows"

    # Design spec (set by __post_init__ from VARIANT_SPECS)
    INPUT_VOLTAGE: float = 0.0
    OUTPUT_V_TARGET: float = 0.0
    LOAD_I_TARGET: float = 0.0
    FEEDBACK_R_TARGET: float = 0.0
    LOAD_R_TARGET: float = 0.0

    # Constants across all variants
    OUTPUT_V_TOL: float = 0.05
    RIPPLE_MAX_MV: float = 30.0
    LOAD_I_TOL: float = 0.10
    FEEDBACK_R_TOL: float = 0.1
    LOAD_R_TOL: float = 0.01

    # Measurement window
    MEAS_START_MS: float = 0.7
    MEAS_END_MS: float = 1.0

    # Output filenames
    CIRCUIT_FILE: str = "circuit.asc"
    RAW_FILE: str = "circuit.raw"
    SCHEMATIC_SCREENSHOT: str = "schematic_screenshot.png"

    def __post_init__(self):
        if self.VARIANT_NAME in VARIANT_SPECS:
            specs = VARIANT_SPECS[self.VARIANT_NAME]
            self.INPUT_VOLTAGE = specs["input_voltage"]
            self.OUTPUT_V_TARGET = specs["output_v_target"]
            self.LOAD_I_TARGET = specs["load_i_target"]
            self.FEEDBACK_R_TARGET = specs["feedback_r_target"]
            self.LOAD_R_TARGET = specs["load_r_target"]

    @property
    def task_dir(self):
        return f"{self.REMOTE_ROOT_DIR}\\{self.DOMAIN_NAME}\\{self.TASK_NAME}\\{self.VARIANT_NAME}"

    @property
    def task_description(self):
        return f"""You are given an LTspice schematic with the LTM4648 µModule DC/DC buck regulator IC already \
        placed on the canvas. Your task is to complete the circuit design by adding all required external components, \
        wiring them to the correct IC pins, and running a transient simulation to verify the output.

        File Structure:
        {self.task_dir}\\
        ├── input\\
        │   ├── circuit.asc                    # Starter LTspice schematic with LTM4648 IC placed
        │   ├── design_spec.txt                # Design specification
        │   └── ltm4648_datasheet.pdf          # Datasheet for LTM4648 IC
        └── output\\                            # Save your completed schematic here

        Before you begin:
        1. Copy `input\\circuit.asc` to `output\\circuit.asc` — this is your working copy.
        2. Open `output\\circuit.asc` in LTspice. In PowerShell:
           `Start-Process '{self.task_dir}\\output\\circuit.asc'`
        3. The output node is pre-labeled "out".

        Design Specification:
        - IC: LTM4648 µModule DC/DC Buck Regulator
        - Input Voltage: {self.INPUT_VOLTAGE}V
        - Output Voltage: {self.OUTPUT_V_TARGET}V
        - Output Current: {self.LOAD_I_TARGET}A
        - Output Ripple: < {self.RIPPLE_MAX_MV}mV peak-to-peak
        - Startup Settling Time: Output must reach {self.OUTPUT_V_TARGET}V within ~0.6ms
        - Simulation: Transient analysis, 1ms duration, with startup.

        Component Value Formats (use these exact formats in LTspice):
        - Voltage source: Set value to "{self.INPUT_VOLTAGE}" (plain number, no suffix)
        - Resistors: Use plain numeric values (e.g., "5700" not "5.7k", "0.19" not "190m")

        Once the circuit is complete:
        1. Save the schematic (Ctrl+S) to ensure circuit.asc is updated
        2. Run the transient simulation to verify output behavior
        3. Save a screenshot of the completed schematic using save_milestone_screenshot(path="{self.task_dir}\\output\\{self.SCHEMATIC_SCREENSHOT}")

        Do not ask for confirmation. Execute each step directly.
        """

    def to_metadata(self):
        metadata = super().to_metadata()
        metadata.update({
            "input_voltage": self.INPUT_VOLTAGE,
            "output_v_target": self.OUTPUT_V_TARGET,
            "output_v_tol": self.OUTPUT_V_TOL,
            "ripple_max_mv": self.RIPPLE_MAX_MV,
            "load_i_target": self.LOAD_I_TARGET,
            "load_i_tol": self.LOAD_I_TOL,
            "feedback_r_target": self.FEEDBACK_R_TARGET,
            "feedback_r_tol": self.FEEDBACK_R_TOL,
            "load_r_target": self.LOAD_R_TARGET,
            "load_r_tol": self.LOAD_R_TOL,
            "meas_start_ms": self.MEAS_START_MS,
            "meas_end_ms": self.MEAS_END_MS,
        })
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=TaskConfig(VARIANT_NAME=tag).task_description,
            metadata=TaskConfig(VARIANT_NAME=tag).to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {
                    "os_type": "windows",
                }
            },
        )
        for tag in VARIANTS
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


# ── Evaluation ──

@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score the task: L1 component checks + VLM, L2 simulation output."""

    meta = task_cfg.metadata
    tag = meta["variant_name"]
    config = TaskConfig(VARIANT_NAME=tag)
    output_dir = meta["remote_output_dir"]
    score = 0.0

    # ══════════════════════════════════════════════
    # L1: Schematic Component Checks (weight: 0.40)
    # ══════════════════════════════════════════════

    # Read ASC file for deterministic checks (Rload value, .tran directive)
    asc_bytes = None
    try:
        asc_path = ntpath.join(output_dir, config.CIRCUIT_FILE)
        asc_bytes = await session.read_bytes(asc_path)
    except Exception:
        pass

    asc_text = asc_bytes.decode('latin-1') if asc_bytes else None

    # ── Checkpoint 0: Voltage Source Value (0.10) ──
    if asc_text and check_voltage_source_value(asc_text, config):
        score += 0.10
        logger.info(f"Checkpoint 0 PASSED: Voltage source {config.INPUT_VOLTAGE}V")
    else:
        logger.info(f"Checkpoint 0 FAILED: Voltage source {config.INPUT_VOLTAGE}V")

    # ── Checkpoint 1: Feedback Resistor Value (0.10) ──
    fb_min_k = config.FEEDBACK_R_TARGET * (1 - config.FEEDBACK_R_TOL) / 1000
    fb_max_k = config.FEEDBACK_R_TARGET * (1 + config.FEEDBACK_R_TOL) / 1000
    if asc_text and check_feedback_resistor_value(asc_text, config):
        score += 0.10
        logger.info(
            "Checkpoint 1 PASSED: "
            f"Feedback resistor between {fb_min_k:.1f}kΩ and {fb_max_k:.1f}kΩ"
        )
    else:
        logger.info(
            "Checkpoint 1 FAILED: "
            f"Feedback resistor between {fb_min_k:.1f}kΩ and {fb_max_k:.1f}kΩ"
        )

    # ── Checkpoint 2: Load Resistor via ASC parsing (0.10) ──
    if asc_text:
        if check_rload_value(asc_text, config):
            score += 0.10
            logger.info("Checkpoint 2 PASSED: Rload value correct")
        else:
            logger.info("Checkpoint 2 FAILED: Rload value incorrect")
    else:
        logger.info("Checkpoint 2 FAILED: circuit.asc not found")

    # ── Checkpoint 3: Capacitor Placement (0.10) ──
    if asc_text and check_capacitor_placement(asc_text):
        score += 0.10
        logger.info("Checkpoint 3 PASSED: Capacitors found on input, output, and Track/SS regions")
    else:
        logger.info("Checkpoint 3 FAILED: Required capacitor placement not detected")

    # ══════════════════════════════════════════════
    # L2: Simulation Output (weight: 0.60)
    # ══════════════════════════════════════════════

    # ── Checkpoint 4: Transient Directive + Simulation Ran (0.10) ──
    tran_ok = False
    if asc_text:
        tran_ok = check_tran_directive(asc_text, config)

    raw_bytes = None
    try:
        raw_path = ntpath.join(output_dir, config.RAW_FILE)
        raw_bytes = await session.read_bytes(raw_path)
        if not (raw_bytes and len(raw_bytes) > 100):
            raw_bytes = None
    except Exception:
        raw_bytes = None

    if tran_ok and raw_bytes:
        score += 0.10
        logger.info(f"Checkpoint 4 PASSED: .tran directive found and circuit.raw exists ({len(raw_bytes)} bytes)")
    else:
        logger.info(f"Checkpoint 4 FAILED: .tran directive={'found' if tran_ok else 'missing'}, "
                     f"circuit.raw={'found' if raw_bytes else 'missing'}")

    if not raw_bytes:
        logger.info(f"Final score: {score:.2f}")
        return [score]

    # Analyze the .raw file for checkpoints 5-7
    raw_results = analyze_raw_output(raw_bytes, config)

    # ── Checkpoint 5: Output Voltage (0.20) ──
    v = raw_results["output_voltage"]["value"]
    voltage_ok = v is not None and raw_results["output_voltage"]["pass"]

    if voltage_ok:
        score += 0.20
        logger.info(f"Checkpoint 5 PASSED: Output voltage {v:.4f}V")

        # ── Checkpoint 6: Output Ripple (0.15) ──
        r = raw_results["output_ripple"]["value_mv"]
        if r is not None and raw_results["output_ripple"]["pass"]:
            score += 0.15
            logger.info(f"Checkpoint 6 PASSED: Output ripple {r:.2f}mV pk-pk")
        else:
            logger.info(f"Checkpoint 6 FAILED: Output ripple {r}mV pk-pk")

        # ── Checkpoint 7: Load Current (0.15) ──
        i = raw_results["load_current"]["value"]
        if i is not None and raw_results["load_current"]["pass"]:
            score += 0.15
            logger.info(f"Checkpoint 7 PASSED: Load current {i:.2f}A")
        else:
            logger.info(f"Checkpoint 7 FAILED: Load current {i}A")
    else:
        logger.info(f"Checkpoint 5 FAILED: Output voltage {v}V "
                     "(checkpoints 6-7 skipped — require correct output voltage)")

    logger.info(f"Final score: {score:.2f}")
    return [score]
