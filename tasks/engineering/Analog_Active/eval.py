"""Evaluation helpers for Buck Converter Design benchmark."""

import re
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _parse_numeric_value(raw_value):
    """Parse LTspice numeric values like '6.65K' or 'DC 4.5'."""

    if not raw_value:
        return None

    value = raw_value.strip().upper().replace("OHM", "")
    value = value.replace("µ", "U").replace("μ", "U")
    if value.startswith("DC "):
        value = value[3:].strip()

    match = re.match(r"^([+-]?\d*\.?\d+(?:E[+-]?\d+)?)\s*([A-Z]+)?$", value)
    if not match:
        return None

    number = float(match.group(1))
    suffix = (match.group(2) or "").strip()
    multipliers = {
        "": 1.0,
        "F": 1.0,
        "R": 1.0,
        "K": 1e3,
        "M": 1e-3,
        "MEG": 1e6,
        "G": 1e9,
        "U": 1e-6,
        "N": 1e-9,
        "P": 1e-12,
    }
    if suffix not in multipliers:
        return None

    return number * multipliers[suffix]


def parse_asc_components(asc_text):
    """Parse LTspice SYMBOL blocks into component dictionaries."""

    components = []
    current = None

    for raw_line in asc_text.split("\n"):
        line = raw_line.strip()
        if line.startswith("SYMBOL"):
            parts = line.split()
            current = {
                "symbol": parts[1].lower() if len(parts) >= 2 else "",
                "x": int(parts[2]) if len(parts) >= 3 else None,
                "y": int(parts[3]) if len(parts) >= 4 else None,
                "inst_name": None,
                "value": None,
            }
            components.append(current)
        elif current and line.startswith("SYMATTR InstName"):
            current["inst_name"] = line.replace("SYMATTR InstName", "", 1).strip()
        elif current and line.startswith("SYMATTR Value"):
            current["value"] = line.replace("SYMATTR Value", "", 1).strip()

    return components


def check_voltage_source_value(asc_text, config):
    """Check that a voltage source matches the task's input voltage."""

    for component in parse_asc_components(asc_text):
        if component["symbol"] != "voltage":
            continue
        value = _parse_numeric_value(component["value"])
        if value is not None and abs(value - config.INPUT_VOLTAGE) <= 1e-6:
            return True
    return False


def check_feedback_resistor_value(asc_text, config):
    """Check that a non-load resistor matches the expected feedback resistor."""

    fb_min = config.FEEDBACK_R_TARGET * (1 - config.FEEDBACK_R_TOL)
    fb_max = config.FEEDBACK_R_TARGET * (1 + config.FEEDBACK_R_TOL)

    for component in parse_asc_components(asc_text):
        if component["symbol"] != "res":
            continue
        if (component["inst_name"] or "").upper() == "RLOAD":
            continue
        value = _parse_numeric_value(component["value"])
        if value is not None and fb_min <= value <= fb_max:
            return True
    return False


def check_capacitor_placement(asc_text):
    """Check that capacitors exist on the input side, output side, and below the IC."""

    components = parse_asc_components(asc_text)
    ic = next((component for component in components if "ltm4648" in component["symbol"]), None)
    if not ic or ic["x"] is None or ic["y"] is None:
        return False

    input_cap = False
    output_cap = False
    soft_start_cap = False

    for component in components:
        if "cap" not in component["symbol"]:
            continue
        x = component["x"]
        y = component["y"]
        if x is None or y is None:
            continue

        if x <= ic["x"] - 100 and y <= ic["y"] - 100:
            input_cap = True
        if x >= ic["x"] + 250 and y <= ic["y"] - 100:
            output_cap = True
        if y >= ic["y"] + 150:
            soft_start_cap = True

    return input_cap and output_cap and soft_start_cap


# ── L1: ASC File Parsing ──

def check_rload_value(asc_text, config):
    """Check that the pre-placed Rload resistor has the correct value (~0.15Ω)."""
    current_instname = None

    for line in asc_text.split('\n'):
        line = line.strip()
        if line.startswith('SYMBOL'):
            current_instname = None
        elif line.startswith('SYMATTR InstName'):
            current_instname = line.replace('SYMATTR InstName', '').strip()
        elif line.startswith('SYMATTR Value') and current_instname == 'Rload':
            val = line.replace('SYMATTR Value', '').strip()
            try:
                parsed = float(val)
            except ValueError:
                return False
            load_min = config.LOAD_R_TARGET * (1 - config.LOAD_R_TOL)
            load_max = config.LOAD_R_TARGET * (1 + config.LOAD_R_TOL)
            return load_min <= parsed <= load_max

    return False


def check_tran_directive(asc_text, config):
    """Check that the ASC file contains a .tran directive with ~1ms duration and startup flag.

    Accepts common LTspice time formats for 1ms: '1m', '1ms', '0.001', '1e-3'.
    The 'startup' keyword must also be present on the directive line.
    """
    for line in asc_text.split('\n'):
        line = line.strip()
        if not line.upper().startswith('TEXT') and not line.upper().startswith('.TRAN'):
            continue

        # LTspice stores directives as: TEXT x y ... ;.tran 1m startup
        # or as raw directives: .tran 1m startup
        tran_match = re.search(r'\.tran\s+(.+)', line, re.IGNORECASE)
        if not tran_match:
            continue

        directive_body = tran_match.group(1).strip()

        # Check for 'startup' keyword
        if 'startup' not in directive_body.lower():
            continue

        # Check for ~1ms duration as first argument
        first_arg = directive_body.split()[0]

        # Parse the time value
        duration_s = None
        try:
            duration_s = float(first_arg)
        except ValueError:
            # Try suffix: "1m", "1ms"
            t_match = re.match(r'^(\d+\.?\d*)\s*(ms?|s)?$', first_arg, re.IGNORECASE)
            if t_match:
                val = float(t_match.group(1))
                suffix = (t_match.group(2) or '').lower()
                if suffix in ('m', 'ms'):
                    duration_s = val * 1e-3
                elif suffix == 's' or suffix == '':
                    duration_s = val

        if duration_s is not None and abs(duration_s - 1e-3) < 1e-4:
            return True

    return False


# ── L2: RAW File Analysis ──

def _parse_raw_binary(raw_path):
    """Parse LTspice .raw file using ltspice lib for header, manual binary read for data."""
    import ltspice as lt

    raw = lt.Ltspice(raw_path)
    # Don't call raw.parse() — it breaks with numpy 2.x
    # Instead read header metadata and parse binary ourselves

    with open(raw_path, 'rb') as f:
        data = f.read()[raw.header_size:]

    n_pts = raw._point_num
    n_vars = raw._variable_num
    var_names = raw._variables

    # Transient sim: time=float64, variables=float32
    y_dtype = np.float32 if 'double' not in raw.flags else np.float64

    record_dtype = np.dtype([('x', np.float64), ('y', y_dtype, (n_vars - 1,))])
    records = np.frombuffer(data, dtype=record_dtype, count=n_pts)

    traces = {var_names[0]: np.abs(records['x'])}
    for i in range(1, n_vars):
        traces[var_names[i]] = records['y'][:, i - 1]

    return traces


def analyze_raw_output(raw_bytes, config):
    """Parse .raw file and check output voltage, ripple, and load current."""
    results = {
        "output_voltage": {"value": None, "pass": False},
        "output_ripple": {"value_mv": None, "pass": False},
        "load_current": {"value": None, "pass": False},
    }

    import tempfile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
            f.write(raw_bytes)
            tmp_path = f.name

        traces = _parse_raw_binary(tmp_path)

        time = traces.get('time')
        vout = traces.get('V(out)')
        if time is None or vout is None:
            logger.info(f"Missing traces. Available: {list(traces.keys())}")
            return results

        # Measurement window 0.7ms–1.0ms
        mask = (time >= config.MEAS_START_MS * 1e-3) & (time <= config.MEAS_END_MS * 1e-3)
        if np.sum(mask) < 10:
            return results

        v_meas = vout[mask].astype(np.float64)

        # Output voltage
        v_mean = float(np.mean(v_meas))
        results["output_voltage"]["value"] = v_mean
        results["output_voltage"]["pass"] = (
            config.OUTPUT_V_TARGET * (1 - config.OUTPUT_V_TOL) <= v_mean
            <= config.OUTPUT_V_TARGET * (1 + config.OUTPUT_V_TOL)
        )

        # Output ripple
        ripple_mv = float((np.max(v_meas) - np.min(v_meas)) * 1000)
        results["output_ripple"]["value_mv"] = ripple_mv
        results["output_ripple"]["pass"] = ripple_mv < config.RIPPLE_MAX_MV

        # Load current
        iload_data = traces.get('I(Rload)')
        if iload_data is None:
            logger.info("I(Rload) not found — Rload may have been removed")
            return results

        iload_mean = float(np.mean(np.abs(iload_data[mask])))

        results["load_current"]["value"] = iload_mean
        results["load_current"]["pass"] = (
            config.LOAD_I_TARGET * (1 - config.LOAD_I_TOL) <= iload_mean
            <= config.LOAD_I_TARGET * (1 + config.LOAD_I_TOL)
        )

    except Exception as e:
        logger.info(f"Failed to analyze .raw file: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return results
