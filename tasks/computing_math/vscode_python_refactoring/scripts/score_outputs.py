"""Score vscode_python_refactoring outputs."""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import io
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_OUTPUTS = (
    "calculator_refactored.py",
    "results.json",
    "pytest_report.txt",
    "radon_report.txt",
)
PUBLIC_FUNCTION = "calculate_order_summary"
TESTS_TOTAL = 13
HELPER_MINIMUM = 4
COMPLEXITY_CEILING = 8
FORBIDDEN_IMPORTS = {"calculator", "input.calculator"}
FORBIDDEN_STRING_FRAGMENTS = (
    "input/calculator.py",
    "input\\calculator.py",
    "input.calculator",
)
DANGEROUS_IMPORT_ROOTS = {
    "os",
    "subprocess",
    "pathlib",
    "importlib",
    "runpy",
    "ctypes",
    "shutil",
    "builtins",
    "io",
    "_io",
}
DANGEROUS_CALL_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "globals",
    "locals",
    "vars",
}


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = float(self.score)
        return payload


def _fail(reason: str, **metrics: Any) -> ScoreResult:
    return ScoreResult(score=0.0, passed=False, reason=reason, metrics=metrics)


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.complexity = 1

    def visit_If(self, node: ast.If) -> Any:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> Any:
        self.complexity += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> Any:
        self.complexity += len(node.cases)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return None


def function_complexity(node: ast.AST) -> int:
    visitor = ComplexityVisitor()
    for child in ast.iter_child_nodes(node):
        visitor.visit(child)
    return visitor.complexity


def function_nodes(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def unique_function_key(node: ast.FunctionDef | ast.AsyncFunctionDef, seen: set[str]) -> str:
    base = node.name
    key = base
    if key in seen:
        key = f"{base}@{node.lineno}"
    seen.add(key)
    return key


def import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.name
            if name in FORBIDDEN_IMPORTS or name.endswith(".calculator"):
                return name
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module in FORBIDDEN_IMPORTS or module.endswith(".calculator"):
            return module
        if module == "input" and any(alias.name == "calculator" for alias in node.names):
            return "input.calculator"
    return None


def dangerous_import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in DANGEROUS_IMPORT_ROOTS:
                return alias.name
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        root = module.split(".", 1)[0]
        if root in DANGEROUS_IMPORT_ROOTS:
            return module
    return None


def static_inspection(module_path: Path) -> dict[str, Any]:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    forbidden_import = None
    forbidden_string = None
    dangerous_import = None
    string_constants: set[str] = set()
    dynamic_code_call = None
    dangerous_call = None
    for node in ast.walk(tree):
        forbidden_import = forbidden_import or import_name(node)
        dangerous_import = dangerous_import or dangerous_import_name(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_constants.add(node.value.replace("\\", "/").lower())
            for fragment in FORBIDDEN_STRING_FRAGMENTS:
                if fragment in node.value:
                    forbidden_string = fragment
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec", "compile"}
        ):
            dynamic_code_call = node.func.id
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALL_NAMES:
                dangerous_call = node.func.id
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in (DANGEROUS_IMPORT_ROOTS | {"__builtins__"})
            ):
                dangerous_call = f"{node.func.value.id}.{node.func.attr}"
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            dangerous_call = dangerous_call or "__builtins__"

    split_forbidden_string = None
    if any(value == "input" or value.endswith("/input") for value in string_constants) and any(
        value == "calculator.py" or value.endswith("/calculator.py") for value in string_constants
    ):
        split_forbidden_string = "input + calculator.py"

    functions = function_nodes(tree)
    function_names = [node.name for node in functions]
    seen_keys: set[str] = set()
    complexities = {
        unique_function_key(node, seen_keys): function_complexity(node)
        for node in functions
    }
    helper_count = len([name for name in function_names if name != PUBLIC_FUNCTION])

    return {
        "forbidden_import": forbidden_import,
        "forbidden_string": forbidden_string,
        "dangerous_import": dangerous_import,
        "dangerous_call": dangerous_call,
        "function_names": function_names,
        "helper_function_count": helper_count,
        "complexities": complexities,
        "max_cyclomatic_complexity": max(complexities.values(), default=0),
        "module_line_count": len(source.splitlines()),
        "dynamic_code_call": dynamic_code_call,
        "imports_input_module": bool(forbidden_import or forbidden_string or split_forbidden_string),
    }


def normalize_resolved_path(value: object) -> str:
    if isinstance(value, int):
        return ""
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    try:
        return str(Path(value).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(value).replace("\\", "/").lower()


def is_forbidden_command(value: object) -> bool:
    text = value if isinstance(value, str) else " ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    normalized = text.replace("\\", "/").lower()
    return "input" in normalized and "calculator.py" in normalized


@contextlib.contextmanager
def deny_input_calculator_access(forbidden_input_dirs: list[Path] | None = None):
    forbidden_input_dirs = [path.resolve() for path in (forbidden_input_dirs or [])]
    forbidden_calculators = [(path / "calculator.py").resolve() for path in forbidden_input_dirs]
    forbidden_input_text = {str(path).replace("\\", "/").lower() for path in forbidden_input_dirs}
    forbidden_calculator_text = {
        str(path).replace("\\", "/").lower() for path in forbidden_calculators
    }
    real_open = builtins.open
    real_io_open = io.open
    real_path_open = Path.open
    real_os_open = os.open
    real_os_read = os.read
    real_os_fdopen = os.fdopen
    real_os_system = os.system
    real_popen = subprocess.Popen
    real_run = subprocess.run
    real_check_call = subprocess.check_call
    real_check_output = subprocess.check_output

    blocked_fds: set[int] = set()
    blocked_dir_fds: set[int] = set()

    def guard_path(path: object) -> None:
        lexical = str(path).replace("\\", "/").lower()
        resolved = normalize_resolved_path(path)
        if (
            lexical in forbidden_calculator_text
            or resolved in forbidden_calculator_text
            or "input/calculator.py" in lexical
            or "input.calculator" in lexical
        ):
            raise RuntimeError("submitted module attempted to read input/calculator.py")

    def is_forbidden_input_dir(path: object) -> bool:
        lexical = str(path).replace("\\", "/").lower()
        resolved = normalize_resolved_path(path)
        return lexical in forbidden_input_text or resolved in forbidden_input_text

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, int) and file in blocked_fds:
            raise RuntimeError("submitted module attempted to read input/calculator.py")
        guard_path(file)
        return real_open(file, *args, **kwargs)

    def guarded_io_open(file, *args, **kwargs):
        if isinstance(file, int) and file in blocked_fds:
            raise RuntimeError("submitted module attempted to read input/calculator.py")
        guard_path(file)
        return real_io_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        guard_path(self)
        return real_path_open(self, *args, **kwargs)

    def guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd in blocked_dir_fds and str(path).replace("\\", "/").lower() == "calculator.py":
            raise RuntimeError("submitted module attempted to read input/calculator.py")
        guard_path(path)
        fd = real_os_open(path, flags, mode, dir_fd=dir_fd)
        if is_forbidden_input_dir(path):
            blocked_dir_fds.add(fd)
        if normalize_resolved_path(path) in forbidden_calculator_text:
            blocked_fds.add(fd)
        return fd

    def guarded_os_read(fd, n):
        if fd in blocked_fds:
            raise RuntimeError("submitted module attempted to read input/calculator.py")
        return real_os_read(fd, n)

    def guarded_os_fdopen(fd, *args, **kwargs):
        if fd in blocked_fds:
            raise RuntimeError("submitted module attempted to read input/calculator.py")
        return real_os_fdopen(fd, *args, **kwargs)

    def guarded_system(command):
        if is_forbidden_command(command):
            raise RuntimeError("submitted module attempted to shell out to input/calculator.py")
        return real_os_system(command)

    class GuardedPopen(real_popen):
        def __init__(self, args, *popen_args, **popen_kwargs):
            if is_forbidden_command(args):
                raise RuntimeError("submitted module attempted to shell out to input/calculator.py")
            super().__init__(args, *popen_args, **popen_kwargs)

    def guarded_run(args, *run_args, **run_kwargs):
        if is_forbidden_command(args):
            raise RuntimeError("submitted module attempted to shell out to input/calculator.py")
        return real_run(args, *run_args, **run_kwargs)

    def guarded_check_call(args, *call_args, **call_kwargs):
        if is_forbidden_command(args):
            raise RuntimeError("submitted module attempted to shell out to input/calculator.py")
        return real_check_call(args, *call_args, **call_kwargs)

    def guarded_check_output(args, *output_args, **output_kwargs):
        if is_forbidden_command(args):
            raise RuntimeError("submitted module attempted to shell out to input/calculator.py")
        return real_check_output(args, *output_args, **output_kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    Path.open = guarded_path_open
    os.open = guarded_os_open
    os.read = guarded_os_read
    os.fdopen = guarded_os_fdopen
    os.system = guarded_system
    subprocess.Popen = GuardedPopen
    subprocess.run = guarded_run
    subprocess.check_call = guarded_check_call
    subprocess.check_output = guarded_check_output
    try:
        yield
    finally:
        builtins.open = real_open
        io.open = real_io_open
        Path.open = real_path_open
        os.open = real_os_open
        os.read = real_os_read
        os.fdopen = real_os_fdopen
        os.system = real_os_system
        subprocess.Popen = real_popen
        subprocess.run = real_run
        subprocess.check_call = real_check_call
        subprocess.check_output = real_check_output


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("calculator_refactored", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("calculator_refactored", None)
    with deny_input_calculator_access():
        spec.loader.exec_module(module)
    return module


def signature_ok(module: Any) -> bool:
    func = getattr(module, PUBLIC_FUNCTION, None)
    if not callable(func):
        return False
    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    return (
        len(params) == 3
        and params[0].name == "order"
        and params[1].name == "customer"
        and params[2].name == "promo_code"
        and params[2].default is None
    )


def run_public_tests(
    agent_module: Path,
    test_file: Path,
    forbidden_input_dir: Path,
) -> tuple[int, int, str, Any]:
    loaded_candidate = None
    with tempfile.TemporaryDirectory(prefix="agenthle_vscode_refactor_") as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        shutil.copy2(test_file, input_dir / "test_calculator_contract.py")
        shutil.copy2(agent_module, output_dir / "calculator_refactored.py")

        spec = importlib.util.spec_from_file_location(
            "test_calculator_contract", input_dir / "test_calculator_contract.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load public tests")
        old_path = list(sys.path)
        old_cwd = Path.cwd()
        try:
            sys.path.insert(0, str(tmp_path))
            module = importlib.util.module_from_spec(spec)
            with deny_input_calculator_access([forbidden_input_dir, input_dir]):
                spec.loader.exec_module(module)
            loaded_candidate = getattr(module, "module", None)
            tests = [
                getattr(module, name)
                for name in sorted(dir(module))
                if name.startswith("test_") and callable(getattr(module, name))
            ]
            passed = 0
            failures: list[str] = []
            for test in tests:
                try:
                    with deny_input_calculator_access([forbidden_input_dir, input_dir]):
                        test()
                    passed += 1
                except Exception as exc:  # noqa: BLE001 - report test failure reason
                    failures.append(f"{test.__name__}: {exc}")
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path
            sys.modules.pop("calculator_refactored", None)
            sys.modules.pop("test_calculator_contract", None)

    total = len(tests)
    if failures:
        report = "\n".join(failures)
    else:
        report = f"{'.' * passed}                                                            [100%]\n{passed} passed"
    return passed, total, report, loaded_candidate


def parse_results_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"results.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("results.json must contain a JSON object")
    return payload


def score_directory(agent_dir: Path, input_dir: Path) -> ScoreResult:
    metrics: dict[str, Any] = {}
    missing = [name for name in REQUIRED_OUTPUTS if not (agent_dir / name).exists()]
    if missing:
        return _fail("missing required output files", missing=missing)

    module_path = agent_dir / "calculator_refactored.py"
    test_file = input_dir / "test_calculator_contract.py"
    if not test_file.exists():
        return _fail("missing staged public test file", test_file=str(test_file))

    try:
        static = static_inspection(module_path)
        metrics.update(static)
    except SyntaxError as exc:
        return _fail(f"calculator_refactored.py has invalid syntax: {exc}", **metrics)

    if static["imports_input_module"]:
        return _fail("submitted module imports or reads the input calculator", **metrics)
    if static["dangerous_import"] or static["dangerous_call"]:
        return _fail("submitted module uses forbidden filesystem/process/dynamic APIs", **metrics)
    if static["dynamic_code_call"]:
        return _fail("submitted module uses dynamic code execution", **metrics)
    if PUBLIC_FUNCTION not in static["function_names"]:
        return _fail("public function is missing", **metrics)
    if static["helper_function_count"] < HELPER_MINIMUM:
        return _fail("not enough helper functions", **metrics)
    if static["max_cyclomatic_complexity"] > COMPLEXITY_CEILING:
        return _fail("cyclomatic complexity exceeds ceiling", **metrics)

    try:
        tests_passed, tests_total, test_report, module = run_public_tests(
            module_path, test_file, input_dir
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"public test execution failed: {exc}", **metrics)
    if not signature_ok(module):
        return _fail("public function signature is incorrect", **metrics)
    metrics["tests_passed"] = tests_passed
    metrics["tests_total"] = tests_total
    if tests_passed != TESTS_TOTAL or tests_total != TESTS_TOTAL:
        return _fail("public contract tests did not all pass", **metrics, test_report=test_report)

    try:
        reported = parse_results_json(agent_dir / "results.json")
    except ValueError as exc:
        return _fail(str(exc), **metrics)

    expected_results = {
        "public_function": PUBLIC_FUNCTION,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "max_cyclomatic_complexity": static["max_cyclomatic_complexity"],
        "helper_function_count": static["helper_function_count"],
        "module_line_count": static["module_line_count"],
        "imports_input_module": False,
    }
    metrics["expected_results"] = expected_results
    metrics["reported_results"] = reported
    for key, expected in expected_results.items():
        if reported.get(key) != expected:
            return _fail(f"results.json mismatch for {key}", **metrics)

    pytest_report = (agent_dir / "pytest_report.txt").read_text(encoding="utf-8", errors="replace")
    radon_report = (agent_dir / "radon_report.txt").read_text(encoding="utf-8", errors="replace")
    if "13 passed" not in pytest_report:
        return _fail("pytest_report.txt does not report 13 passed tests", **metrics)
    if "calculator_refactored.py" not in radon_report:
        return _fail("radon_report.txt does not mention calculator_refactored.py", **metrics)

    return ScoreResult(score=1.0, passed=True, reason="all checks passed", metrics=metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = score_directory(args.agent_dir, args.input_dir)
    print(json.dumps(result.to_dict(), indent=None if args.json_only else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    blocked_fds: set[int] = set()
