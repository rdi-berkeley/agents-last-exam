import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def review():
    script = (
        Path(__file__).resolve().parents[3]
        / "release-staging/ale-task-fixes-20260913/post-validation-20260918"
        / "docker-audit-review/review_layers.py"
    )
    if not script.exists():
        pytest.skip("release-scoped Docker layer review is not staged")
    spec = importlib.util.spec_from_file_location("review_docker_layers_sidecar", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory_row(**changes):
    return {
        "ordinal": 23,
        "value": "Bearer",
        "sources": ["ANTIGRAVITY_OAUTH_TOKEN_PATH"],
        "fields": ["ANTIGRAVITY_OAUTH_TOKEN_PATH.token.token_type"],
        "classification": "sensitive_literal",
    } | changes


def test_observed_nested_protocol_metadata_false_positive(review):
    row = inventory_row()
    assert review.adapted.inventory_classification(row) == "sensitive_literal"
    corrected = review.classify_inventory(
        [row], {"ANTIGRAVITY_OAUTH_TOKEN_PATH": {"token": {"token_type": "Bearer"}}}
    )
    assert corrected[0]["classification"] == review.PUBLIC
    assert row["classification"] == "sensitive_literal"


@pytest.mark.parametrize(
    "document",
    [
        {"token": {"token_type": "Bearer", "access_token": "Bearer"}},
        {"token": {"token_type": "Bearer"}, "credentials": ["Bearer"]},
        {"token": {"token_type": "Bearer"}, "tokens": [{"token_type": "Bearer"}]},
        {"credentials": {"token": {"token_type": "Bearer"}}},
    ],
)
def test_metadata_literal_collisions_stay_sensitive(review, document):
    rows = review.classify_inventory([inventory_row()], {"ANTIGRAVITY_OAUTH_TOKEN_PATH": document})
    assert rows[0]["classification"] == "sensitive_literal"


def test_metadata_value_in_scalar_source_stays_sensitive(review):
    row = inventory_row(sources=["ANTIGRAVITY_OAUTH_TOKEN_PATH", "secret/.env"])
    documents = {"ANTIGRAVITY_OAUTH_TOKEN_PATH": {"token": {"token_type": "Bearer"}}}
    assert review.classify_inventory([row], documents)[0]["classification"] == "sensitive_literal"
    row = inventory_row(value="synthetic-secret-not-the-protocol")
    assert review.classify_inventory([row], documents)[0]["classification"] == "sensitive_literal"


def manifest_row(path, content=b"synthetic package payload"):
    return {
        "type": "file",
        "path": path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_task_shared_library_is_not_task_answer(review):
    row = manifest_row("domain/task/base/software/python_pkgs/scipy/linalg/example.py")
    destination = "/opt/runtime/lib/python3.11/site-packages/scipy/linalg/example.py"
    assert review.classify_hash(destination, [row], {}, {}) == "task_shared_runtime_library"


@pytest.mark.parametrize(
    "destination",
    [
        "/home/user/.local/lib/python3.10/site-packages/certifi/cacert.pem",
        "/home/user/.hermes/hermes-agent/venv/lib/python3.11/site-packages/certifi/cacert.pem",
        "/opt/runtime/ssl/cacert.pem",
        "/opt/runtime/pkgs/certifi/info/licenses/LICENSE",
        "/home/user/.local/share/micromamba/pkgs/certifi/ssl/cacert.pem",
        "/home/user/.local/share/uv/python/cpython-3.11/lib/python3.11/test.py",
        "/opt/runtime/etc/conda/test-files/certifi/LICENSE",
        "/opt/runtime/python_standard_lib/test.py",
        "/opt/runtime/node_modules/package/LICENSE",
    ],
)
def test_shared_runtime_roles_include_reference_evaluator_environments(review, destination):
    row = manifest_row(
        "domain/task/base/reference/evaluator_env/.venv/Lib/site-packages/certifi/cacert.pem"
    )
    assert review.classify_hash(destination, [row], {}, {}) == "task_shared_runtime_library"


@pytest.mark.parametrize(
    "source,destination",
    [
        ("domain/task/base/reference/answer.txt", "/usr/lib/site-packages/package/data.txt"),
        ("domain/task/base/input/answer.txt", "/usr/lib/site-packages/package/data.txt"),
        ("domain/task/base/software/solution.txt", "/usr/lib/site-packages/package/data.txt"),
        ("domain/task/base/software/python_pkgs/package/data.txt", "/opt/innocent.dat"),
        ("domain/task/base/software/python_pkgs/package/data.txt", "/reference/data.txt"),
        (
            "domain/task/base/software/python_pkgs/package/data.txt",
            "/opt/runtime/lib/output/data.txt",
        ),
    ],
)
def test_renamed_payload_not_blanket_exempted(review, source, destination):
    result = review.classify_hash(destination, [manifest_row(source)], {}, {})
    assert result in {"task_hash_at_forbidden_path", "task_payload_hash_requires_review"}


@pytest.mark.parametrize("content", [b"*", b"not a standard ignore file"])
def test_runtime_metadata_requires_exact_content(review, content):
    row = manifest_row("domain/task/base/reference/evaluator_env/.venv/.gitignore", content)
    result = review.classify_hash("/opt/runtime/.venv/.gitignore", [row], {}, {})
    assert (result == "task_shared_runtime_metadata") == (content == b"*")


def test_observed_cachedir_signature_without_newline(review):
    content = b"Signature: 8a477f597d28d172789f06886806bc55"
    row = manifest_row("domain/task/base/reference/evaluator_env/.venv/CACHEDIR.TAG", content)
    result = review.classify_hash("/opt/runtime/.venv/CACHEDIR.TAG", [row], {}, {})
    assert result == "task_shared_runtime_metadata"


def test_blank_library_marker_is_not_task_content_but_paths_stay_checked(review):
    row = manifest_row("domain/task/base/software/python_pkgs/package/zip-safe", b"\n")
    assert (
        review.classify_hash("/opt/package/empty.txt", [row], {}, {})
        == "non_task_specific_blank_file"
    )
    assert (
        review.classify_hash("/reference/empty.txt", [row], {}, {}) == "task_hash_at_forbidden_path"
    )


def test_export_exclusions_are_only_observed_agent_run_state(review):
    exclusions = json.loads((Path(review.__file__).parent / "export-exclusions.json").read_text())
    assert exclusions == ["/home/user/terminus2"]


@pytest.mark.parametrize("prefix,expected", [(b"export ", True), (b"# ", False), (b"echo ", False)])
def test_focused_credential_evidence_never_records_value(review, monkeypatch, prefix, expected):
    script = Path(review.__file__).parent / "focused_findings.py"
    monkeypatch.setitem(sys.modules, "review_layers", review)
    spec = importlib.util.spec_from_file_location("focused_docker_findings", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = b"sk-or-v1-" + b"a" * 64
    record = module.credential_metadata(prefix + b"OPENROUTER_API_KEY='" + value + b"'\n")
    assert bool(record["credential_shaped_assignments"]) == expected
    assert value.decode() not in json.dumps(record)
    assert record["secret_values_recorded"] is False


def test_generic_license_review_bound_to_path_and_content(review):
    row = manifest_row("domain/task/base/input/LICENSE")
    path = "/opt/package/LICENSE"
    reviewed = {(path, row["bytes"], row["sha256"]): "generic_git_template_or_license_duplicate"}
    assert review.classify_hash(path, [row], reviewed, {}) in review.prior.BENIGN
    assert (
        review.classify_hash("/opt/other/LICENSE", [row], reviewed, {}) not in review.prior.BENIGN
    )
    changed = manifest_row(row["path"], b"different reference answer")
    assert review.classify_hash(path, [changed], reviewed, {}) not in review.prior.BENIGN


@pytest.fixture
def native_fixture(review, tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    roots = [engine / "overlay2" / (digit * 64) / "diff" for digit in ("a", "b")]
    for root in roots:
        root.mkdir(parents=True)
    content = b"synthetic-task-reference"
    (roots[0] / "reference").mkdir()
    (roots[0] / "reference/answer.txt").write_bytes(content)
    manifest = tmp_path / "members.jsonl"
    manifest.write_text(
        json.dumps(manifest_row("domain/task/base/reference/answer.txt", content)) + "\n"
    )
    layers = ["sha256:" + digit * 64 for digit in ("c", "d")]
    snapshot = {
        "Id": "sha256:" + "e" * 64,
        "RootFS": {"Layers": layers},
        "GraphDriver": {
            "Name": "overlay2",
            "Data": {"LowerDir": str(roots[0]), "UpperDir": str(roots[1])},
        },
    }
    document = {
        "layers": [{"diff_id": value} for value in layers],
        "finding_counts": {},
        "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(review.adapted, "ENGINE_ROOT", engine)
    monkeypatch.setattr(review.prior.legacy.signal, "alarm", lambda seconds: None)
    return snapshot, document, manifest, roots


def test_privileged_worker_uses_snapshot_and_native_read_only_scan(
    review, native_fixture, monkeypatch
):
    snapshot, document, manifest, _ = native_fixture
    events = []

    def no_docker(*args, **kwargs):
        raise AssertionError("privileged worker must not run Docker")

    monkeypatch.setattr(review.subprocess, "run", no_docker)
    worker = review.bound_native_worker(
        snapshot,
        document,
        {},
        manifest,
        lambda kind, **fields: events.append({"kind": kind, **fields}),
    )
    worker([inventory_row(value="synthetic-secret-unused-in-layer")])
    assert any(row["kind"] == "data_hash_match" for row in events)
    assert any(row["kind"] == "policy_match" for row in events)
    assert events[-1]["kind"] == "complete"
    assert events[-1]["completed_layers"] == [0, 1]
    assert "synthetic-secret-unused-in-layer" not in json.dumps(events)


def test_unreadable_layer_prevents_completion(review, native_fixture, monkeypatch):
    snapshot, document, manifest, _ = native_fixture
    events = []

    def unreadable(root, **kwargs):
        kwargs["onerror"](PermissionError("synthetic unreadable layer"))

    monkeypatch.setattr(review.prior.legacy.os, "walk", unreadable)
    worker = review.bound_native_worker(
        snapshot, document, {}, manifest, lambda kind, **fields: events.append(kind)
    )
    with pytest.raises(review.adapted.launch.LaunchError, match="native_path_unreadable"):
        worker([inventory_row(value="synthetic-secret-unused-in-layer")])
    assert "complete" not in events


def test_snapshot_layer_count_and_engine_scope(review, native_fixture):
    snapshot, _, _, roots = native_fixture
    assert review.layer_roots(snapshot, review.adapted.ENGINE_ROOT) == roots
    snapshot["RootFS"]["Layers"].append("sha256:" + "f" * 64)
    with pytest.raises(review.adapted.launch.LaunchError, match="layer_count_mismatch"):
        review.layer_roots(snapshot, review.adapted.ENGINE_ROOT)
    snapshot["RootFS"]["Layers"].pop()
    snapshot["GraphDriver"]["Data"]["LowerDir"] = "/var/lib/docker/overlay2/" + "a" * 64 + "/diff"
    with pytest.raises(review.adapted.launch.LaunchError, match="layer_path_outside_bound_engine"):
        review.layer_roots(snapshot, review.adapted.ENGINE_ROOT)


@pytest.mark.parametrize("filename", ["runner.sh", "origin_log.tar.gz", "prompt.txt", "stdout.log"])
def test_legacy_solver_credentials_and_logs_excluded(review, filename):
    assert review.adapted.audit.rootfs_policy.forbidden("/home/user/terminus2/" + filename, {})


def test_export_drops_legacy_run_state_without_removing_runtimes(review, tmp_path, monkeypatch):
    policy = review.adapted.audit.rootfs_policy
    monkeypatch.setattr(policy, "REQUIRED", ())
    contract = {
        "required_paths": ["/opt/runtime/bin/python"],
        "probes": [["/usr/bin/python3", "--version"], ["/usr/bin/Rscript", "--version"]],
    }
    for name in (
        "opt/runtime/bin/python",
        "opt/runtime/lib/crypto/test.key",
        "home/user/terminus2/runner.sh",
        "home/user/terminus2/origin_log.tar.gz",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic test fixture")
    output = io.BytesIO()
    policy.export_manifest(tmp_path, contract, output)
    names = output.getvalue().split(b"\0")
    assert not any(b"/terminus2" in name for name in names)
    assert b"./opt/runtime/bin/python" in names
    assert b"./opt/runtime/lib/crypto/test.key" in names


def test_new_image_export_binds_its_own_builder_inventory(review, tmp_path, monkeypatch):
    build = tmp_path / "build"
    builder = build / ("a" * 32)
    builder.mkdir(parents=True)
    runtime, data = tmp_path / "runtime.json", tmp_path / "members.jsonl"
    runtime.write_text("{}\n")
    data.write_text("{}\n")
    identity = {
        "runtime_manifest_sha256": review.adapted.digest(runtime),
        "data_manifest_sha256": review.adapted.digest(data),
    }
    (builder / "identity.json").write_text(json.dumps(identity))
    exported = {"builder_evidence": str(builder), "identity": identity}
    receipt = build / "export.json"
    receipt.write_text(json.dumps(exported))
    monkeypatch.setattr(review.adapted.launch, "HERE", review.adapted.launch.HERE)
    monkeypatch.setattr(review.adapted.launch, "BUILDER", review.adapted.launch.BUILDER)
    assert review.configure_export(receipt, runtime, data) == exported
    assert review.adapted.launch.HERE == tmp_path
    assert review.adapted.launch.BUILDER == builder
    data.write_text('{"changed": true}\n')
    with pytest.raises(review.adapted.launch.LaunchError, match="data_manifest_changed"):
        review.configure_export(receipt, runtime, data)
    data.write_text("{}\n")
    (builder / "identity.json").write_text("{}\n")
    with pytest.raises(review.adapted.launch.LaunchError, match="builder_identity_changed"):
        review.configure_export(receipt, runtime, data)
