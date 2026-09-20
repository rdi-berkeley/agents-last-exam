from ale_run.environments.providers.gcloud import (
    _build_create_args,
    _build_snapshot_config,
)


def test_snapshot_boot_disk_size_is_optional() -> None:
    default = _build_snapshot_config({
        "image": "ale-ubuntu22",
        "zones": ["us-central1-a"],
    })
    expanded = _build_snapshot_config({
        "image": "ale-ubuntu22",
        "zones": ["us-central1-a"],
        "boot_disk_size_gb": 100,
    })

    assert default.boot_disk_size_gb is None
    assert expanded.boot_disk_size_gb == 100


def test_create_args_include_boot_disk_size_override() -> None:
    args = _build_create_args(
        name="ale-test",
        image="ale-ubuntu22",
        gpu=None,
        os_type="linux",
        network="default",
        subnet="default",
        machine_type="c4-standard-4",
        zone="us-central1-a",
        label_str="purpose=ale-run",
        project="test-project",
        boot_disk_type="hyperdisk-balanced",
        boot_disk_size_gb=100,
    )

    assert "--boot-disk-size=100GB" in args
