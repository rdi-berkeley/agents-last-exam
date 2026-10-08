import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def evaluator():
    scripts = Path(__file__).resolve().parents[2] / "tasks/visual_media/project_migration/scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "migration_copy_test", scripts / "evaluate_remote.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_evaluator_library_aliases_are_compact_and_isolated(tmp_path, evaluator):
    source, destination = tmp_path / "submitted", tmp_path / "evaluator"
    (source / "externals").mkdir(parents=True)
    (source / "plugins").mkdir()
    library = source / "externals/bank.sf2"
    library.write_bytes(b"retained sample library")
    (source / "plugins/bank.sf2").symlink_to("../externals/bank.sf2")
    (source / "notes.mid").write_bytes(b"original editable notes")
    evaluator.copy_project(source, destination)
    copied = destination / "externals/bank.sf2"
    alias = destination / "plugins/bank.sf2"
    assert copied.read_bytes() == alias.read_bytes() == library.read_bytes()
    assert alias.is_symlink()
    assert copied.stat().st_ino == alias.stat().st_ino != library.stat().st_ino
    copied.write_bytes(b"evaluator changed its own copy")
    (destination / "notes.mid").write_bytes(b"native saved notes")
    assert library.read_bytes() == b"retained sample library"
    assert (source / "notes.mid").read_bytes() == b"original editable notes"


@pytest.mark.parametrize("absolute", [False, True])
def test_copied_configuration_resolves_its_original_relative_assets(tmp_path, evaluator, absolute):
    source, destination = tmp_path / "submitted", tmp_path / "evaluator"
    (source / "instruments/patches").mkdir(parents=True)
    (source / "plugins/state3").mkdir(parents=True)
    config = source / "instruments/Violin_1.tsv"
    config.write_text("patches/violin.sfz\n")
    asset = source / "instruments/patches/violin.sfz"
    asset.write_text("original sample mapping")
    (source / "instrument-alias").symlink_to("instruments", target_is_directory=True)
    target = (
        source / "instrument-alias/Violin_1.tsv"
        if absolute
        else "../../instrument-alias/Violin_1.tsv"
    )
    (source / "plugins/state3/Violin_1.tsv").symlink_to(target)

    evaluator.copy_project(source, destination)

    copied_config = (destination / "plugins/state3/Violin_1.tsv").resolve()
    copied_asset = copied_config.parent / copied_config.read_text().strip()
    assert copied_config.is_relative_to(destination)
    assert copied_asset.read_text() == "original sample mapping"
    copied_asset.write_text("evaluator-only change")
    assert asset.read_text() == "original sample mapping"


@pytest.mark.parametrize("directory", [False, True])
def test_external_alias_is_copied_without_write_through(tmp_path, evaluator, directory):
    source, destination = tmp_path / "submitted", tmp_path / "evaluator"
    source.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    original = external / "bank.sf2"
    original.write_bytes(b"original library")
    (source / "asset").symlink_to(
        external if directory else original, target_is_directory=directory
    )

    evaluator.copy_project(source, destination)

    copied = destination / ("asset/bank.sf2" if directory else "asset")
    assert not (destination / "asset").is_symlink()
    assert copied.read_bytes() == original.read_bytes()
    copied.write_bytes(b"evaluator-only change")
    assert original.read_bytes() == b"original library"


def test_regular_library_hardlinks_remain_compact_and_isolated(tmp_path, evaluator):
    source, destination = tmp_path / "submitted", tmp_path / "evaluator"
    source.mkdir()
    original = source / "bank.sf2"
    original.write_bytes(b"original library")
    os.link(original, source / "alias.sf2")

    evaluator.copy_project(source, destination)

    copied, alias = destination / "bank.sf2", destination / "alias.sf2"
    assert copied.stat().st_ino == alias.stat().st_ino != original.stat().st_ino


@pytest.mark.parametrize(
    "inventory", [None, " \n\t", "Installed sample library and plugin inventory"]
)
def test_original_inventory_requirement_rejects_missing_or_empty(tmp_path, inventory):
    scripts = Path(__file__).resolve().parents[2] / "tasks/visual_media/project_migration/scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "migration_inventory_test", scripts / "evaluate_remote.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    references, output = tmp_path / "references", tmp_path / "submission"
    references.mkdir()
    output.mkdir()
    sf.write(references / "Harp.wav", np.ones((100, 2)) * 0.1, 48000)
    if inventory is not None:
        (output / "available_plugins.txt").write_text(inventory)
    result = module.evaluate_local(
        output, tmp_path / "source.ardour", references, tmp_path / "evidence", native=False
    )
    missing_inventory = any(
        name.startswith("available_plugins.txt") for name in result["missing_files"]
    )
    assert missing_inventory is not bool(inventory and inventory.strip())
    assert result["inventory_nonempty"] is bool(inventory and inventory.strip())
