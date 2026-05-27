from __future__ import annotations

import argparse
import html
import io
import json
import re
import string
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONTRACT = {
    "output_filename": "migrated_output.unitypackage",
    "scene_targets": {
        "An FPS Game": "_FPS",
        "An RPG Game": "_RPG",
        "Visual Novel Game": "_VisualNovel",
        "Strategy Game": "_Strategy",
        "D&D RPG Game": "_D&DRPG",
        "ARPG Game": "_ARPG",
    },
    "visible_ui_requirements": {
        "header_text": "Select Minigame Scene",
        "load_button_text": "Load Scene",
        "card_labels": [
            "An FPS Game",
            "An RPG Game",
            "Visual Novel Game",
            "Strategy Game",
            "D&D RPG Game",
            "ARPG Game",
        ],
        "requires_scrollable_card_layout": True,
        "requires_icon_element_per_card": True,
        "requires_selected_highlight_state": True,
    },
    "behavioral_requirements": {
        "load_button_disabled_until_selection": True,
        "preserve_additive_scene_loading_flow": True,
        "single_selection_behavior": True,
    },
    "structural_requirements": {
        "requires_unity_scene": True,
        "requires_ui_toolkit_migration": True,
        "requires_runtime_ui_toolkit_setup": True,
    },
}

TEXT_EXTENSIONS = {
    ".asset",
    ".asmdef",
    ".cs",
    ".json",
    ".md",
    ".meta",
    ".txt",
    ".unity",
    ".inputactions",
    ".uss",
    ".uxml",
    ".xml",
    ".yaml",
    ".yml",
}
UITK_MARKERS = (
    "UnityEngine.UIElements",
    "UIDocument",
    "<ui:UXML",
    "VisualElement",
    "ScrollView",
    "Button",
    "SetEnabled(",
)


@dataclass
class ParsedUnityPackage:
    pathnames: list[str]
    text_assets: dict[str, str]
    guid_to_path: dict[str, str]

    @property
    def scene_paths(self) -> list[str]:
        return [path for path in self.pathnames if path.lower().endswith(".unity")]

    @property
    def corpus(self) -> str:
        pieces = list(self.pathnames) + list(self.text_assets.values())
        return html.unescape("\n".join(pieces))

    def referenced_paths_for_scene(self, scene_path: str) -> list[str]:
        scene_text = self.text_assets.get(scene_path, "")
        guids = re.findall(r"guid:\s*([0-9a-f]{32})", scene_text, flags=re.IGNORECASE)
        return sorted({self.guid_to_path[guid] for guid in guids if guid in self.guid_to_path})


def _member_prefix(name: str) -> str:
    if "/" not in name:
        return ""
    return name.rsplit("/", 1)[0] + "/"


def _looks_like_text(path: str, data: bytes) -> bool:
    if Path(path).suffix.lower() in TEXT_EXTENSIONS:
        return True
    if not data:
        return True
    if b"\x00" in data:
        return False
    sample = data[:4096]
    printable = sum(chr(byte) in string.printable or byte >= 0x80 for byte in sample)
    return printable / max(len(sample), 1) >= 0.85


def parse_unitypackage_bytes(blob: bytes) -> ParsedUnityPackage:
    pathnames: list[str] = []
    text_assets: dict[str, str] = {}
    guid_to_path: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
        prefix_to_path: dict[str, str] = {}

        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith("/pathname"):
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            pathname = extracted.read().decode("utf-8", "replace").strip()
            prefix = _member_prefix(member.name)
            prefix_to_path[prefix] = pathname
            guid = prefix.rstrip("/").split("/")[-1]
            if guid:
                guid_to_path[guid] = pathname
            pathnames.append(pathname)

        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith("/asset"):
                continue
            pathname = prefix_to_path.get(_member_prefix(member.name))
            if not pathname:
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            if not _looks_like_text(pathname, data):
                continue
            text_assets[pathname] = data.decode("utf-8", "replace")

    return ParsedUnityPackage(pathnames=sorted(pathnames), text_assets=text_assets, guid_to_path=guid_to_path)


def _count_present(items: list[str], corpus: str) -> int:
    corpus_lower = corpus.lower()
    return sum(1 for item in items if html.unescape(item).lower() in corpus_lower)


def _contains_any(corpus: str, markers: tuple[str, ...]) -> bool:
    corpus_lower = corpus.lower()
    return any(marker.lower() in corpus_lower for marker in markers)


def _scene_flow_ok(corpus: str) -> bool:
    return bool(re.search(r"SceneController\s*\.\s*Instance\s*\.\s*LoadScene", corpus))


def _selection_state_ok(corpus: str) -> bool:
    markers = (
        "is-selected",
        "AddToClassList",
        "RemoveFromClassList",
        "EnableInClassList",
        "Highlight",
        "selected",
    )
    return _contains_any(corpus, markers)


def _icon_evidence_ok(corpus: str) -> bool:
    markers = (
        'name="Icon"',
        'Q<Image>("Icon")',
        'Q<VisualElement>("Icon")',
        "class=\"card-icon\"",
        "backgroundImage",
        "new Image(",
        "new Image()",
    )
    return _contains_any(corpus, markers)


def _scrollable_layout_ok(corpus: str) -> bool:
    markers = (
        "ScrollView",
        "unity-scroll-view__content-container",
        "scroll-view",
        "scroller-visibility",
        "new ScrollView(",
        "new ScrollView()",
    )
    return _contains_any(corpus, markers)


def _button_disabled_until_selection_ok(corpus: str) -> bool:
    disabled = _contains_any(corpus, ("SetEnabled(false)", ".interactable = false"))
    enabled = _contains_any(corpus, ("SetEnabled(true)", ".interactable = true"))
    return disabled and enabled


def _single_selection_ok(corpus: str) -> bool:
    markers = (
        "RemoveFromClassList",
        "ToggleGroup",
        "mutually exclusive",
        "single selection",
        "_allHighlights",
    )
    return _contains_any(corpus, markers)


def _code_generated_uitk_ok(corpus: str) -> bool:
    markers = (
        "new VisualElement(",
        "new Button(",
        "new Label(",
        "new ScrollView(",
        "rootVisualElement.Add(",
    )
    return _contains_any(corpus, markers)


def _join_texts(package: ParsedUnityPackage, paths: list[str]) -> str:
    pieces = [package.text_assets[path] for path in paths if path in package.text_assets]
    return html.unescape("\n".join(pieces))


def _candidate_scene_result(
    package: ParsedUnityPackage,
    scene_path: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    visible = contract.get("visible_ui_requirements", {})
    required_labels = list(visible.get("card_labels") or [])
    required_targets = list((contract.get("scene_targets") or {}).values())
    header_text = visible.get("header_text", DEFAULT_CONTRACT["visible_ui_requirements"]["header_text"])
    load_button_text = visible.get(
        "load_button_text", DEFAULT_CONTRACT["visible_ui_requirements"]["load_button_text"]
    )

    scene_text = package.text_assets.get(scene_path, "")
    referenced_paths = package.referenced_paths_for_scene(scene_path)
    script_paths = [path for path in referenced_paths if path.lower().endswith(".cs")]
    uxml_paths = [path for path in referenced_paths if path.lower().endswith(".uxml")]
    uss_paths = [path for path in referenced_paths if path.lower().endswith(".uss")]

    script_corpus = _join_texts(package, script_paths)
    uxml_corpus = _join_texts(package, uxml_paths + uss_paths)
    linked_corpus = html.unescape(
        "\n".join([scene_path, scene_text, *referenced_paths, script_corpus, uxml_corpus])
    )

    labels_found = _count_present(required_labels, linked_corpus)
    targets_found = _count_present(required_targets, script_corpus + "\n" + linked_corpus)
    labels_ratio = labels_found / max(len(required_labels), 1)
    targets_ratio = targets_found / max(len(required_targets), 1)

    header_ok = header_text.lower() in linked_corpus.lower()
    load_button_ok = load_button_text.lower() in linked_corpus.lower()
    icon_ok = _icon_evidence_ok(uxml_corpus + "\n" + script_corpus)
    scroll_ok = _scrollable_layout_ok(uxml_corpus + "\n" + script_corpus)
    selection_ok = _selection_state_ok(uxml_corpus + "\n" + script_corpus)
    disabled_ok = _button_disabled_until_selection_ok(script_corpus + "\n" + linked_corpus)
    single_selection_ok = _single_selection_ok(script_corpus + "\n" + linked_corpus)
    additive_flow_ok = _scene_flow_ok(script_corpus)

    has_uidocument = _contains_any(scene_text, ("UIDocument", "UnityEngine.UIElements.UIDocument"))
    has_panel_settings = any(path.lower().endswith("panelsettings.asset") for path in referenced_paths)
    has_uitk_script = _contains_any(script_corpus, ("UnityEngine.UIElements", "UIDocument", "VisualElement"))
    generated_tree_ok = _code_generated_uitk_ok(script_corpus)
    has_authoring_surface = bool(uxml_paths) or generated_tree_ok

    component_values = [
        1.0 if header_ok else 0.0,
        1.0 if load_button_ok else 0.0,
        labels_ratio,
        targets_ratio,
        1.0 if icon_ok else 0.0,
        1.0 if scroll_ok else 0.0,
        1.0 if selection_ok else 0.0,
        1.0 if disabled_ok else 0.0,
        1.0 if single_selection_ok else 0.0,
        1.0 if additive_flow_ok else 0.0,
        1.0 if has_uidocument else 0.0,
        1.0 if has_panel_settings else 0.0,
        1.0 if has_uitk_script else 0.0,
        1.0 if has_authoring_surface else 0.0,
    ]

    return {
        "scene_path": scene_path,
        "referenced_paths": referenced_paths,
        "script_paths": script_paths,
        "uxml_paths": uxml_paths,
        "uss_paths": uss_paths,
        "header_ok": header_ok,
        "load_button_ok": load_button_ok,
        "labels_found": labels_found,
        "labels_expected": len(required_labels),
        "labels_ratio": labels_ratio,
        "targets_found": targets_found,
        "targets_expected": len(required_targets),
        "targets_ratio": targets_ratio,
        "icon_ok": icon_ok,
        "scroll_ok": scroll_ok,
        "selection_ok": selection_ok,
        "disabled_ok": disabled_ok,
        "single_selection_ok": single_selection_ok,
        "additive_flow_ok": additive_flow_ok,
        "has_uidocument": has_uidocument,
        "has_panel_settings": has_panel_settings,
        "has_uitk_script": has_uitk_script,
        "has_authoring_surface": has_authoring_surface,
        "generated_tree_ok": generated_tree_ok,
        "linked_score": sum(component_values) / len(component_values),
    }


def evaluate_unitypackage_bytes(
    blob: bytes,
    *,
    contract: dict[str, Any] | None = None,
    package_name: str = "migrated_output.unitypackage",
) -> dict[str, Any]:
    contract = contract or DEFAULT_CONTRACT
    try:
        package = parse_unitypackage_bytes(blob)
    except tarfile.TarError as exc:
        return {
            "final_score": 0.0,
            "hard_fail": "unreadable_unitypackage",
            "details": {"error": str(exc), "package_name": package_name},
        }

    corpus = package.corpus
    structure = contract.get("structural_requirements", {})

    has_scene = bool(package.scene_paths)
    has_uitk_evidence = _contains_any(corpus, UITK_MARKERS)
    candidate_scene_paths = [
        scene_path
        for scene_path in package.scene_paths
        if _contains_any(package.text_assets.get(scene_path, ""), ("UIDocument", "UnityEngine.UIElements.UIDocument"))
    ]
    candidate_results = [
        _candidate_scene_result(package, scene_path, contract) for scene_path in candidate_scene_paths
    ]
    best = max(candidate_results, key=lambda item: item["linked_score"], default=None)

    if structure.get("requires_unity_scene", True) and not has_scene:
        return {
            "final_score": 0.0,
            "hard_fail": "missing_unity_scene",
            "details": {"scene_paths": package.scene_paths, "path_count": len(package.pathnames)},
        }
    if structure.get("requires_ui_toolkit_migration", True) and not has_uitk_evidence:
        return {
            "final_score": 0.0,
            "hard_fail": "missing_ui_toolkit_evidence",
            "details": {"scene_paths": package.scene_paths, "path_count": len(package.pathnames)},
        }
    if best is None:
        return {
            "final_score": 0.0,
            "hard_fail": "missing_scene_linked_ui_toolkit_scene",
            "details": {"candidate_scene_paths": candidate_scene_paths, "path_count": len(package.pathnames)},
        }
    has_runtime_setup = best["has_uidocument"] and (best["has_panel_settings"] or best["has_authoring_surface"])
    if structure.get("requires_runtime_ui_toolkit_setup", True) and not has_runtime_setup:
        return {
            "final_score": 0.0,
            "hard_fail": "missing_runtime_ui_toolkit_setup",
            "details": best,
        }
    if (
        not best["header_ok"]
        or not best["load_button_ok"]
        or best["labels_ratio"] < 1.0
        or not best["icon_ok"]
        or not best["selection_ok"]
    ):
        return {
            "final_score": 0.0,
            "hard_fail": "missing_required_visible_ui",
            "details": best,
        }

    structure_score = sum(
        [
            1.0 if best["has_authoring_surface"] else 0.0,
            1.0 if best["scroll_ok"] else 0.0,
            1.0 if best["icon_ok"] else 0.0,
        ]
    ) / 3.0
    behavior_score = sum(
        [
            1.0 if best["additive_flow_ok"] else 0.0,
            1.0 if best["disabled_ok"] else 0.0,
            1.0 if best["selection_ok"] else 0.0,
            1.0 if best["single_selection_ok"] else 0.0,
            best["targets_ratio"],
        ]
    ) / 5.0
    migration_score = sum(
        [
            1.0 if has_uitk_evidence else 0.0,
            1.0 if has_runtime_setup else 0.0,
            1.0 if has_scene else 0.0,
        ]
    ) / 3.0
    visible_score = sum(
        [
            1.0 if best["header_ok"] else 0.0,
            1.0 if best["load_button_ok"] else 0.0,
            best["labels_ratio"],
        ]
    ) / 3.0

    final_score = round(
        (0.25 * structure_score) + (0.35 * behavior_score) + (0.20 * migration_score) + (0.20 * visible_score),
        4,
    )

    return {
        "final_score": final_score,
        "hard_fail": None,
        "details": {
            "path_count": len(package.pathnames),
            "scene_paths": package.scene_paths,
            "has_uitk_evidence": has_uitk_evidence,
            "candidate_scene_paths": candidate_scene_paths,
            "best_candidate": best,
        },
    }


def evaluate_unitypackage_file(
    package_path: Path, *, contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    return evaluate_unitypackage_bytes(package_path.read_bytes(), contract=contract, package_name=package_path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_path", type=Path)
    parser.add_argument("--contract-json", type=Path)
    args = parser.parse_args()

    contract = DEFAULT_CONTRACT
    if args.contract_json:
        contract = json.loads(args.contract_json.read_text(encoding="utf-8"))
    result = evaluate_unitypackage_file(args.package_path, contract=contract)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
