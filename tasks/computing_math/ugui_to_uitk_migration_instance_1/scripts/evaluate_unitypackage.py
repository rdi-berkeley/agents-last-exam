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

from tasks.computing_math.ugui_to_uitk_migration_instance_1.scripts.task_specs import VARIANT_SPECS

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
        return html.unescape("\n".join([*self.pathnames, *self.text_assets.values()]))

    def referenced_paths_for_scene(self, scene_path: str) -> list[str]:
        scene_text = self.text_assets.get(scene_path, "")
        guids = re.findall(r"guid:\s*([0-9a-f]{32})", scene_text, flags=re.IGNORECASE)
        return sorted({self.guid_to_path[guid] for guid in guids if guid in self.guid_to_path})


@dataclass
class SceneCandidate:
    scene_path: str
    scene_text: str
    referenced_paths: list[str]
    script_paths: list[str]
    uxml_paths: list[str]
    uss_paths: list[str]
    script_corpus: str
    layout_corpus: str
    linked_corpus: str


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


def _contains_any(corpus: str, markers: tuple[str, ...] | list[str]) -> bool:
    corpus_lower = corpus.lower()
    return any(marker.lower() in corpus_lower for marker in markers)


def _contains_all(corpus: str, markers: tuple[str, ...] | list[str]) -> bool:
    corpus_lower = corpus.lower()
    return all(marker.lower() in corpus_lower for marker in markers)


def _count_present(items: list[str], corpus: str) -> int:
    corpus_lower = corpus.lower()
    return sum(1 for item in items if html.unescape(item).lower() in corpus_lower)


def _count_occurrences(corpus: str, text: str) -> int:
    return corpus.lower().count(text.lower())


def _join_texts(package: ParsedUnityPackage, paths: list[str]) -> str:
    return html.unescape("\n".join(package.text_assets[path] for path in paths if path in package.text_assets))


def _uxml_referenced_uss_paths(package: ParsedUnityPackage, uxml_paths: list[str]) -> list[str]:
    discovered: set[str] = set()
    known_paths = set(package.text_assets)
    for uxml_path in uxml_paths:
        uxml_text = package.text_assets.get(uxml_path, "")
        parent = Path(uxml_path).parent
        for raw_ref in re.findall(r'<Style\s+src="([^"]+)"', uxml_text):
            candidate = str(parent / raw_ref).replace("\\", "/")
            if candidate in known_paths and candidate.lower().endswith(".uss"):
                discovered.add(candidate)
    return sorted(discovered)


def _scene_candidates(package: ParsedUnityPackage) -> list[SceneCandidate]:
    candidates: list[SceneCandidate] = []
    for scene_path in package.scene_paths:
        scene_text = package.text_assets.get(scene_path, "")
        if "uidocument" not in scene_text.lower():
            continue
        referenced_paths = package.referenced_paths_for_scene(scene_path)
        script_paths = [path for path in referenced_paths if path.lower().endswith(".cs")]
        uxml_paths = [path for path in referenced_paths if path.lower().endswith(".uxml")]
        uss_paths = [path for path in referenced_paths if path.lower().endswith(".uss")]
        for extra_uss in _uxml_referenced_uss_paths(package, uxml_paths):
            if extra_uss not in uss_paths:
                uss_paths.append(extra_uss)
        script_corpus = _join_texts(package, script_paths)
        layout_corpus = _join_texts(package, [*uxml_paths, *uss_paths])
        linked_corpus = html.unescape(
            "\n".join([scene_path, scene_text, *referenced_paths, script_corpus, layout_corpus])
        )
        candidates.append(
            SceneCandidate(
                scene_path=scene_path,
                scene_text=scene_text,
                referenced_paths=referenced_paths,
                script_paths=script_paths,
                uxml_paths=uxml_paths,
                uss_paths=uss_paths,
                script_corpus=script_corpus,
                layout_corpus=layout_corpus,
                linked_corpus=linked_corpus,
            )
        )
    return candidates


def _runtime_setup_ok(candidate: SceneCandidate) -> bool:
    has_uidocument = "uidocument" in candidate.scene_text.lower()
    has_panel_settings = any(path.lower().endswith("panelsettings.asset") for path in candidate.referenced_paths)
    has_authoring_surface = bool(candidate.uxml_paths) or _contains_any(
        candidate.script_corpus,
        (
            "new VisualElement(",
            "new Button(",
            "new Label(",
            "new ScrollView(",
            "rootVisualElement.Add(",
        ),
    )
    return has_uidocument and (has_panel_settings or has_authoring_surface)


def _candidate_summary(candidate: SceneCandidate) -> dict[str, Any]:
    return {
        "scene_path": candidate.scene_path,
        "referenced_paths": candidate.referenced_paths,
        "script_paths": candidate.script_paths,
        "uxml_paths": candidate.uxml_paths,
        "uss_paths": candidate.uss_paths,
    }


def _score_candidates(candidates: list[SceneCandidate], score_fn) -> list[tuple[SceneCandidate, dict[str, Any]]]:
    return [(candidate, score_fn(candidate)) for candidate in candidates]


def _round_score(*parts: float) -> float:
    return round(sum(parts) / len(parts), 4)


def _scrollview_blocks(layout_corpus: str) -> list[str]:
    return re.findall(r"(?is)<ui:ScrollView\b.*?</ui:ScrollView>", layout_corpus)


def _start_menu_label_to_card(layout_corpus: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pattern = re.compile(
        r'(?is)<ui:VisualElement[^>]*name="([^"]+)"[^>]*class="scene-card"[^>]*>.*?<ui:Label[^>]*text="([^"]+)"'
    )
    for card_name, label in pattern.findall(layout_corpus):
        mapping[html.unescape(label)] = card_name
    return mapping


def _start_menu_binding_ratio(candidate: SceneCandidate, expected_targets: dict[str, str]) -> tuple[float, dict[str, str]]:
    label_to_card = _start_menu_label_to_card(candidate.layout_corpus)
    matched = 0
    for label, target in expected_targets.items():
        card_name = label_to_card.get(label)
        if not card_name:
            continue
        pair_pattern = re.compile(
            rf"cardElementName:\s*{re.escape(card_name)}\s+targetSceneName:\s*{re.escape(target)}",
            flags=re.IGNORECASE,
        )
        if pair_pattern.search(candidate.scene_text):
            matched += 1
    return matched / max(len(expected_targets), 1), label_to_card


def _start_menu_scroll_layout_ok(candidate: SceneCandidate, labels: list[str]) -> bool:
    blocks = _scrollview_blocks(candidate.layout_corpus)
    return any(_contains_all(html.unescape(block), labels) for block in blocks)


def _score_start_menu(candidate: SceneCandidate) -> dict[str, Any]:
    contract = VARIANT_SPECS["start_menu"].output_contract
    visible = contract["visible_ui_requirements"]
    linked = candidate.linked_corpus
    ui_and_script = candidate.layout_corpus + "\n" + candidate.script_corpus

    card_labels = list(visible["card_labels"])
    scene_targets = dict(visible["scene_targets"])
    labels_ratio = _count_present(card_labels, linked) / len(card_labels)
    targets_ratio, label_to_card = _start_menu_binding_ratio(candidate, scene_targets)
    header_ok = visible["header_text"].lower() in linked.lower()
    load_button_ok = visible["load_button_text"].lower() in linked.lower()
    scroll_ok = _start_menu_scroll_layout_ok(candidate, card_labels)
    icon_ok = _contains_any(
        ui_and_script,
        ('name="Icon"', 'Q<Image>("Icon")', "backgroundImage", "new Image(", "new Image()"),
    )
    selection_ok = _contains_any(
        ui_and_script,
        ("is-selected", "AddToClassList", "RemoveFromClassList", "EnableInClassList", "Highlight"),
    )
    disabled_ok = _contains_any(candidate.script_corpus, ("SetEnabled(false)", ".interactable = false")) and _contains_any(
        candidate.script_corpus, ("SetEnabled(true)", ".interactable = true")
    )
    single_selection_ok = _contains_any(candidate.script_corpus, ("RemoveFromClassList", "_allHighlights", "single selection"))
    additive_flow_ok = "scenecontroller.instance.loadscene" in candidate.script_corpus.lower()

    visible_score = _round_score(
        1.0 if header_ok else 0.0,
        1.0 if load_button_ok else 0.0,
        labels_ratio,
    )
    structure_score = _round_score(
        1.0 if scroll_ok else 0.0,
        1.0 if icon_ok else 0.0,
        1.0 if _runtime_setup_ok(candidate) else 0.0,
    )
    behavior_score = _round_score(
        1.0 if selection_ok else 0.0,
        1.0 if disabled_ok else 0.0,
        1.0 if single_selection_ok else 0.0,
        1.0 if additive_flow_ok else 0.0,
        targets_ratio,
    )
    candidate_score = round((0.30 * visible_score) + (0.35 * behavior_score) + (0.35 * structure_score), 4)
    return {
        **_candidate_summary(candidate),
        "candidate_score": candidate_score,
        "header_ok": header_ok,
        "load_button_ok": load_button_ok,
        "label_to_card": label_to_card,
        "labels_ratio": labels_ratio,
        "targets_ratio": targets_ratio,
        "scroll_ok": scroll_ok,
        "icon_ok": icon_ok,
        "selection_ok": selection_ok,
        "disabled_ok": disabled_ok,
        "single_selection_ok": single_selection_ok,
        "additive_flow_ok": additive_flow_ok,
    }


def _score_fps_hud(candidate: SceneCandidate) -> dict[str, Any]:
    contract = VARIANT_SPECS["fps_hud"].output_contract
    visible = contract["visible_ui_requirements"]
    linked = candidate.linked_corpus
    ui_and_script = candidate.layout_corpus + "\n" + candidate.script_corpus

    names = list(visible["required_named_elements"])
    named_ratio = _count_present(names, linked) / len(names)
    numbers_ok = _count_occurrences(linked, "100") >= 2
    weapon_icons_ok = _contains_any(
        ui_and_script,
        ("WeaponIcon", "PistolIcon", "RifleIcon", "KnifeIcon", "HeartIcon", "ArmorIcon"),
    )
    buy_menu_toggle_ok = _contains_all(candidate.script_corpus, ("ToggleBuyMenu", "<Keyboard>/b")) and _contains_any(
        candidate.script_corpus, ("DisplayStyle.None", "DisplayStyle.Flex", "_isBuyMenuOpen")
    )
    hud_update_ok = _contains_any(
        candidate.script_corpus,
        ("HealthText", "ArmorText", "heartIconSprite", "armorIconSprite", "SetBackgroundImage"),
    )
    minimap_ok = _contains_any(ui_and_script, ("MinimapContainer", "Map_Texture", "de_dust2_radar"))
    crosshair_ok = "crosshair" in ui_and_script.lower()

    visible_score = _round_score(
        named_ratio,
        1.0 if numbers_ok else 0.0,
        1.0 if weapon_icons_ok else 0.0,
    )
    behavior_score = _round_score(
        1.0 if buy_menu_toggle_ok else 0.0,
        1.0 if hud_update_ok else 0.0,
        1.0 if minimap_ok else 0.0,
        1.0 if crosshair_ok else 0.0,
    )
    structure_score = _round_score(
        1.0 if _runtime_setup_ok(candidate) else 0.0,
        1.0 if bool(candidate.uxml_paths) else 0.0,
        1.0 if bool(candidate.script_paths) else 0.0,
    )
    candidate_score = round((0.35 * visible_score) + (0.35 * behavior_score) + (0.30 * structure_score), 4)
    return {
        **_candidate_summary(candidate),
        "candidate_score": candidate_score,
        "named_ratio": named_ratio,
        "numbers_ok": numbers_ok,
        "weapon_icons_ok": weapon_icons_ok,
        "buy_menu_toggle_ok": buy_menu_toggle_ok,
        "hud_update_ok": hud_update_ok,
        "minimap_ok": minimap_ok,
        "crosshair_ok": crosshair_ok,
    }


def _score_visual_novel(candidate: SceneCandidate) -> dict[str, Any]:
    contract = VARIANT_SPECS["visual_novel"].output_contract
    visible = contract["visible_ui_requirements"]
    linked = candidate.linked_corpus
    ui_and_script = candidate.layout_corpus + "\n" + candidate.script_corpus

    names = list(visible["required_named_elements"])
    texts = list(visible["required_text_values"])
    named_ratio = _count_present(names, linked) / len(names)
    text_ratio = _count_present(texts, linked) / len(texts)
    background_and_portrait_ok = _contains_all(linked, ("BG", "Character_Portrait"))
    save_slot_template_ok = _contains_any(ui_and_script, ("saveSlotTemplate", "VN_SaveSlotLayout.uxml", "Btn_Delete"))
    confirm_control_ok = _contains_any(linked, ("Btn_Confirm", "CONFIRM", "confirm"))
    dialogue_flow_ok = _contains_any(
        candidate.script_corpus,
        ("OnDialogueUpdate", "UpdateDialogueDisplay", "_canAdvanceDialogue", "RegisterCallback<ClickEvent>"),
    )
    options_ok = _contains_any(candidate.script_corpus, ("OnOptionsProvided", "CreateOptionButtons", "Options_Container"))
    save_load_ok = _contains_any(candidate.script_corpus, ("SaveGame", "LoadGame", "Slots_Container", "Btn_Confirm", "Btn_Close"))

    visible_score = _round_score(
        named_ratio,
        text_ratio,
        1.0 if background_and_portrait_ok else 0.0,
    )
    behavior_score = _round_score(
        1.0 if dialogue_flow_ok else 0.0,
        1.0 if options_ok else 0.0,
        1.0 if save_load_ok else 0.0,
        1.0 if save_slot_template_ok else 0.0,
        1.0 if confirm_control_ok else 0.0,
    )
    structure_score = _round_score(
        1.0 if _runtime_setup_ok(candidate) else 0.0,
        1.0 if bool(candidate.uxml_paths) else 0.0,
        1.0 if bool(candidate.script_paths) else 0.0,
    )
    candidate_score = round((0.35 * visible_score) + (0.35 * behavior_score) + (0.30 * structure_score), 4)
    return {
        **_candidate_summary(candidate),
        "candidate_score": candidate_score,
        "named_ratio": named_ratio,
        "text_ratio": text_ratio,
        "background_and_portrait_ok": background_and_portrait_ok,
        "save_slot_template_ok": save_slot_template_ok,
        "confirm_control_ok": confirm_control_ok,
        "dialogue_flow_ok": dialogue_flow_ok,
        "options_ok": options_ok,
        "save_load_ok": save_load_ok,
    }


def _score_world_inventory(candidate: SceneCandidate) -> dict[str, Any]:
    contract = VARIANT_SPECS["world_inventory"].output_contract
    visible = contract["visible_ui_requirements"]
    linked = candidate.linked_corpus
    ui_and_script = candidate.layout_corpus + "\n" + candidate.script_corpus

    slot_numbers = {
        int(match.group(1))
        for match in re.finditer(r"\bslot[_\s-]?([0-9]+)\b", linked, flags=re.IGNORECASE)
        if int(match.group(1)) < 12
    }
    slot_count = len(slot_numbers)
    slot_ratio = min(slot_count / int(visible["minimum_slot_count"]), 1.0)
    bottom_bar_ok = _contains_any(ui_and_script, ("Inventory_BottomBar", "bottom inventory bar", "flex-direction: row"))
    drag_ghost_ok = _contains_any(ui_and_script, ("DragGhost", "drag ghost", "display = DisplayStyle.None"))
    slot_highlight_ok = _contains_any(ui_and_script, ("slot--highlight", "AddToClassList(\"slot--highlight\")"))
    item_count_ok = _contains_any(ui_and_script, ("ui-item__count", "x{itemStack.count}", "count > 1"))
    pointer_events_ok = _contains_all(
        candidate.script_corpus,
        ("RegisterCallback<PointerDownEvent>", "RegisterCallback<PointerMoveEvent>", "RegisterCallback<PointerUpEvent>"),
    )
    drag_drop_ok = _contains_any(candidate.script_corpus, ("DragGhost", "TryAddItemToSlot", "ClearSlot", "draggedItem"))
    inventory_events_ok = _contains_any(candidate.script_corpus, ("InventoryDataManager", "OnSlotUpdated", "ItemStack"))
    world_spawn_ok = _contains_any(candidate.script_corpus, ("SpawnManager", "WorldItemData", "world item"))

    visible_score = _round_score(
        slot_ratio,
        1.0 if bottom_bar_ok else 0.0,
        1.0 if drag_ghost_ok else 0.0,
        1.0 if slot_highlight_ok else 0.0,
        1.0 if item_count_ok else 0.0,
    )
    behavior_score = _round_score(
        1.0 if pointer_events_ok else 0.0,
        1.0 if drag_drop_ok else 0.0,
        1.0 if inventory_events_ok else 0.0,
        1.0 if world_spawn_ok else 0.0,
    )
    structure_score = _round_score(
        1.0 if _runtime_setup_ok(candidate) else 0.0,
        1.0 if bool(candidate.uxml_paths) else 0.0,
        1.0 if bool(candidate.script_paths) else 0.0,
    )
    candidate_score = round((0.35 * visible_score) + (0.40 * behavior_score) + (0.25 * structure_score), 4)
    return {
        **_candidate_summary(candidate),
        "candidate_score": candidate_score,
        "slot_count": slot_count,
        "slot_ratio": slot_ratio,
        "bottom_bar_ok": bottom_bar_ok,
        "drag_ghost_ok": drag_ghost_ok,
        "slot_highlight_ok": slot_highlight_ok,
        "item_count_ok": item_count_ok,
        "pointer_events_ok": pointer_events_ok,
        "drag_drop_ok": drag_drop_ok,
        "inventory_events_ok": inventory_events_ok,
        "world_spawn_ok": world_spawn_ok,
    }


def _score_lore_codex(candidate: SceneCandidate) -> dict[str, Any]:
    contract = VARIANT_SPECS["lore_codex"].output_contract
    visible = contract["visible_ui_requirements"]
    linked = candidate.linked_corpus
    ui_and_script = candidate.layout_corpus + "\n" + candidate.script_corpus

    texts = list(visible["required_text_values"])
    text_ratio = _count_present(texts, linked) / len(texts)
    sidebar_ok = _contains_any(ui_and_script, ("SidebarPanel", "sidebar-panel", "category-btn"))
    scroll_ok = _contains_any(ui_and_script, ("LoreScrollView", "<ui:ScrollView", "lore-scrollview"))
    pointer_move_ok = _contains_any(candidate.script_corpus, ("PointerMoveEvent", "OnPointerMove", "_mousePercent"))
    scanline_ok = _contains_any(candidate.script_corpus, ("DrawScanlines", "scanlineSpeed", "generateVisualContent"))
    parallax_ok = _contains_any(candidate.script_corpus, ("ParallaxDeep", "ParallaxMid", "translate = new Translate"))
    procedural_ok = _contains_any(candidate.script_corpus, ("DrawCyberGrid", "generateVisualContent", "MarkDirtyRepaint"))

    visible_score = _round_score(
        text_ratio,
        1.0 if sidebar_ok else 0.0,
        1.0 if scroll_ok else 0.0,
    )
    behavior_score = _round_score(
        1.0 if pointer_move_ok else 0.0,
        1.0 if scanline_ok else 0.0,
        1.0 if parallax_ok else 0.0,
        1.0 if procedural_ok else 0.0,
    )
    structure_score = _round_score(
        1.0 if _runtime_setup_ok(candidate) else 0.0,
        1.0 if bool(candidate.uxml_paths) else 0.0,
        1.0 if bool(candidate.script_paths) else 0.0,
    )
    candidate_score = round((0.35 * visible_score) + (0.40 * behavior_score) + (0.25 * structure_score), 4)
    return {
        **_candidate_summary(candidate),
        "candidate_score": candidate_score,
        "text_ratio": text_ratio,
        "sidebar_ok": sidebar_ok,
        "scroll_ok": scroll_ok,
        "pointer_move_ok": pointer_move_ok,
        "scanline_ok": scanline_ok,
        "parallax_ok": parallax_ok,
        "procedural_ok": procedural_ok,
    }


SCORERS = {
    "start_menu": _score_start_menu,
    "fps_hud": _score_fps_hud,
    "visual_novel": _score_visual_novel,
    "world_inventory": _score_world_inventory,
    "lore_codex": _score_lore_codex,
}


def _visible_ok(variant_name: str, scored: dict[str, Any]) -> bool:
    if variant_name == "start_menu":
        return (
            scored["header_ok"]
            and scored["load_button_ok"]
            and scored["labels_ratio"] == 1.0
            and scored["scroll_ok"]
            and scored["icon_ok"]
            and scored["selection_ok"]
        )
    if variant_name == "fps_hud":
        return scored["named_ratio"] == 1.0 and scored["numbers_ok"] and scored["weapon_icons_ok"]
    if variant_name == "visual_novel":
        return (
            scored["named_ratio"] == 1.0
            and scored["text_ratio"] == 1.0
            and scored["background_and_portrait_ok"]
            and scored["save_slot_template_ok"]
            and scored["confirm_control_ok"]
        )
    if variant_name == "world_inventory":
        return (
            scored["slot_ratio"] == 1.0
            and scored["bottom_bar_ok"]
            and scored["drag_ghost_ok"]
            and scored["slot_highlight_ok"]
            and scored["item_count_ok"]
        )
    return scored["text_ratio"] == 1.0 and scored["sidebar_ok"] and scored["scroll_ok"]


def _behavior_ok(variant_name: str, scored: dict[str, Any]) -> bool:
    behavior_keys = {
        "start_menu": ("disabled_ok", "additive_flow_ok"),
        "fps_hud": ("buy_menu_toggle_ok", "hud_update_ok", "minimap_ok", "crosshair_ok"),
        "visual_novel": ("dialogue_flow_ok", "options_ok", "save_load_ok"),
        "world_inventory": ("drag_drop_ok", "inventory_events_ok", "world_spawn_ok"),
        "lore_codex": (),
    }[variant_name]
    return all(bool(scored[key]) for key in behavior_keys)


def evaluate_unitypackage_bytes(
    blob: bytes,
    *,
    variant_name: str,
    package_name: str = "migrated_output.unitypackage",
) -> dict[str, Any]:
    if variant_name not in VARIANT_SPECS:
        raise ValueError(f"unknown variant_name: {variant_name}")

    contract = VARIANT_SPECS[variant_name].output_contract
    try:
        package = parse_unitypackage_bytes(blob)
    except tarfile.TarError as exc:
        return {
            "final_score": 0.0,
            "hard_fail": "unreadable_unitypackage",
            "details": {"error": str(exc), "package_name": package_name, "variant_name": variant_name},
        }

    has_scene = bool(package.scene_paths)
    has_uitk_evidence = _contains_any(package.corpus, UITK_MARKERS)
    candidates = _scene_candidates(package)

    if contract["structural_requirements"]["requires_unity_scene"] and not has_scene:
        return {
            "final_score": 0.0,
            "hard_fail": "missing_unity_scene",
            "details": {"scene_paths": package.scene_paths, "path_count": len(package.pathnames)},
        }
    if contract["structural_requirements"]["requires_ui_toolkit_migration"] and not has_uitk_evidence:
        return {
            "final_score": 0.0,
            "hard_fail": "missing_ui_toolkit_evidence",
            "details": {"scene_paths": package.scene_paths, "path_count": len(package.pathnames)},
        }
    if not candidates:
        return {
            "final_score": 0.0,
            "hard_fail": "missing_scene_linked_ui_toolkit_scene",
            "details": {"scene_paths": package.scene_paths, "path_count": len(package.pathnames)},
        }

    scored_candidates = _score_candidates(candidates, SCORERS[variant_name])
    visible_requirements = contract["visible_ui_requirements"]

    passing_candidates: list[tuple[SceneCandidate, dict[str, Any]]] = []
    runtime_ready_candidates: list[tuple[SceneCandidate, dict[str, Any]]] = []
    visible_ready_candidates: list[tuple[SceneCandidate, dict[str, Any]]] = []
    for candidate, scored in scored_candidates:
        runtime_ok = not contract["structural_requirements"]["requires_runtime_ui_toolkit_setup"] or _runtime_setup_ok(candidate)
        if not runtime_ok:
            continue
        runtime_ready_candidates.append((candidate, scored))
        if not _visible_ok(variant_name, scored):
            continue
        visible_ready_candidates.append((candidate, scored))
        if _behavior_ok(variant_name, scored):
            passing_candidates.append((candidate, scored))

    if passing_candidates:
        candidate, scored = max(passing_candidates, key=lambda item: item[1]["candidate_score"])
    elif visible_ready_candidates:
        _, scored = max(visible_ready_candidates, key=lambda item: item[1]["candidate_score"])
        return {
            "final_score": 0.0,
            "hard_fail": "missing_required_behavioral_evidence",
            "details": {"variant_name": variant_name, **scored},
        }
    elif runtime_ready_candidates:
        _, scored = max(runtime_ready_candidates, key=lambda item: item[1]["candidate_score"])
        return {
            "final_score": 0.0,
            "hard_fail": "missing_required_visible_ui",
            "details": {"variant_name": variant_name, **scored},
        }
    else:
        _, scored = max(scored_candidates, key=lambda item: item[1]["candidate_score"])
        return {
            "final_score": 0.0,
            "hard_fail": "missing_runtime_ui_toolkit_setup",
            "details": scored,
        }

    return {
        "final_score": float(scored["candidate_score"]),
        "hard_fail": None,
        "details": {
            "variant_name": variant_name,
            "path_count": len(package.pathnames),
            "scene_paths": package.scene_paths,
            "has_uitk_evidence": has_uitk_evidence,
            "candidate_scene_count": len(candidates),
            "best_candidate": scored,
            "visible_ui_requirements": visible_requirements,
        },
    }


def evaluate_unitypackage_file(package_path: Path, *, variant_name: str) -> dict[str, Any]:
    return evaluate_unitypackage_bytes(
        package_path.read_bytes(),
        variant_name=variant_name,
        package_name=package_path.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_path", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANT_SPECS), required=True)
    args = parser.parse_args()
    result = evaluate_unitypackage_file(args.package_path, variant_name=args.variant)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
