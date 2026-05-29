from __future__ import annotations

from dataclasses import dataclass


VISIBLE_OUTPUT_NAME = "migrated_output.unitypackage"
# Preferred Unity 6 LTS versions in fallback order. 6000.3.13f1 was originally
# pinned; 6000.4.3f1 is the approved rotation when 6000.3.13f1 is unavailable.
# Keep this in sync with build_stage1_package.py's UNITY_VERSIONS.
UNITY_VERSIONS = ("6000.3.13f1", "6000.4.3f1")
UNITY_EXE = rf"C:\Program Files\Unity\Hub\Editor\{UNITY_VERSIONS[0]}\Editor\Unity.exe"
VARIANT_ORDER = [
    "start_menu",
    "fps_hud",
    "visual_novel",
    "world_inventory",
    "lore_codex",
]


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    label: str
    source_scene: str
    required_visible_behavior: tuple[str, ...]
    required_runtime_behavior: tuple[str, ...]
    dependency_note: str | None
    output_contract: dict[str, object]


VARIANT_SPECS = {
    "start_menu": VariantSpec(
        variant_name="start_menu",
        label="Start Menu",
        source_scene="Assets/Datapoint_StartMenu/UGUI_src/StartMenu_UGUI.unity",
        required_visible_behavior=(
            "Keep the header text `Select Minigame Scene`.",
            "Keep one scrollable scene-card area.",
            "Keep the six selectable cards for FPS, RPG, Visual Novel, Strategy, D&D RPG, and ARPG, with the documented scene targets.",
            "Keep an icon element on each card.",
            "Keep a visible selected/highlighted state for the chosen card.",
            "Keep `Load Scene` disabled until a card is selected.",
        ),
        required_runtime_behavior=(
            "Preserve additive scene loading through `SceneController.Instance.LoadScene(...)`.",
        ),
        dependency_note=None,
        output_contract={
            "variant_name": "start_menu",
            "output_filename": VISIBLE_OUTPUT_NAME,
            "source_scene": "Assets/Datapoint_StartMenu/UGUI_src/StartMenu_UGUI.unity",
            "structural_requirements": {
                "requires_unity_scene": True,
                "requires_ui_toolkit_migration": True,
                "requires_runtime_ui_toolkit_setup": True,
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
                "scene_targets": {
                    "An FPS Game": "_FPS",
                    "An RPG Game": "_RPG",
                    "Visual Novel Game": "_VisualNovel",
                    "Strategy Game": "_Strategy",
                    "D&D RPG Game": "_D&DRPG",
                    "ARPG Game": "_ARPG",
                },
                "requires_scrollable_card_layout": True,
                "requires_icon_element_per_card": True,
                "requires_selected_highlight_state": True,
            },
            "behavioral_requirements": {
                "load_button_disabled_until_selection": True,
                "single_selection_behavior": True,
                "preserve_additive_scene_loading_flow": True,
            },
        },
    ),
    "fps_hud": VariantSpec(
        variant_name="fps_hud",
        label="FPS HUD",
        source_scene="Assets/Datapoint_FPS/UGUI_src/FPS_UGUI.unity",
        required_visible_behavior=(
            "Keep the HUD group with minimap, crosshair, health, armor, and weapon-panel surfaces.",
            "Keep the purchase menu with pistol and rifle sections.",
            "Keep visible weapon icon slots for both the shop and HUD surfaces.",
        ),
        required_runtime_behavior=(
            "Preserve the buy-menu toggle behavior.",
            "Preserve HUD state updates driven by the existing gameplay scripts.",
        ),
        dependency_note=None,
        output_contract={
            "variant_name": "fps_hud",
            "output_filename": VISIBLE_OUTPUT_NAME,
            "source_scene": "Assets/Datapoint_FPS/UGUI_src/FPS_UGUI.unity",
            "structural_requirements": {
                "requires_unity_scene": True,
                "requires_ui_toolkit_migration": True,
                "requires_runtime_ui_toolkit_setup": True,
            },
            "visible_ui_requirements": {
                "required_named_elements": [
                    "MinimapContainer",
                    "Crosshair",
                    "HealthText",
                    "ArmorText",
                    "PurchaseMenu_Group",
                    "PistolPanel",
                    "RiflePanel",
                ],
                "required_text_values": ["100", "100"],
                "requires_weapon_icon_surfaces": True,
            },
            "behavioral_requirements": {
                "preserve_buy_menu_toggle": True,
                "preserve_hud_state_updates": True,
            },
        },
    ),
    "visual_novel": VariantSpec(
        variant_name="visual_novel",
        label="Visual Novel",
        source_scene="Assets/Datapoint_VisualNovel/UGUI_src/VisualNovel_UGUI.unity",
        required_visible_behavior=(
            "Keep the background image and character portrait.",
            "Keep the dialogue panel, speaker name, and dialogue text.",
            "Keep the system button row with `Save`, `Load`, `Q.Save`, and `Q.Load`.",
            "Keep the options container for choice buttons.",
            "Keep the save/load archive panel with slot cards plus close and confirm controls.",
        ),
        required_runtime_behavior=(
            "Preserve dialogue advance and options rendering.",
            "Preserve save/load slot interactions.",
            "Preserve the existing backend event wiring.",
        ),
        dependency_note=(
            "The visible input notes that the project depends on "
            "`https://github.com/Blind-Guess-Senior/ArtifactDialoguer.git` via UPM. "
            "If the imported Unity project is missing that package, add it before validating the migrated scene."
        ),
        output_contract={
            "variant_name": "visual_novel",
            "output_filename": VISIBLE_OUTPUT_NAME,
            "source_scene": "Assets/Datapoint_VisualNovel/UGUI_src/VisualNovel_UGUI.unity",
            "structural_requirements": {
                "requires_unity_scene": True,
                "requires_ui_toolkit_migration": True,
                "requires_runtime_ui_toolkit_setup": True,
            },
            "visible_ui_requirements": {
                "required_named_elements": [
                    "BG",
                    "Character_Portrait",
                    "Dialogue_Panel",
                    "Name_Text",
                    "DialogueText",
                    "Options_Container",
                    "SLMenu_Panel",
                    "Slots_Container",
                    "Btn_Close",
                    "Btn_Confirm",
                ],
                "required_text_values": [
                    "Save",
                    "Load",
                    "Q.Save",
                    "Q.Load",
                    "Memory Archive",
                    "CLOSE",
                ],
                "requires_save_slot_template": True,
                "requires_background_and_portrait": True,
                "requires_archive_confirm_control": True,
            },
            "behavioral_requirements": {
                "preserve_dialogue_advance": True,
                "preserve_options_rendering": True,
                "preserve_save_load_interactions": True,
            },
        },
    ),
    "world_inventory": VariantSpec(
        variant_name="world_inventory",
        label="2D World Inventory",
        source_scene="Assets/Datapoint_Inventory/UGUI_src/Inventory_UGUI.unity",
        required_visible_behavior=(
            "Keep the six-slot bottom inventory bar.",
            "Keep item sprite and count rendering in the slots.",
            "Keep a drag ghost that follows the pointer during item drag.",
            "Keep highlighted target-slot state during drag/drop.",
        ),
        required_runtime_behavior=(
            "Preserve world-item pickup and drag/drop behavior.",
            "Preserve slot updates through `InventoryDataManager`.",
            "Preserve the world drop/spawn flow.",
        ),
        dependency_note=None,
        output_contract={
            "variant_name": "world_inventory",
            "output_filename": VISIBLE_OUTPUT_NAME,
            "source_scene": "Assets/Datapoint_Inventory/UGUI_src/Inventory_UGUI.unity",
            "structural_requirements": {
                "requires_unity_scene": True,
                "requires_ui_toolkit_migration": True,
                "requires_runtime_ui_toolkit_setup": True,
            },
            "visible_ui_requirements": {
                "minimum_slot_count": 6,
                "requires_bottom_bar_layout": True,
                "requires_drag_ghost": True,
                "requires_slot_highlight_state": True,
                "requires_item_count_rendering": True,
            },
            "behavioral_requirements": {
                "preserve_drag_drop": True,
                "preserve_inventory_data_events": True,
                "preserve_world_spawn_flow": True,
                "requires_pointer_events": True,
            },
        },
    ),
    "lore_codex": VariantSpec(
        variant_name="lore_codex",
        label="Lore Codex",
        source_scene="Assets/Datapoint_Codex/UGUI_src/Codex_UGUI.unity",
        required_visible_behavior=(
            "Keep the codex main window and a stylized futuristic visual treatment.",
            "Keep the database-style sidebar with category buttons.",
            "Keep the scrollable lore content area.",
        ),
        required_runtime_behavior=(
            "Preserve parallax background layers, scanlines, and procedural grid/glow intent.",
            "Preserve pointer-driven interaction for the animated surface.",
            "Avoid brute-force DOM bloat when recreating the visual effects.",
        ),
        dependency_note=None,
        output_contract={
            "variant_name": "lore_codex",
            "output_filename": VISIBLE_OUTPUT_NAME,
            "source_scene": "Assets/Datapoint_Codex/UGUI_src/Codex_UGUI.unity",
            "structural_requirements": {
                "requires_unity_scene": True,
                "requires_ui_toolkit_migration": True,
                "requires_runtime_ui_toolkit_setup": True,
            },
            "visible_ui_requirements": {
                "required_text_values": ["DATABASE", "HISTORY"],
                "requires_scroll_view": True,
                "requires_sidebar_and_main_panel": True,
            },
            "behavioral_requirements": {
                "requires_pointer_move_handling": True,
                "requires_procedural_runtime_drawing": True,
                "requires_parallax_effect": True,
                "requires_scanline_effect": True,
            },
        },
    ),
}
