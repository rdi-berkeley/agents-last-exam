import asyncio
import json
from pathlib import Path

from sgfmill import sgf

from tasks.computing_math.go_game_reconstruction_1.main import GoGameReconstructionConfig
from tasks.visual_media.uv_reproduction.main import _resolve_candidate_mtl


class _Session:
    def __init__(self, files: set[str], contents: dict[str, str]):
        self.files = files
        self.contents = contents

    async def file_exists(self, path: str) -> bool:
        return path in self.files

    async def read_file(self, path: str) -> str:
        return self.contents[path]


def test_uv_evaluator_follows_obj_mtl_reference():
    obj_path = r"C:\Users\User\output\singing.obj"
    declared_mtl = r"C:\Users\User\output\material.mtl"
    referenced_mtl = r"C:\Users\User\output\singing.mtl"
    session = _Session(
        {obj_path, referenced_mtl},
        {obj_path: "mtllib singing.mtl\nvt 0.0 0.0\n"},
    )

    resolved = asyncio.run(_resolve_candidate_mtl(session, obj_path, declared_mtl))

    assert resolved == referenced_mtl


def test_go_prompt_matches_verified_reference_orientation():
    prompt = GoGameReconstructionConfig().task_description
    task_card = json.loads(
        Path(__file__).parents[2]
        .joinpath("tasks/computing_math/go_game_reconstruction_1/task_card.json")
        .read_text()
    )["taskPrompt"]

    opening = sgf.Sgf_game.from_bytes(b"(;SZ[19];B[qd];W[pp];B[cd];W[cp];B[ec])")
    columns = "ABCDEFGHJKLMNOPQRST"
    for text in (prompt, task_card):
        for move_number, node in enumerate(opening.get_main_sequence()[1:], start=1):
            color, (row, column) = node.get_move()
            coordinate = f"{columns[column]}{row + 1}"
            assert f"Move {move_number}: `{color.upper()} at {coordinate}`" in text
        assert "Move 1: `B at R4`" not in text
        assert "Move 2: `W at Q16`" not in text
        assert "Move 5: `B at E3`" not in text
