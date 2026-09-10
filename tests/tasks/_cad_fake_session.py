"""Fake DesktopSession for the CAD engineering task tests: text files in a dict, directory listings by count."""
from __future__ import annotations


class FakeSession:
    def __init__(self, files: dict[str, str], dir_counts: dict[str, int] | None = None):
        self.files = dict(files); self.dir_counts = dir_counts or {}; self.commands: list[str] = []

    async def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def file_exists(self, path: str) -> bool:
        return path in self.files

    async def directory_exists(self, path: str) -> bool:
        return path in self.dir_counts

    async def run_command(self, cmd: str, check: bool = True):
        self.commands.append(cmd)
        for d, n in self.dir_counts.items():
            if f"ls {d!r} | wc -l" in cmd or f"ls '{d}' | wc -l" in cmd:
                return {"return_code": 0, "stdout": f"{n}\n", "stderr": ""}
        return {"return_code": 0, "stdout": "", "stderr": ""}
