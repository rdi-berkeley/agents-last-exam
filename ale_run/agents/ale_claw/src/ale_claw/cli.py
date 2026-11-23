"""Command-line entry point for one ALE-Claw session."""

import argparse
import asyncio
import json
import sys
import tomllib
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

from .config import AleClawConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a task on an existing CUA sandbox.")
    parser.add_argument("prompt")
    parser.add_argument("--config", type=Path, help="TOML file with sandbox settings and [agent]")
    parser.add_argument("--model", help="Override agent.model")
    parser.add_argument("--sandbox-url", help="CUA endpoint (default: http://127.0.0.1:8000)")
    parser.add_argument(
        "--os-type", choices=("linux", "windows"), help="Sandbox OS (default: linux)"
    )
    parser.add_argument("--output-dir", type=Path, help="Local directory for session artifacts")
    args = parser.parse_args()

    try:
        settings = {}
        if args.config:
            with args.config.open("rb") as stream:
                settings = tomllib.load(stream)
        agent = settings.pop("agent", {})
        if not isinstance(agent, dict):
            raise TypeError("[agent] must be a table")
        if args.model is not None:
            agent["model"] = args.model
        for key in ("sandbox_url", "os_type", "output_dir"):
            if (value := getattr(args, key)) is not None:
                settings[key] = value
        unknown = settings.keys() - {"sandbox_url", "os_type", "output_dir"}
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        with redirect_stdout(sys.stderr):
            from .runner import run

            result = asyncio.run(run(args.prompt, config=AleClawConfig(**agent), **settings))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"ale-claw: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.status == "completed" else 1
