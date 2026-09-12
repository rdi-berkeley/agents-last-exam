"""One-shot subprocess driver for the synchronous DeepSeek Harness SDK."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _emit(record: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--session-root", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one SDK turn and stream every notification as JSONL."""
    args = _parser().parse_args(argv)
    try:
        from deepseek_harness import DeepSeekHarness

        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

        def on_notification(notification: Any) -> None:
            _emit(
                {
                    "type": "notification",
                    "method": notification.method,
                    "params": notification.payload,
                }
            )

        with DeepSeekHarness(
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            cwd=args.cwd,
            runtime_cwd=args.cwd,
            session_root=args.session_root,
            shutdown_timeout_seconds=2.0,
        ) as harness:
            result = harness.run(prompt, on_notification=on_notification)

        _emit(
            {
                "type": "result",
                "session_id": result.session_id,
                "final_response": result.final_response,
                "finish_reason": result.finish_reason,
            }
        )
        return 1 if result.finish_reason == "error" else 0
    except Exception as exc:  # noqa: BLE001 - subprocess boundary serializes failures
        _emit(
            {
                "type": "driver_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
