from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from agents import Runner, SQLiteSession

from dionisio_agent.config import Settings
from dionisio_agent.factory import create_runtime
from dionisio_agent.operation_catalog import OperationCatalog


def main() -> None:
    parser = argparse.ArgumentParser(prog="dionisio-agent")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("prompt")

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--session-id", default="default")
    chat_parser.add_argument("--db-path", default=".dionisio_agent/sessions.sqlite")

    operations_parser = subparsers.add_parser("operations")
    operations_parser.add_argument("--query", default="")
    operations_parser.add_argument("--domain")
    operations_parser.add_argument("--destructive", action="store_true")
    operations_parser.add_argument("--limit", type=int, default=12)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if args.command == "ask":
        agent, _runtime = await create_runtime(settings)
        result = await Runner.run(agent, args.prompt)
        print(result.final_output)
        return

    if args.command == "chat":
        agent, _runtime = await create_runtime(settings)
        Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
        session = SQLiteSession(args.session_id, db_path=args.db_path)
        print(
            f"Dionisio chat started. session_id={args.session_id}. "
            "Type 'exit', 'quit' or Ctrl+C to leave."
        )
        while True:
            try:
                prompt = input("\nvoce> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if prompt.lower() in {"exit", "quit", "sair"}:
                return
            if not prompt:
                continue
            result = await Runner.run(agent, prompt, session=session)
            print(f"\nagente> {result.final_output}")

        return

    if args.command == "operations":
        catalog = await OperationCatalog.from_url(
            settings.dionisio_openapi_url,
            timeout_seconds=settings.request_timeout_seconds,
        )
        destructive: bool | None = True if args.destructive else None
        operations = catalog.search(
            args.query,
            domain=args.domain,
            destructive=destructive,
            limit=args.limit,
        )
        print(json.dumps([_operation_to_dict(op) for op in operations], indent=2, ensure_ascii=True))
        return


def _operation_to_dict(operation: Any) -> dict[str, Any]:
    return {
        "operation_id": operation.operation_id,
        "method": operation.method,
        "path": operation.path,
        "domain": operation.domain,
        "summary": operation.summary,
        "destructive": operation.destructive,
    }


if __name__ == "__main__":
    main()
