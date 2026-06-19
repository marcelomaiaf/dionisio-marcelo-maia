from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from agents import MaxTurnsExceeded

from dionisio_agent.config import Settings
from dionisio_agent.agent_runner import (
    AGENT_MAX_TURNS_MESSAGE,
    result_text,
    run_agent_until_text,
)
from dionisio_agent.factory import create_runtime
from dionisio_agent.operation_catalog import OperationCatalog
from dionisio_agent.sessions import create_limited_sqlite_session


def main() -> None:
    parser = argparse.ArgumentParser(prog="dionisio-agent")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("prompt")

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--session-id", default="default")
    chat_parser.add_argument("--db-path", default=".dionisio_agent/sessions.sqlite")
    chat_parser.add_argument("--history-limit", type=int, default=None)

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
        try:
            result = await run_agent_until_text(
                agent,
                args.prompt,
                max_turns=settings.agent_max_turns,
            )
        except MaxTurnsExceeded:
            print(AGENT_MAX_TURNS_MESSAGE)
            return
        print(result_text(result))
        return

    if args.command == "chat":
        agent, _runtime = await create_runtime(settings)
        Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
        session = create_limited_sqlite_session(
            args.session_id,
            db_path=args.db_path,
            history_limit=(
                args.history_limit
                if args.history_limit is not None
                else settings.web_chat_session_history_limit
            ),
        )
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
            try:
                result = await run_agent_until_text(
                    agent,
                    prompt,
                    session=session,
                    max_turns=settings.agent_max_turns,
                )
            except MaxTurnsExceeded:
                print(f"\nagente> {AGENT_MAX_TURNS_MESSAGE}")
                continue
            print(f"\nagente> {result_text(result)}")

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
