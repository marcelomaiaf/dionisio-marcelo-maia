from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agents import Runner

from dionisio_agent.config import Settings
from dionisio_agent.factory import create_runtime

CASES_PATH = Path(__file__).with_name("cases.json")


def main() -> None:
    parser = argparse.ArgumentParser(prog="dionisio-evals")
    parser.add_argument("--live", action="store_true", help="Run cases against the configured agent.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    if not args.live:
        for case in cases:
            print(f"[checklist] {case['id']}: {case['expected_behavior']}")
        return

    settings = Settings.from_env()
    agent, _runtime = await create_runtime(settings)
    results = []
    for case in cases:
        result = await Runner.run(agent, case["input"])
        results.append(
            {
                "id": case["id"],
                "input": case["input"],
                "output": result.final_output,
                "expected_behavior": case["expected_behavior"],
                "must_not": case["must_not"],
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False, indent=2))

    output_path = Path("evals/results.latest.json")
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

