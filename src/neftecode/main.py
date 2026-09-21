from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from dateutil.parser import isoparse

from neftecode.demo.scenarios import DEMO_SCENARIOS, get_scenario
from neftecode.orchestration.orchestrator import Orchestrator


def _parse_ts(value: str) -> datetime:
    return isoparse(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neftecode", description="Neftecode multi-agent decision cycle")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one decision cycle")
    run.add_argument("--timestamp", type=str, default=None, help="ISO timestamp")
    run.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=sorted(DEMO_SCENARIOS),
        help="Demo scenario name (sets timestamp)",
    )
    run.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    sub.add_parser("scenarios", help="List demo scenarios")
    return parser


def main(argv: list[str] | None = None) -> None:
    # Output is Russian and carries arrows; a Windows console defaults to cp1251 and the
    # first print raises UnicodeEncodeError before anything is shown.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scenarios":
        for name, sc in DEMO_SCENARIOS.items():
            print(f"{name}: {sc.timestamp.isoformat()} — {sc.description}")
        return

    if args.command == "run":
        scenario = args.scenario
        if scenario:
            ts = get_scenario(scenario).timestamp
        elif args.timestamp:
            ts = _parse_ts(args.timestamp)
        else:
            ts = datetime(2024, 6, 1, 12, 0, 0)

        rec = Orchestrator().run_cycle(ts, scenario=scenario)
        payload = rec.model_dump(mode="json")
        if args.pretty:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
