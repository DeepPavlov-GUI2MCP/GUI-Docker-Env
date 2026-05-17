#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from common import (
    DEFAULT_DESKTOP_SERVER_URL,
    DEFAULT_TOKEN,
    LiveEmulatorError,
    get_emulator,
    list_emulators,
    resolve_task_config,
    seed_task,
    start_emulator,
    stop_emulator,
    write_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage live OSWorld emulator containers.")
    parser.add_argument("--server-url", default=DEFAULT_DESKTOP_SERVER_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--format", choices=["json", "text"], default="json")

    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="List running emulators.")
    list_parser.add_argument("--all-tokens", action="store_true", help="Do not filter by the current token.")

    start_parser = subparsers.add_parser("start", help="Start a fresh emulator, optionally seeded from a task.")
    add_task_args(start_parser)

    stop_parser = subparsers.add_parser("stop", help="Stop one emulator.")
    stop_parser.add_argument("--emulator-id", required=True)

    reset_parser = subparsers.add_parser("reset", help="Replace an emulator with a fresh one, optionally reseeded from a task.")
    reset_parser.add_argument("--emulator-id", required=True)
    add_task_args(reset_parser)

    return parser.parse_args()


def add_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", help="Task ref like chrome/<task_id> for immediate seeding.")
    parser.add_argument("--task-json", help="Direct path to a task json for immediate seeding.")
    parser.add_argument(
        "--enable-task-proxy",
        action="store_true",
        help="Honor task proxy setup when the task requests it.",
    )
    parser.add_argument(
        "--client-password",
        default="",
        help="Password passed to SetupController proxy setup when enabled.",
    )


def main() -> int:
    args = parse_args()
    try:
        if args.action == "list":
            items = list_emulators(args.server_url, args.token, filter_by_token=not args.all_tokens)
            write_output([item.to_dict() for item in items], args.format)
            return 0

        if args.action == "stop":
            payload = stop_emulator(args.server_url, args.emulator_id, args.token)
            write_output(payload, args.format)
            return 0

        if args.action == "start":
            payload = start_and_optionally_seed(args)
            write_output(payload, args.format)
            return 0

        if args.action == "reset":
            old_emulator = get_emulator(args.server_url, args.emulator_id, args.token)
            stop_emulator(args.server_url, args.emulator_id, args.token)
            payload = start_and_optionally_seed(args)
            payload["replaced_emulator_id"] = old_emulator.emulator_id
            write_output(payload, args.format)
            return 0
    except LiveEmulatorError as exc:
        if args.format == "json":
            write_output({"status": "error", "error": str(exc)}, "json")
        else:
            print(f"error: {exc}")
        return 1

    return 1


def start_and_optionally_seed(args: argparse.Namespace) -> dict[str, Any]:
    emulator = start_emulator(args.server_url, args.token)
    payload: dict[str, Any] = {
        "status": "ok",
        "emulator": emulator.to_dict(),
    }
    if args.task or args.task_json:
        task_config, task_path, domain = resolve_task_config(args.task, args.task_json)
        seed_result = seed_task(
            emulator,
            task_config,
            enable_task_proxy=args.enable_task_proxy,
            client_password=args.client_password,
        )
        payload["task"] = {
            "domain": domain,
            "path": str(task_path),
            **seed_result,
        }
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
