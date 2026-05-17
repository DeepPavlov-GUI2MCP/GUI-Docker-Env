#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import (
    DEFAULT_DESKTOP_SERVER_URL,
    DEFAULT_TOKEN,
    LiveEmulatorError,
    EmulatorInfo,
    get_emulator,
    list_emulators,
    resolve_task_config,
    seed_task,
    write_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed or reseed an existing live emulator from an OSWorld task config.")
    parser.add_argument("--server-url", default=DEFAULT_DESKTOP_SERVER_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--emulator-id", help="Existing emulator id.")
    parser.add_argument("--latest", action="store_true", help="Use the newest emulator for the current token.")
    parser.add_argument("--task", help="Task ref like chrome/<task_id>.")
    parser.add_argument("--task-json", help="Direct path to a task json.")
    parser.add_argument("--enable-task-proxy", action="store_true")
    parser.add_argument("--client-password", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        emulator = resolve_target_emulator(args)
        task_config, task_path, domain = resolve_task_config(args.task, args.task_json)
        seed_result = seed_task(
            emulator,
            task_config,
            enable_task_proxy=args.enable_task_proxy,
            client_password=args.client_password,
        )
        payload = {
            "status": "ok",
            "emulator": emulator.to_dict(),
            "task": {
                "domain": domain,
                "path": str(task_path),
                **seed_result,
            },
        }
        write_output(payload, args.format)
        return 0
    except LiveEmulatorError as exc:
        if args.format == "json":
            write_output({"status": "error", "error": str(exc)}, "json")
        else:
            print(f"error: {exc}")
        return 1


def resolve_target_emulator(args: argparse.Namespace) -> EmulatorInfo:
    if bool(args.emulator_id) == bool(args.latest):
        raise LiveEmulatorError("Provide exactly one of --emulator-id or --latest.")
    if args.emulator_id:
        return get_emulator(args.server_url, args.emulator_id, args.token)
    emulators = list_emulators(args.server_url, args.token, filter_by_token=True)
    if not emulators:
        raise LiveEmulatorError("No emulators found for the current token.")
    emulators.sort(key=lambda item: item.start_time or 0, reverse=True)
    return emulators[0]


if __name__ == "__main__":
    raise SystemExit(main())
