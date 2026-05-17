#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import (
    DEFAULT_DESKTOP_SERVER_URL,
    DEFAULT_TOKEN,
    LiveEmulatorError,
    build_python_command,
    execute_python,
    get_emulator,
    write_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Python or pyautogui code in a live emulator.")
    parser.add_argument("--server-url", default=DEFAULT_DESKTOP_SERVER_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--emulator-id")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--pyautogui", help="PyAutoGUI snippet, for example: pyautogui.click(500, 400)")
    parser.add_argument("--python", dest="python_code", help="Raw Python snippet to run as-is.")
    parser.add_argument("--timeout", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        server_port = resolve_server_port(args)
        wrap_pyautogui, code = resolve_code(args)
        payload = execute_python(
            server_port,
            build_python_command(code, wrap_pyautogui=wrap_pyautogui),
            timeout=args.timeout,
        )
        payload["server_port"] = server_port
        write_output(payload, args.format)
        return 0
    except LiveEmulatorError as exc:
        if args.format == "json":
            write_output({"status": "error", "error": str(exc)}, "json")
        else:
            print(f"error: {exc}")
        return 1


def resolve_server_port(args: argparse.Namespace) -> int:
    if bool(args.emulator_id) == bool(args.server_port):
        raise LiveEmulatorError("Provide exactly one of --emulator-id or --server-port.")
    if args.server_port:
        return args.server_port
    emulator = get_emulator(args.server_url, args.emulator_id, args.token)
    return emulator.server_port


def resolve_code(args: argparse.Namespace) -> tuple[bool, str]:
    if bool(args.pyautogui) == bool(args.python_code):
        raise LiveEmulatorError("Provide exactly one of --pyautogui or --python.")
    if args.pyautogui:
        return True, args.pyautogui
    return False, args.python_code


if __name__ == "__main__":
    raise SystemExit(main())
