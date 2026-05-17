#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_DESKTOP_SERVER_URL,
    DEFAULT_TOKEN,
    LiveEmulatorError,
    fetch_accessibility_tree,
    fetch_screenshot,
    get_emulator,
    timestamp_name,
    write_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a screenshot or accessibility tree from a live emulator.")
    parser.add_argument("--server-url", default=DEFAULT_DESKTOP_SERVER_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--emulator-id")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--screenshot", action="store_true", help="Capture a screenshot. Default when no mode is selected.")
    parser.add_argument("--accessibility", action="store_true", help="Capture the accessibility tree.")
    parser.add_argument("--output", help="Screenshot output path.")
    parser.add_argument("--accessibility-output", help="Accessibility output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        server_port = resolve_server_port(args)
        want_screenshot = args.screenshot or not args.accessibility
        payload: dict[str, object] = {"status": "ok", "server_port": server_port}

        if want_screenshot:
            screenshot_bytes = fetch_screenshot(server_port)
            screenshot_path = Path(args.output or timestamp_name("live_emulator_screenshot", ".png")).resolve()
            screenshot_path.write_bytes(screenshot_bytes)
            payload["screenshot_path"] = str(screenshot_path)
            payload["screenshot_bytes"] = len(screenshot_bytes)

        if args.accessibility:
            tree = fetch_accessibility_tree(server_port)
            output_path = Path(args.accessibility_output or timestamp_name("live_emulator_accessibility", ".xml")).resolve()
            output_path.write_text(tree, encoding="utf-8")
            payload["accessibility_path"] = str(output_path)
            payload["accessibility_chars"] = len(tree)

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


if __name__ == "__main__":
    raise SystemExit(main())
