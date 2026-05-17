#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from typing import Any

from common import (
    DEFAULT_DESKTOP_SERVER_URL,
    DEFAULT_TOKEN,
    GeometryBounds,
    GeometryThreshold,
    LiveEmulatorError,
    ScreenSize,
    WindowGeometry,
    apply_window_geometry,
    clamp_window_position,
    clamp_window_size,
    compute_coordinate_limits,
    fetch_screen_size,
    get_emulator,
    query_window_geometry,
    resolve_geometry_threshold,
    write_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move, resize, or randomize a live emulator window.")
    parser.add_argument("--server-url", default=DEFAULT_DESKTOP_SERVER_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--format", choices=["json", "text"], default="json")

    subparsers = parser.add_subparsers(dest="action", required=True)

    bounds_parser = subparsers.add_parser("bounds", help="Show resolved threshold and safe geometry bounds.")
    add_common_args(bounds_parser)
    bounds_parser.add_argument("--width", type=int, help="Candidate width. Defaults to the current window width or threshold minimum.")
    bounds_parser.add_argument("--height", type=int, help="Candidate height. Defaults to the current window height or threshold minimum.")
    bounds_parser.add_argument("--allow-below-min", action="store_true")

    move_parser = subparsers.add_parser("move", help="Move the active or named window.")
    add_common_args(move_parser)
    move_parser.add_argument("--x", type=int, required=True)
    move_parser.add_argument("--y", type=int, required=True)
    move_parser.add_argument("--allow-offscreen", action="store_true")

    resize_parser = subparsers.add_parser("resize", help="Resize the active or named window.")
    add_common_args(resize_parser)
    resize_parser.add_argument("--width", type=int, required=True)
    resize_parser.add_argument("--height", type=int, required=True)
    resize_parser.add_argument("--allow-below-min", action="store_true")
    resize_parser.add_argument("--allow-offscreen", action="store_true")

    randomize_parser = subparsers.add_parser("randomize", help="Randomize size and position within safe bounds.")
    add_common_args(randomize_parser)
    randomize_parser.add_argument("--seed", type=int, help="Optional RNG seed for reproducibility.")

    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--emulator-id")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--task", help="Task ref like thunderbird/<task_id> to select task-specific thresholds.")
    parser.add_argument("--app", help="App name like thunderbird to select app-level thresholds.")
    parser.add_argument("--window-name", help="Window title or class to target. Defaults to the active window.")
    parser.add_argument("--by-class", action="store_true", help="Interpret --window-name as WM_CLASS instead of title.")
    parser.add_argument("--timeout", type=int, default=60)


def main() -> int:
    args = parse_args()
    try:
        server_port = resolve_server_port(args)
        threshold = resolve_geometry_threshold(task=args.task, app=args.app)
        screen = fetch_screen_size(server_port)

        if args.action == "bounds":
            payload = handle_bounds(args, server_port, screen, threshold)
        elif args.action == "move":
            payload = handle_move(args, server_port, screen, threshold)
        elif args.action == "resize":
            payload = handle_resize(args, server_port, screen, threshold)
        elif args.action == "randomize":
            payload = handle_randomize(args, server_port, screen, threshold)
        else:
            raise LiveEmulatorError(f"Unsupported action: {args.action}")

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
    if args.by_class and not args.window_name:
        raise LiveEmulatorError("--by-class requires --window-name.")
    if args.server_port:
        return args.server_port
    emulator = get_emulator(args.server_url, args.emulator_id, args.token)
    return emulator.server_port


def get_target_window(args: argparse.Namespace, server_port: int) -> WindowGeometry:
    return query_window_geometry(
        server_port,
        window_name=args.window_name,
        by_class=args.by_class,
        timeout=args.timeout,
    )


def target_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "class" if args.window_name and args.by_class else "title" if args.window_name else "active",
        "value": args.window_name,
    }


def handle_bounds(
    args: argparse.Namespace,
    server_port: int,
    screen: ScreenSize,
    threshold: GeometryThreshold,
) -> dict[str, Any]:
    current_window: WindowGeometry | None = None
    try:
        current_window = get_target_window(args, server_port)
    except LiveEmulatorError:
        if args.width is None or args.height is None:
            current_window = None

    requested_width = args.width if args.width is not None else current_window.width if current_window else threshold.min_width
    requested_height = args.height if args.height is not None else current_window.height if current_window else threshold.min_height
    applied_width, applied_height, size_clamped = clamp_window_size(
        requested_width,
        requested_height,
        screen,
        threshold,
        allow_below_min=args.allow_below_min,
    )
    bounds = compute_coordinate_limits(screen, threshold, width=applied_width, height=applied_height)
    return {
        "status": "ok",
        "server_port": server_port,
        "screen": screen.to_dict(),
        "target": target_summary(args),
        "current_window": current_window.to_dict() if current_window else None,
        "threshold": threshold.to_dict(),
        "requested_size": {"width": requested_width, "height": requested_height},
        "applied_size": {"width": applied_width, "height": applied_height},
        "size_clamped": size_clamped,
        "bounds": bounds.to_dict(),
    }


def handle_move(
    args: argparse.Namespace,
    server_port: int,
    screen: ScreenSize,
    threshold: GeometryThreshold,
) -> dict[str, Any]:
    current = get_target_window(args, server_port)
    bounds = compute_coordinate_limits(screen, threshold, width=current.width, height=current.height)
    applied_x, applied_y, position_clamped = clamp_window_position(
        args.x,
        args.y,
        bounds,
        allow_offscreen=args.allow_offscreen,
    )
    applied = apply_window_geometry(
        server_port,
        current,
        x=applied_x,
        y=applied_y,
        width=current.width,
        height=current.height,
        timeout=args.timeout,
    )
    return {
        "status": "ok",
        "server_port": server_port,
        "screen": screen.to_dict(),
        "target": target_summary(args),
        "threshold": threshold.to_dict(),
        "requested_position": {"x": args.x, "y": args.y},
        "applied_position": {"x": applied_x, "y": applied_y},
        "position_clamped": position_clamped,
        "bounds": bounds.to_dict(),
        "before": current.to_dict(),
        "after": applied.to_dict(),
    }


def handle_resize(
    args: argparse.Namespace,
    server_port: int,
    screen: ScreenSize,
    threshold: GeometryThreshold,
) -> dict[str, Any]:
    current = get_target_window(args, server_port)
    applied_width, applied_height, size_clamped = clamp_window_size(
        args.width,
        args.height,
        screen,
        threshold,
        allow_below_min=args.allow_below_min,
    )
    bounds = compute_coordinate_limits(screen, threshold, width=applied_width, height=applied_height)
    applied_x, applied_y, position_clamped = clamp_window_position(
        current.x,
        current.y,
        bounds,
        allow_offscreen=args.allow_offscreen,
    )
    applied = apply_window_geometry(
        server_port,
        current,
        x=applied_x,
        y=applied_y,
        width=applied_width,
        height=applied_height,
        timeout=args.timeout,
    )
    return {
        "status": "ok",
        "server_port": server_port,
        "screen": screen.to_dict(),
        "target": target_summary(args),
        "threshold": threshold.to_dict(),
        "requested_size": {"width": args.width, "height": args.height},
        "applied_size": {"width": applied_width, "height": applied_height},
        "size_clamped": size_clamped,
        "position_clamped": position_clamped,
        "bounds": bounds.to_dict(),
        "before": current.to_dict(),
        "after": applied.to_dict(),
    }


def handle_randomize(
    args: argparse.Namespace,
    server_port: int,
    screen: ScreenSize,
    threshold: GeometryThreshold,
) -> dict[str, Any]:
    current = get_target_window(args, server_port)
    rng = random.Random(args.seed)
    initial_bounds = compute_coordinate_limits(
        screen,
        threshold,
        width=threshold.min_width,
        height=threshold.min_height,
    )

    width = rng.randint(initial_bounds.min_width, initial_bounds.max_width)
    height = rng.randint(initial_bounds.min_height, initial_bounds.max_height)
    width, height, size_clamped = clamp_window_size(width, height, screen, threshold, allow_below_min=False)
    bounds = compute_coordinate_limits(screen, threshold, width=width, height=height)
    x = rng.randint(bounds.min_x, bounds.max_x)
    y = rng.randint(bounds.min_y, bounds.max_y)
    applied = apply_window_geometry(
        server_port,
        current,
        x=x,
        y=y,
        width=width,
        height=height,
        timeout=args.timeout,
    )
    return {
        "status": "ok",
        "server_port": server_port,
        "screen": screen.to_dict(),
        "target": target_summary(args),
        "threshold": threshold.to_dict(),
        "seed": args.seed,
        "randomized_size": {"width": width, "height": height},
        "randomized_position": {"x": x, "y": y},
        "size_clamped": size_clamped,
        "bounds": bounds.to_dict(),
        "before": current.to_dict(),
        "after": applied.to_dict(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
