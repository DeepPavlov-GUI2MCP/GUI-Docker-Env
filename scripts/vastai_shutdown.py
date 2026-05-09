#!/usr/bin/env python3
import argparse
import json
import signal
import subprocess
import sys
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="dart-rollouter")
    parser.add_argument("--label", default="dart-rollouter")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required; pass it after --")
    return args


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def resolve_instance_id(host: str, label: str) -> int:
    run_checked(["ssh", host, "which vastai"])
    proc = run_checked(["ssh", host, "vastai show instances --raw"])
    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("vastai show instances --raw did not return a list")
    matches = [item for item in payload if isinstance(item, dict) and item.get("label") == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one instance with label {label!r}, found {len(matches)}")
    instance_id = matches[0].get("id")
    if not isinstance(instance_id, int):
        raise RuntimeError("resolved instance id is missing or invalid")
    return instance_id


def stop_instance(host: str, instance_id: int, dry_run: bool) -> None:
    command = ["ssh", host, f"vastai stop instance {instance_id}"]
    if dry_run:
        print("DRY_RUN", " ".join(command), flush=True)
        return
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.rstrip(), flush=True)
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"shutdown command failed with exit code {proc.returncode}")


def terminate_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def main() -> int:
    args = parse_args()
    instance_id = args.instance_id or resolve_instance_id(args.host, args.label)
    print(f"ROLL_OUTER_INSTANCE_ID={instance_id}", flush=True)

    child = subprocess.Popen(args.command)
    interrupted = False

    def handle_signal(signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True
        terminate_child(child)

    old_sigint = signal.signal(signal.SIGINT, handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    try:
        return_code = child.wait()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if child.poll() is None:
            terminate_child(child)
        try:
            stop_instance(args.host, instance_id, args.dry_run)
        except Exception as exc:
            print(f"ROLL_OUTER_SHUTDOWN_FAILED={exc}", file=sys.stderr, flush=True)
            return 1 if child.poll() in (None, 0) else child.returncode

    if interrupted:
        return 130
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
