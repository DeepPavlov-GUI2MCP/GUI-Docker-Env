#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GUI_DOCKER_ENV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GUI_DOCKER_ENV))

from lib_results_archive import zip_run_dir  # noqa: E402
from lib_results_paths import list_result_run_dirs, results_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive an entire results/results_* folder into run.zip for Hub upload.",
    )
    parser.add_argument("run_dirs", nargs="*", help="Specific results/results_* folders")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Process every results/results_* folder")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Create run.zip but keep the original loose files",
    )
    return parser.parse_args()


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dirs:
        resolved: list[Path] = []
        for item in args.run_dirs:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            resolved.append(path.resolve())
        return resolved
    if args.all:
        root = args.results_root.resolve() if args.results_root else results_root()
        return list_result_run_dirs(root)
    raise SystemExit("Pass run_dirs or use --all")


def main() -> int:
    args = parse_args()
    summaries = [
        zip_run_dir(run_dir, dry_run=args.dry_run, remove_files=not args.keep_files)
        for run_dir in resolve_run_dirs(args)
    ]
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
