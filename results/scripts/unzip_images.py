#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

GUI_DOCKER_ENV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GUI_DOCKER_ENV))

from lib_results_paths import IMAGES_ZIP_NAME, list_result_run_dirs, results_root  # noqa: E402


def unzip_run_dir(run_dir: Path, *, dry_run: bool = False, remove_zip: bool = False) -> dict:
    zip_path = run_dir / IMAGES_ZIP_NAME
    if not zip_path.is_file():
        return {
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "skipped": True,
            "reason": "missing zip",
        }
    if dry_run:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
        return {
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "file_count": len(members),
            "dry_run": True,
        }

    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            target = run_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as handle:
                handle.write(source.read())
            extracted += 1

    if remove_zip:
        zip_path.unlink()

    return {
        "run_dir": str(run_dir),
        "zip_path": str(zip_path),
        "extracted_files": extracted,
        "removed_zip": remove_zip,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore PNG/JPG screenshots from images.zip.")
    parser.add_argument("run_dirs", nargs="*", help="Specific results/results_* folders")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Process every results/results_* folder")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--remove-zip",
        action="store_true",
        help="Delete images.zip after successful extraction",
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
        unzip_run_dir(run_dir, dry_run=args.dry_run, remove_zip=args.remove_zip)
        for run_dir in resolve_run_dirs(args)
    ]
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
