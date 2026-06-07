#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

GUI_DOCKER_ENV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GUI_DOCKER_ENV))

from lib_results_paths import IMAGES_ZIP_NAME, IMAGE_SUFFIXES, list_result_run_dirs, results_root  # noqa: E402


def iter_image_files(run_dir: Path) -> list[Path]:
    images = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.name != IMAGES_ZIP_NAME
    ]
    images.sort(key=lambda path: path.relative_to(run_dir).as_posix())
    return images


def zip_run_dir(run_dir: Path, *, dry_run: bool = False, remove_images: bool = True) -> dict:
    images = iter_image_files(run_dir)
    zip_path = run_dir / IMAGES_ZIP_NAME
    if not images:
        return {
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "image_count": 0,
            "skipped": True,
        }
    if dry_run:
        return {
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "image_count": len(images),
            "dry_run": True,
        }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in images:
            archive.write(image_path, arcname=image_path.relative_to(run_dir).as_posix())

    removed = 0
    if remove_images:
        for image_path in images:
            image_path.unlink()
            removed += 1

    return {
        "run_dir": str(run_dir),
        "zip_path": str(zip_path),
        "image_count": len(images),
        "removed_images": removed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zip PNG/JPG screenshots in result run folders.")
    parser.add_argument("run_dirs", nargs="*", help="Specific results/results_* folders")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Process every results/results_* folder")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Create images.zip but do not delete the original image files",
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
        zip_run_dir(run_dir, dry_run=args.dry_run, remove_images=not args.keep_images)
        for run_dir in resolve_run_dirs(args)
    ]
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
