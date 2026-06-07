#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "tony-pitchblack/dart-gui.GUI-Docker-Env.results"
DEFAULT_RESULTS_GLOB = "GUI-Docker-Env/results/results*"
GITATTRIBUTES_SOURCE = Path(__file__).resolve().parents[1] / "results" / ".gitattributes"
DEFAULT_IGNORE_PATTERNS = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*", help="Folders to upload")
    parser.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=[],
        help="Glob pattern relative to the workspace root",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching result folders and exit",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Dataset repo id",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Repo revision",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message to reuse for every uploaded folder",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Allow pattern passed to Hugging Face",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional ignore patterns passed to Hugging Face",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing folders instead of failing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved uploads without sending them",
    )
    return parser.parse_args()


def resolve_folder(path_text: str) -> Path:
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    return candidate.resolve()


def resolve_glob(pattern: str) -> list[Path]:
    return [path.resolve() for path in WORKSPACE_ROOT.glob(pattern) if path.is_dir()]


def list_result_folders() -> list[Path]:
    return sorted(
        [path.resolve() for path in WORKSPACE_ROOT.glob(DEFAULT_RESULTS_GLOB) if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def relative_repo_path(folder: Path) -> str:
    repo_path = folder.relative_to(WORKSPACE_ROOT).as_posix()
    prefix = "GUI-Docker-Env/"
    if repo_path.startswith(prefix):
        return repo_path[len(prefix) :]
    return repo_path


def collect_folders(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    selected: list[Path] = []
    missing: list[str] = []

    for folder_text in args.folders:
        folder = resolve_folder(folder_text)
        if folder.is_dir():
            selected.append(folder)
        else:
            missing.append(folder_text)

    for pattern in args.globs:
        matches = resolve_glob(pattern)
        if matches:
            selected.extend(matches)
        else:
            missing.append(pattern)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for folder in selected:
        if folder not in seen:
            seen.add(folder)
            deduped.append(folder)
    return deduped, missing


def print_folder_list(folders: list[Path]) -> None:
    for folder in folders:
        print(relative_repo_path(folder))


def upload_gitattributes(api: HfApi, args: argparse.Namespace) -> None:
    if not GITATTRIBUTES_SOURCE.is_file() or args.dry_run:
        return
    api.upload_file(
        path_or_fileobj=str(GITATTRIBUTES_SOURCE),
        path_in_repo=".gitattributes",
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        commit_message="Update dataset gitattributes for archived screenshots",
    )


def upload_folders(args: argparse.Namespace, folders: list[Path]) -> int:
    api = HfApi()
    failures = 0
    allow_patterns = args.include or None
    ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
    if args.exclude:
        ignore_patterns.extend(args.exclude)

    if not args.dry_run:
        try:
            upload_gitattributes(api, args)
        except Exception as exc:
            failures += 1
            print(f"FAILED .gitattributes: {exc}", file=sys.stderr)

    for folder in folders:
        repo_path = relative_repo_path(folder)
        commit_message = args.commit_message or f"Upload {repo_path}"

        if args.dry_run:
            print(f"DRY RUN {repo_path}")
            continue

        try:
            commit_info = api.upload_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                folder_path=folder,
                path_in_repo=repo_path,
                commit_message=commit_message,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        except Exception as exc:
            failures += 1
            print(f"FAILED {repo_path}: {exc}", file=sys.stderr)
            continue

        print(f"UPLOADED {repo_path}")
        print(commit_info.commit_url)

    return failures


def main() -> int:
    args = parse_args()

    if args.list:
        print_folder_list(list_result_folders())
        return 0

    folders, missing = collect_folders(args)
    if missing and not args.skip_missing:
        for item in missing:
            print(f"MISSING {item}", file=sys.stderr)
        return 1

    for item in missing:
        print(f"SKIPPED {item}", file=sys.stderr)

    if not folders:
        print("No folders selected.", file=sys.stderr)
        return 1

    return 1 if upload_folders(args, folders) else 0


if __name__ == "__main__":
    raise SystemExit(main())
