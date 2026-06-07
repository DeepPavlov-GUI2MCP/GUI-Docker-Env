from __future__ import annotations

import zipfile
from pathlib import Path

from lib_results_paths import IMAGES_ZIP_NAME, RUN_ZIP_NAME

ARCHIVE_SKIP_NAMES = {RUN_ZIP_NAME, IMAGES_ZIP_NAME}


def flatten_images_zip(run_dir: Path) -> int:
    images_zip = run_dir / IMAGES_ZIP_NAME
    if not images_zip.is_file():
        return 0
    extracted = 0
    with zipfile.ZipFile(images_zip, "r") as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            target = run_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as handle:
                handle.write(source.read())
            extracted += 1
    images_zip.unlink()
    return extracted


def iter_run_files(run_dir: Path) -> list[Path]:
    files = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in ARCHIVE_SKIP_NAMES
    ]
    files.sort(key=lambda path: path.relative_to(run_dir).as_posix())
    return files


def zip_run_dir(
    run_dir: Path,
    *,
    dry_run: bool = False,
    remove_files: bool = True,
    zip_name: str = RUN_ZIP_NAME,
) -> dict:
    run_dir = run_dir.resolve()
    zip_path = run_dir / zip_name
    flattened_images = 0 if dry_run else flatten_images_zip(run_dir)
    files = iter_run_files(run_dir)
    if not files:
        return {
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "file_count": 0,
            "flattened_images": flattened_images,
            "skipped": True,
            "reason": "no files to archive",
        }
    if dry_run:
        total_bytes = sum(path.stat().st_size for path in files)
        if (run_dir / IMAGES_ZIP_NAME).is_file():
            with zipfile.ZipFile(run_dir / IMAGES_ZIP_NAME, "r") as archive:
                flattened_images = sum(1 for name in archive.namelist() if not name.endswith("/"))
        return {
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "file_count": len(files) + flattened_images,
            "total_bytes": total_bytes,
            "flattened_images": flattened_images,
            "dry_run": True,
        }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.relative_to(run_dir).as_posix())

    removed = 0
    if remove_files:
        for file_path in files:
            file_path.unlink()
            removed += 1

    return {
        "run_dir": str(run_dir),
        "zip_path": str(zip_path),
        "file_count": len(files),
        "flattened_images": flattened_images,
        "removed_files": removed,
        "zip_bytes": zip_path.stat().st_size,
    }


def unzip_run_dir(
    run_dir: Path,
    *,
    dry_run: bool = False,
    remove_zip: bool = False,
    zip_name: str = RUN_ZIP_NAME,
) -> dict:
    run_dir = run_dir.resolve()
    zip_path = run_dir / zip_name
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
