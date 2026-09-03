#!/usr/bin/env python3
"""Build and verify a deterministic source-only release ZIP."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


BLOCKED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".mat",
    ".joblib",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
}
BLOCKED_PARTS = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
    "dist",
}
TOP_LEVEL_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "LICENSE",
    "METHODS_CODE_MAP.md",
    "README.md",
    "SUBMISSION_DATA_NOTE.md",
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-full.txt",
}
SOURCE_DIRECTORIES = {"config", "data", "scripts", "src", "tests"}


def is_blocked_part(part: str) -> bool:
    return part in BLOCKED_PARTS or part.endswith(".egg-info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/SustainableConcrete_All_Methods_Source.zip"),
    )
    parser.add_argument(
        "--root-name",
        default="SustainableConcrete_All_Methods_Source",
        help="Single top-level directory name stored inside the ZIP.",
    )
    return parser.parse_args()


def included_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(is_blocked_part(part) for part in relative.parts):
            continue
        if len(relative.parts) == 1:
            if relative.name not in TOP_LEVEL_FILES:
                continue
        elif relative.parts[0] not in SOURCE_DIRECTORIES:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Symlinks are not permitted in the archive: {relative}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            continue
        if relative.parts[0] == "data" and relative.name not in {
            "README.md",
            "schema.json",
        }:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (
        args.output.resolve()
        if args.output.is_absolute()
        else (root / args.output).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.root_name
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("--root-name must be one safe directory name")
    files = included_files(root)
    if not files:
        raise RuntimeError("No source files were selected")

    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        names = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive_name = f"{prefix}/{relative}"
            info = ZipInfo(archive_name, date_time=(2026, 8, 18, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
            names.append(archive_name)
        manifest = "\n".join(names) + "\n"
        manifest_info = ZipInfo(
            f"{prefix}/ARCHIVE_CONTENTS.txt",
            date_time=(2026, 8, 18, 0, 0, 0),
        )
        manifest_info.compress_type = ZIP_DEFLATED
        manifest_info.external_attr = 0o644 << 16
        archive.writestr(manifest_info, manifest)

    with ZipFile(output) as archive:
        archived = archive.namelist()
        blocked = [
            name
            for name in archived
            if Path(name).suffix.lower() in BLOCKED_SUFFIXES
            or any(is_blocked_part(part) for part in Path(name).parts)
        ]
        if blocked:
            raise RuntimeError(f"Blocked files entered the archive: {blocked}")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Archive CRC verification failed for {bad}")

    checksum = archive_sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n", encoding="ascii"
    )
    print(f"Created {output}")
    print(f"Files: {len(files) + 1}")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()
