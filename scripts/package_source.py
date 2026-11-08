#!/usr/bin/env python3
"""Create a clean source ZIP that honours this repository's .gitignore rules.

Unlike ``zip -r .``, this deliberately excludes .git, local build products,
virtual environments, datasets, trained artifacts, caches, and prior ZIPs.
It includes both tracked source files and non-ignored worktree files so it is
useful before the first commit as well.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/mini-inference-engine-source.zip"),
        help="Destination ZIP path (default: %(default)s)",
    )
    return parser.parse_args()


def source_paths() -> list[Path]:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [ROOT / line for line in completed.stdout.splitlines() if line]
    return [path for path in paths if path.is_file()]


def main() -> None:
    args = parse_args()
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination == ROOT:
        raise SystemExit("The output path must be a file, not the project directory.")

    paths = source_paths()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.resolve() == destination:
                continue
            archive.write(path, path.relative_to(ROOT))
    print(f"Wrote {len(paths)} source files to {destination}")


if __name__ == "__main__":
    main()
