#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated promptopt workspace.")
    parser.add_argument("--source", type=Path, required=True, help="Source repo/worktree to copy.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination workspace path.")
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to ws-matmul-performance-report.md to place in the workspace artifact dir.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()
    report = args.report.resolve()

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy the current source tree but leave out git metadata, build products, and other initiative
    # artifacts. The report itself is re-added explicitly below so the worker gets only that prompt.
    run(
        [
            "rsync",
            "-a",
            "--exclude",
            ".git",
            "--exclude",
            "build",
            "--exclude",
            ".codex/initiatives/artifacts/*",
            f"{source}/",
            f"{dest}/",
        ]
    )

    # Discover the active Triton runtime path from the current Python environment. This is more
    # reliable than copying whatever happens to be present in the worktree snapshot, because those
    # paths may be symlinked or incomplete.
    runtime_root = Path(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib, triton; "
                    "print(pathlib.Path(triton.__file__).resolve().parent)"
                ),
            ],
            text=True,
        ).strip()
    )

    dest_triton = dest / "python" / "triton"
    dest_extra = dest_triton / "language" / "extra"
    dest_extra.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(dest_triton / "backends", ignore_errors=True)
    shutil.rmtree(dest_extra / "cuda", ignore_errors=True)

    # Copy runtime directories with symlinks dereferenced so the isolated workspace is self-contained.
    run(["cp", "-aL", str(runtime_root / "backends"), str(dest_triton / "")])
    run(["cp", "-aL", str(runtime_root / "language" / "extra" / "cuda"), str(dest_extra / "")])

    # Defensive cleanup: a stray package directory named "libdevice" shadows the real module
    # "libdevice.py" and breaks JIT dependency resolution for libdevice.exp.
    shadow_pkg = dest_extra / "cuda" / "libdevice"
    if shadow_pkg.is_dir():
        shutil.rmtree(shadow_pkg)

    artifact_dir = dest / ".codex" / "initiatives" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report, artifact_dir / report.name)

    print(dest)


if __name__ == "__main__":
    main()
