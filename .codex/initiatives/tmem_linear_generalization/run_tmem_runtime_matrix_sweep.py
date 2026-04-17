#!/usr/bin/env python3
"""Run the full Gluon TMEM runtime matrix without dropping coverage.

The raw file has a few compile-heavy families.  This runner keeps the same
pytest cases but schedules them in buckets that match their cost profile:
small buckets use simple four-way pytest-split, while ld.red and ld/st use
finer splits plus conservative xdist to parallelize CPU-bound compilation.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_FILE = "python/test/gluon/test_tmem_runtime_matrix.py"
LDST_DURATIONS = (
    ".codex/initiatives/tmem_linear_generalization/experiments/results/"
    "ldst_pytest_durations_20260413.json"
)
PYTHONPATH = os.pathsep.join(
    [
        str(REPO_ROOT),
        str(REPO_ROOT / "python"),
        str(REPO_ROOT / "python/test/gluon"),
    ]
)

SPLITN_NODEIDS = [
    f"{TEST_FILE}::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep",
    f"{TEST_FILE}::test_tmem_runtime_matrix_splitn_immediates",
    f"{TEST_FILE}::test_tmem_runtime_matrix_splitn_auto_selects_16x32bx2",
    f"{TEST_FILE}::test_tmem_runtime_matrix_explicit_16x32bx2_matches_splitn",
    f"{TEST_FILE}::test_tmem_runtime_matrix_splitn_16bit_m64_auto_matches_explicit",
    f"{TEST_FILE}::test_tmem_runtime_matrix_blocked_layout_reports_clean_error",
    f"{TEST_FILE}::test_tmem_runtime_matrix_block_descriptor_reports_clean_error",
    f"{TEST_FILE}::test_tmem_runtime_matrix_splitn_rowcol_permuted_auto_selects_16x32bx2",
    f"{TEST_FILE}::test_tmem_runtime_matrix_alloc_source_initialization_lifetime",
]


@dataclass(frozen=True)
class Bucket:
    name: str
    pytest_args: tuple[str, ...]
    splits: int
    xdist: int = 0
    durations_path: str | None = None
    splitting_algorithm: str | None = None


BUCKETS = {
    "cp": Bucket("cp", (TEST_FILE, "-k", "cp"), splits=4),
    "mma": Bucket("mma", (TEST_FILE, "-k", "test_tmem_runtime_matrix_mma"), splits=4),
    "splitn": Bucket("splitn", tuple(SPLITN_NODEIDS), splits=4),
    "ld_red": Bucket("ld_red", (TEST_FILE, "-k", "ld_red"), splits=16, xdist=4),
    "ldst": Bucket(
        "ldst",
        (TEST_FILE, "-k", "ldst"),
        splits=16,
        xdist=4,
        durations_path=LDST_DURATIONS,
        splitting_algorithm="least_duration",
    ),
}

DEFAULT_BUCKETS = ("cp", "mma", "splitn", "ld_red", "ldst")


def make_command(
    bucket: Bucket,
    group: int,
    extra_pytest_args: list[str],
    durations_path: str | None = None,
    store_durations: bool = False,
    clean_durations: bool = False,
    xdist_override: int | None = None,
) -> list[str]:
    cmd = [
        "pytest",
        "-s",
        "--tb=short",
        "--splits",
        str(bucket.splits),
        "--group",
        str(group),
        "-q",
    ]
    effective_durations_path = durations_path if durations_path is not None else bucket.durations_path
    if effective_durations_path:
        cmd.extend(["--durations-path", effective_durations_path])
    if store_durations:
        cmd.append("--store-durations")
    if clean_durations:
        cmd.append("--clean-durations")
    if bucket.splitting_algorithm:
        cmd.extend(["--splitting-algorithm", bucket.splitting_algorithm])
    xdist = bucket.xdist if xdist_override is None else xdist_override
    if xdist:
        cmd.extend(["-n", str(xdist)])
    cmd.extend(bucket.pytest_args)
    cmd.extend(extra_pytest_args)
    return cmd


def duration_file_for_group(bucket: Bucket, group: int, log_dir: Path) -> Path:
    return log_dir / f"{bucket.name}_g{group:02d}_durations.json"


def prepare_duration_file(bucket: Bucket, group: int, args: argparse.Namespace, log_dir: Path) -> str | None:
    if not args.store_durations:
        return bucket.durations_path

    duration_path = duration_file_for_group(bucket, group, log_dir)
    if bucket.durations_path:
        source = REPO_ROOT / bucket.durations_path
        if source.exists():
            shutil.copyfile(source, duration_path)
    return str(duration_path)


def merge_duration_files(bucket: Bucket, args: argparse.Namespace, log_dir: Path) -> Path | None:
    if not args.store_durations:
        return None

    if bucket.durations_path:
        target = REPO_ROOT / bucket.durations_path
    else:
        target = log_dir / f"{bucket.name}_pytest_durations.json"

    merged: dict[str, float] = {}
    if target.exists() and not args.clean_durations:
        merged.update(json.loads(target.read_text()))

    for group in range(1, bucket.splits + 1):
        duration_path = duration_file_for_group(bucket, group, log_dir)
        if duration_path.exists():
            merged.update(json.loads(duration_path.read_text()))

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2) + "\n")
    return target


def run_one(
    bucket: Bucket, group: int, gpu: str, args: argparse.Namespace, log_dir: Path
) -> tuple[int, float, Path]:
    durations_path = prepare_duration_file(bucket, group, args, log_dir)
    cmd = make_command(
        bucket,
        group,
        args.pytest_arg,
        durations_path=durations_path,
        store_durations=args.store_durations,
        clean_durations=args.clean_durations,
        xdist_override=args.xdist_overrides.get(bucket.name),
    )
    log_path = log_dir / f"{bucket.name}_g{group:02d}_gpu{gpu}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["TRITON_CACHE_DIR"] = f"{args.cache_prefix}-gpu{gpu}"

    start = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"cwd: {REPO_ROOT}\n")
        log.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
        log.write(f"PYTHONPATH={env['PYTHONPATH']}\n")
        log.write(f"TRITON_CACHE_DIR={env['TRITON_CACHE_DIR']}\n")
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_per_group if args.timeout_per_group else None,
            )
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {args.timeout_per_group}s\n")
            rc = 124
    elapsed = time.monotonic() - start
    return rc, elapsed, log_path


def format_command(bucket: Bucket, group: int, gpu: str, args: argparse.Namespace, log_dir: Path) -> str:
    durations_path = None
    if args.store_durations:
        durations_path = str(duration_file_for_group(bucket, group, log_dir))
    cmd = make_command(
        bucket,
        group,
        args.pytest_arg,
        durations_path=durations_path,
        store_durations=args.store_durations,
        clean_durations=args.clean_durations,
        xdist_override=args.xdist_overrides.get(bucket.name),
    )
    cache_dir = f"{args.cache_prefix}-gpu{gpu}"
    return " ".join(
        [
            f"CUDA_VISIBLE_DEVICES={gpu}",
            f"PYTHONPATH={PYTHONPATH}",
            f"TRITON_CACHE_DIR={cache_dir}",
        ]
        + cmd
    )


def parse_xdist_overrides(values: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--xdist-override expects BUCKET=N, got {value!r}")
        bucket_name, count_text = value.split("=", 1)
        if bucket_name not in BUCKETS:
            raise SystemExit(f"unknown bucket in --xdist-override: {bucket_name!r}")
        try:
            count = int(count_text)
        except ValueError:
            raise SystemExit(f"xdist worker count must be an integer, got {count_text!r}") from None
        if count < 0:
            raise SystemExit("xdist worker count must be non-negative")
        overrides[bucket_name] = count
    return overrides


def selected_groups(bucket: Bucket, args: argparse.Namespace) -> list[int]:
    groups = list(range(1, bucket.splits + 1)) if args.groups is None else sorted(set(args.groups))
    invalid = [group for group in groups if group < 1 or group > bucket.splits]
    if invalid:
        raise SystemExit(
            f"bucket {bucket.name!r} has {bucket.splits} groups; invalid --groups values: {invalid}"
        )
    return groups


def gpu_for_group(group: int, args: argparse.Namespace) -> str:
    return args.gpus[(group - 1) % len(args.gpus)]


def group_waves(groups: list[int], args: argparse.Namespace) -> list[list[tuple[int, str]]]:
    by_gpu = {gpu: [group for group in groups if gpu_for_group(group, args) == gpu] for gpu in args.gpus}
    waves: list[list[tuple[int, str]]] = []
    while any(by_gpu.values()):
        wave: list[tuple[int, str]] = []
        for gpu in args.gpus:
            if by_gpu[gpu]:
                wave.append((by_gpu[gpu].pop(0), gpu))
        waves.append(wave)
    return waves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(BUCKETS),
        default=list(DEFAULT_BUCKETS),
        help="runtime-matrix buckets to run; defaults to the full matrix partition",
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["0", "1", "2", "3"], help="GPU ids to use"
    )
    parser.add_argument(
        "--cache-prefix",
        default="/tmp/triton-cache-tmem-runtime-matrix",
        help="stable per-GPU cache prefix; the runner appends -gpu<id>",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="directory for per-shard logs; defaults under the initiative results directory",
    )
    parser.add_argument(
        "--timeout-per-group", type=int, default=1800, help="0 disables per-group timeout"
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    parser.add_argument(
        "--store-durations",
        action="store_true",
        help=(
            "store per-test durations without shard write races; each group writes a "
            "private durations file, then the runner merges them after the bucket passes"
        ),
    )
    parser.add_argument(
        "--clean-durations",
        action="store_true",
        help="with --store-durations, drop stale duration entries not seen by any group",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=int,
        default=None,
        help=(
            "run only these pytest-split group numbers for each selected bucket; "
            "use this for exact shard reruns after a failure or timeout"
        ),
    )
    parser.add_argument(
        "--xdist-override",
        action="append",
        default=[],
        metavar="BUCKET=N",
        help=(
            "override inner pytest-xdist worker count for one bucket, for example "
            "ldst=8; repeat for multiple buckets. Defaults remain unchanged."
        ),
    )
    parser.add_argument(
        "--pytest-arg", action="append", default=[], help="extra argument appended to every pytest command"
    )
    args = parser.parse_args()
    args.xdist_overrides = parse_xdist_overrides(args.xdist_override)
    return args


def main() -> int:
    args = parse_args()
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else Path(
        f".codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_{stamp}"
    )
    log_dir = (REPO_ROOT / log_dir).resolve()
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int, str, int, float, Path]] = []
    for name in args.categories:
        bucket = BUCKETS[name]
        groups_to_run = selected_groups(bucket, args)
        xdist = args.xdist_overrides.get(bucket.name, bucket.xdist)
        print(f"== {bucket.name}: {len(groups_to_run)}/{bucket.splits} split groups, xdist={xdist or 0} ==")
        waves = group_waves(groups_to_run, args)
        if args.dry_run:
            for wave in waves:
                for group, gpu in wave:
                    print(format_command(bucket, group, gpu, args, log_dir))
            continue
        for wave in waves:
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                future_map = {
                    pool.submit(run_one, bucket, group, gpu, args, log_dir): (group, gpu)
                    for group, gpu in wave
                }
                for future in as_completed(future_map):
                    group, gpu = future_map[future]
                    rc, elapsed, log_path = future.result()
                    status = "PASS" if rc == 0 else f"FAIL({rc})"
                    print(
                        f"{bucket.name} group {group}/{bucket.splits} "
                        f"gpu {gpu}: {status} {elapsed:.1f}s {log_path}"
                    )
                    if rc != 0:
                        failures.append((bucket.name, group, gpu, rc, elapsed, log_path))
            if failures and args.dry_run is False:
                print("Stopping after failing wave.")
                break
        if failures and args.dry_run is False:
            break
        merged_durations = merge_duration_files(bucket, args, log_dir)
        if merged_durations is not None:
            print(f"{bucket.name}: merged durations into {merged_durations}")

    if failures:
        print("\nFailures:")
        for bucket, group, gpu, rc, elapsed, log_path in failures:
            print(f"- {bucket} group {group} gpu {gpu}: rc={rc} elapsed={elapsed:.1f}s log={log_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
