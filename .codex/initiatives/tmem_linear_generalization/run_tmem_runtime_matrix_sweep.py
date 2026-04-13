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
import os
from pathlib import Path
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

SPLITN_NODEIDS = [
    f"{TEST_FILE}::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep",
    f"{TEST_FILE}::test_tmem_runtime_matrix_splitn_immediates",
    f"{TEST_FILE}::test_tmem_runtime_matrix_splitn_auto_selects_16x32bx2",
    f"{TEST_FILE}::test_tmem_runtime_matrix_explicit_16x32bx2_matches_splitn",
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


def make_command(bucket: Bucket, group: int, extra_pytest_args: list[str]) -> list[str]:
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
    if bucket.durations_path:
        cmd.extend(["--durations-path", bucket.durations_path])
    if bucket.splitting_algorithm:
        cmd.extend(["--splitting-algorithm", bucket.splitting_algorithm])
    if bucket.xdist:
        cmd.extend(["-n", str(bucket.xdist)])
    cmd.extend(bucket.pytest_args)
    cmd.extend(extra_pytest_args)
    return cmd


def run_one(
    bucket: Bucket, group: int, gpu: str, args: argparse.Namespace, log_dir: Path
) -> tuple[int, float, Path]:
    cmd = make_command(bucket, group, args.pytest_arg)
    log_path = log_dir / f"{bucket.name}_g{group:02d}_gpu{gpu}.log"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["TRITON_CACHE_DIR"] = f"{args.cache_prefix}-gpu{gpu}"

    start = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"cwd: {REPO_ROOT}\n")
        log.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
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


def format_command(bucket: Bucket, group: int, gpu: str, args: argparse.Namespace) -> str:
    cmd = make_command(bucket, group, args.pytest_arg)
    cache_dir = f"{args.cache_prefix}-gpu{gpu}"
    return " ".join([f"CUDA_VISIBLE_DEVICES={gpu}", f"TRITON_CACHE_DIR={cache_dir}"] + cmd)


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
        "--pytest-arg", action="append", default=[], help="extra argument appended to every pytest command"
    )
    return parser.parse_args()


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
        print(f"== {bucket.name}: {bucket.splits} split groups, xdist={bucket.xdist or 0} ==")
        if args.dry_run:
            for group in range(1, bucket.splits + 1):
                gpu = args.gpus[(group - 1) % len(args.gpus)]
                print(format_command(bucket, group, gpu, args))
            continue
        for wave_start in range(1, bucket.splits + 1, len(args.gpus)):
            groups = list(range(wave_start, min(bucket.splits, wave_start + len(args.gpus) - 1) + 1))
            with ThreadPoolExecutor(max_workers=len(groups)) as pool:
                future_map = {
                    pool.submit(run_one, bucket, group, args.gpus[i], args, log_dir): (group, args.gpus[i])
                    for i, group in enumerate(groups)
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

    if failures:
        print("\nFailures:")
        for bucket, group, gpu, rc, elapsed, log_path in failures:
            print(f"- {bucket} group {group} gpu {gpu}: rc={rc} elapsed={elapsed:.1f}s log={log_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
