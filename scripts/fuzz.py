#!/usr/bin/env python3
"""Seed, build, and replay (or fuzz) cargo-fuzz targets in parallel."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The crates deny unsafe code, so AddressSanitizer costs ~4x throughput for little coverage.
SANITIZER = ("--sanitizer", "none")


def main():
    targets = subprocess.check_output(
        ["cargo", "fuzz", "list"], cwd=ROOT, text=True
    ).splitlines()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets", nargs="*", choices=targets, metavar="TARGET",
        help=f"default: all of {', '.join(targets)}",
    )
    parser.add_argument(
        "--seconds", type=int, default=0,
        help="fuzzing seconds per target; zero replays the corpus",
    )
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 1,
        help="targets run concurrently",
    )
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must be nonnegative")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", "fuzz/Cargo.toml", "--example", "seed"],
        cwd=ROOT, check=True,
    )
    subprocess.run(["cargo", "fuzz", "build", *SANITIZER], cwd=ROOT, check=True)
    limit = f"-max_total_time={args.seconds}" if args.seconds > 0 else "-runs=0"

    def run(target):
        command = ["cargo", "fuzz", "run", *SANITIZER, target, "--", "-timeout=10", limit]
        return subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = {pool.submit(run, target): target for target in args.targets or targets}
        for job in as_completed(jobs):
            target = jobs[job]
            result = job.result()
            if result.returncode:
                print(result.stdout, flush=True)
                failed.append(target)
                summary = "FAILED"
            else:
                summary = next(
                    (line for line in reversed(result.stdout.splitlines()) if line.startswith("Done ")),
                    "Done",
                )
            print(f"{target}: {summary}", flush=True)
    if failed:
        print(f"Failed targets: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
