#!/usr/bin/env python3
"""Run the publisher's MPRNA counter for one E-MTAB-10902 sample."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


MAIN_CLASS = "scripts.lincs.patch.AnalyzeConservedPatches"
NZIP_READ1_ADAPTER = "ATAATTCGATATCCGCATGCTAGC"


def _run_slice(
    classpath: str,
    library: Path,
    fastq: Path,
    output_dir: Path,
    sample_id: str,
    slice_count: int,
    slice_index: int,
) -> str:
    output_path = output_dir / f"{sample_id}.counts.{slice_index}.txt"
    partial_path = output_dir / f"{sample_id}.counts.{slice_index}.txt.partial"
    log_path = output_dir / f"{sample_id}.counts.{slice_index}.log"
    command = [
        "java",
        "-Xmx2g",
        "-cp",
        classpath,
        MAIN_CLASS,
        "count_reads",
        str(library),
        str(fastq),
        "-adapter",
        NZIP_READ1_ADAPTER,
        "-slice",
        str(slice_count),
        str(slice_index),
        str(partial_path),
        "1000000000",
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=True)
    partial_path.replace(output_path)
    return output_path.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--fastq", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slice-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.slice_count <= 0 or not 1 <= args.workers <= args.slice_count:
        raise ValueError("workers and slice count are inconsistent")
    if not args.library.is_file() or not args.fastq.is_file():
        raise FileNotFoundError("library or FASTQ input is absent")
    jars = [args.jar_dir / name for name in ("compbioLib.jar", "compbio.jar", "picard.jar")]
    if not all(path.is_file() for path in jars):
        raise FileNotFoundError("publisher MPRNA jars are incomplete")
    classpath = ":".join(str(path) for path in jars)
    args.output_dir.mkdir(parents=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _run_slice,
                classpath,
                args.library,
                args.fastq,
                args.output_dir,
                args.sample_id,
                args.slice_count,
                slice_index,
            )
            for slice_index in range(args.slice_count)
        ]
        completed = [future.result() for future in concurrent.futures.as_completed(futures)]

    prefix = args.output_dir / f"{args.sample_id}.counts"
    combined = args.output_dir / f"{args.sample_id}.combined.txt"
    combine_log = args.output_dir / f"{args.sample_id}.combine.log"
    command = [
        "java",
        "-Xmx2g",
        "-cp",
        classpath,
        MAIN_CLASS,
        "combine_count_files",
        str(prefix),
        "0",
        str(args.slice_count - 1),
        str(combined),
    ]
    with combine_log.open("w", encoding="utf-8") as log_handle:
        subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=True)

    summary = {
        "schema_version": "route_a_v3_route2_emtab10902_sample_count.v1",
        "sample_id": args.sample_id,
        "slice_count": args.slice_count,
        "worker_count": args.workers,
        "completed_slice_count": len(completed),
        "combined_count_filename": combined.name,
        "publisher_counter_main_class": MAIN_CLASS,
        "read1_adapter": NZIP_READ1_ADAPTER,
    }
    (args.output_dir / "count_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
