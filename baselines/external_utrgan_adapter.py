"""Run official UTRGAN generation on the standardized 5'UTR input pack.

UTRGAN generates a batch of optimized 5'UTRs rather than editing each source
transcript conditionally. The adapter pairs generated candidates with the
matched public input rows, keeps CDS and 3'UTR fixed, and scores both source and
candidate UTRs with the shared independent MEF proxy oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.oracle import score_utr
from mrna_editflow.eval.metrics import edit_distance


CLAIM_POLICY = (
    "This artifact contains measured UTRGAN outputs scored with the shared MEF "
    "proxy oracle. It is an external 5'UTR baseline, not wet-lab evidence and "
    "not proof of MEF superiority."
)
MODEL_NAME = "UTRGAN"
TASK_FAMILY = "utr5_only"


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain an object")
            rows.append(dict(payload))
    return rows


def _write_json(payload: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _write_jsonl(rows: Sequence[Mapping[str, object]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def _normalise_rna(seq: object) -> str:
    return "".join(str(seq or "").upper().replace("T", "U").split())


def _resolve_pack_file(summary_path: str, summary: Mapping[str, object], key: str, default_name: str) -> str:
    outputs = summary.get("outputs", {})
    outputs = outputs if isinstance(outputs, Mapping) else {}
    value = outputs.get(key)
    if isinstance(value, str):
        candidate = value if os.path.isabs(value) else os.path.join(os.path.dirname(summary_path), value)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(os.path.dirname(summary_path), default_name)


def _read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _version(executable: str) -> str:
    proc = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return proc.stdout.strip() or f"unknown exit={proc.returncode}"


def run_utrgan_adapter(
    *,
    input_pack_summary: str,
    executable: str,
    tool_root: str,
    out_dir: str,
    limit: Optional[int] = None,
    steps: int = 10,
    learning_rate_exponent: int = 5,
    timeout_s: float = 1800.0,
) -> dict[str, object]:
    """Generate UTRGAN candidates and write a real-run output contract."""
    start = time.perf_counter()
    pack = _load_json(input_pack_summary)
    utr_inputs_path = _resolve_pack_file(
        input_pack_summary, pack, "utr5_jsonl", "utr5_inputs.jsonl"
    )
    input_rows = _load_jsonl(utr_inputs_path)
    selected = input_rows[:limit] if limit is not None else input_rows
    work_dir = os.path.join(tool_root, "src", "mrl_te_optimization")
    official_out = os.path.join(work_dir, "outputs")
    os.makedirs(official_out, exist_ok=True)
    for name in (
        "init_mrl_FMRL.txt",
        "opt_mrl_FMRL.txt",
        "opt_seqs_FMRL.txt",
        "init_seqs_FMRL.txt",
    ):
        path = os.path.join(official_out, name)
        if os.path.exists(path):
            os.remove(path)

    proc = subprocess.run(
        [
            executable,
            "-bs",
            str(len(selected)),
            "-lr",
            str(int(learning_rate_exponent)),
            "-task",
            "mrl",
            "-gpu",
            "-1",
            "-s",
            str(int(steps)),
        ],
        cwd=work_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
    )
    elapsed_s = float(time.perf_counter() - start)
    if proc.returncode != 0:
        raise RuntimeError(f"UTRGAN exit={proc.returncode}: {proc.stdout[-4000:]}")

    generated = [_normalise_rna(seq) for seq in _read_lines(os.path.join(official_out, "opt_seqs_FMRL.txt"))]
    external_scores = [
        float(value) for value in _read_lines(os.path.join(official_out, "opt_mrl_FMRL.txt"))
    ]
    if len(generated) != len(selected) or len(external_scores) != len(selected):
        raise ValueError(
            f"UTRGAN output coverage mismatch: inputs={len(selected)} "
            f"sequences={len(generated)} scores={len(external_scores)}"
        )
    wall_per_row = elapsed_s / max(len(selected), 1)
    outputs: list[dict[str, object]] = []
    for source, designed_utr, external_mrl in zip(selected, generated, external_scores):
        native_utr = _normalise_rna(source.get("native_five_utr"))
        cds = _normalise_rna(source.get("fixed_cds_context"))
        native_score = score_utr(native_utr, cds[:12])
        candidate_score = score_utr(designed_utr, cds[:12])
        features = candidate_score.get("features", {})
        features = features if isinstance(features, Mapping) else {}
        utr_edit_distance = int(edit_distance(designed_utr, native_utr))
        outputs.append(
            {
                "transcript_id": source.get("transcript_id"),
                "model_name": MODEL_NAME,
                "designed_five_utr": designed_utr,
                "wall_clock_s": float(wall_per_row),
                "cds_unchanged": True,
                "three_utr_unchanged": True,
                "protein_identity_exact_1": True,
                "te_proxy": float(candidate_score["ensemble_te"]),
                "te_proxy_delta_vs_native": float(
                    candidate_score["ensemble_te"] - native_score["ensemble_te"]
                ),
                "uaug_count": float(features.get("uaug_count", 0.0)),
                "kozak_score": float(features.get("kozak", 0.0)),
                "start_accessibility_proxy": float(
                    features.get("start_accessibility", 0.0)
                ),
                "external_utrgan_mrl_score": float(external_mrl),
                "utr_edit_distance_vs_native": utr_edit_distance,
                "normalized_utr_edit_distance_vs_native": float(
                    utr_edit_distance / max(len(designed_utr), len(native_utr), 1)
                ),
                "designed_utr_length": len(designed_utr),
                "native_utr_length": len(native_utr),
                "utr_length_delta": len(designed_utr) - len(native_utr),
                "exact_native_utr_match": bool(designed_utr == native_utr),
                "native_te_proxy": float(native_score["ensemble_te"]),
                "native_five_utr": native_utr,
                "fixed_cds_context": cds,
                "fixed_three_utr_context": _normalise_rna(
                    source.get("fixed_three_utr_context")
                ),
            }
        )

    os.makedirs(out_dir, exist_ok=True)
    outputs_path = os.path.join(out_dir, "utr5_outputs.jsonl")
    stdout_path = os.path.join(out_dir, "official_stdout.log")
    _write_jsonl(outputs, outputs_path)
    with open(stdout_path, "w", encoding="utf-8") as fh:
        fh.write(proc.stdout)

    def mean(key: str) -> float:
        values = [float(row[key]) for row in outputs if isinstance(row.get(key), (int, float))]
        return float(sum(values) / len(values)) if values else 0.0

    pack_outputs = pack.get("outputs", {})
    pack_outputs = pack_outputs if isinstance(pack_outputs, Mapping) else {}
    dataset = pack.get("dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    summary: dict[str, object] = {
        "artifact_kind": "external_sota_real_run_summary",
        "claim_policy": CLAIM_POLICY,
        "model_name": MODEL_NAME,
        "task_family": TASK_FAMILY,
        "protocol_fidelity": (
            "official_code_paper_default_10000_steps"
            if int(steps) == 10000
            else f"official_code_budgeted_{int(steps)}_steps_vs_paper_default_10000"
        ),
        "protocol_fidelity_sufficient_for_sota_reproduction": bool(
            int(steps) == 10000
        ),
        "input_pack": {
            "summary_sha256": _sha256_file(input_pack_summary),
            "cds_protein_jsonl_sha256": pack_outputs.get("cds_protein_jsonl_sha256"),
            "utr5_jsonl_sha256": pack_outputs.get("utr5_jsonl_sha256"),
        },
        "dataset": {
            "records_jsonl_sha256": dataset.get("records_jsonl_sha256"),
            "split_name": dataset.get("split_name"),
            "seed": dataset.get("seed"),
        },
        "runtime": {
            "elapsed_s": elapsed_s,
            "batch_size": len(selected),
            "steps": int(steps),
            "learning_rate_exponent": int(learning_rate_exponent),
            "timeout_s": float(timeout_s),
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "configured_cuda_visible_devices": os.environ.get(
                "UTRGAN_CUDA_VISIBLE_DEVICES"
            ),
        },
        "executable": {
            "path": os.path.abspath(executable),
            "version": _version(executable),
            "sha256": _sha256_file(executable),
        },
        "adapter": {
            "module": "mrna_editflow.baselines.external_utrgan_adapter",
            "pairing_policy": "one generated UTR per ordered input-pack row",
            "fixed_regions": ["CDS", "3UTR"],
            "shared_proxy_oracle": "mrna_editflow.eval.oracle.score_utr",
        },
        "n_inputs": len(selected),
        "n_outputs": len(outputs),
        "n_failures": 0,
        "mean_wall_clock_s": mean("wall_clock_s"),
        "cds_unchanged_fraction": 1.0 if outputs else 0.0,
        "three_utr_unchanged_fraction": 1.0 if outputs else 0.0,
        "protein_identity_exact_1_fraction": 1.0 if outputs else 0.0,
        "mean_te_proxy_delta_vs_native": mean("te_proxy_delta_vs_native"),
        "mean_te_proxy": mean("te_proxy"),
        "mean_external_utrgan_mrl_score": mean("external_utrgan_mrl_score"),
        "mean_uaug_count": mean("uaug_count"),
        "mean_kozak_score": mean("kozak_score"),
        "mean_start_accessibility_proxy": mean("start_accessibility_proxy"),
        "mean_utr_edit_distance_vs_native": mean("utr_edit_distance_vs_native"),
        "mean_normalized_utr_edit_distance_vs_native": mean(
            "normalized_utr_edit_distance_vs_native"
        ),
        "mean_designed_utr_length": mean("designed_utr_length"),
        "mean_native_utr_length": mean("native_utr_length"),
        "mean_utr_length_delta": mean("utr_length_delta"),
        "exact_native_utr_match_fraction": (
            float(sum(bool(row.get("exact_native_utr_match")) for row in outputs) / len(outputs))
            if outputs
            else 0.0
        ),
        "unique_designed_utr_fraction": (
            float(len({str(row.get("designed_five_utr")) for row in outputs}) / len(outputs))
            if outputs
            else 0.0
        ),
        "outputs": {
            "utr5_outputs_jsonl": outputs_path,
            "utr5_outputs_sha256": _sha256_file(outputs_path),
            "official_stdout_log": stdout_path,
            "official_stdout_sha256": _sha256_file(stdout_path),
        },
    }
    _write_json(summary, os.path.join(out_dir, "summary.json"))
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack-summary", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--tool-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate-exponent", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_utrgan_adapter(
        input_pack_summary=args.input_pack_summary,
        executable=args.executable,
        tool_root=args.tool_root,
        out_dir=args.out_dir,
        limit=args.limit,
        steps=args.steps,
        learning_rate_exponent=args.learning_rate_exponent,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLAIM_POLICY", "run_utrgan_adapter", "main"]
