"""Run the official Hugging Face codonGPT checkpoint on the CDS input pack.

The public checkpoint exposes the pretrained GPT-2 codon model, tokenizer, and
synonymous-logit processor. It does not expose the paper's task-specific RL
policies. This adapter therefore measures constrained pretrained generation and
keeps the protocol-fidelity and superiority gates closed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections.abc import Callable
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.external_lineardesign_adapter import (
    _codon_kl,
    _codons,
    _load_json,
    _load_jsonl,
    _normalise_rna,
    _protein_identity,
    _resolve_pack_file,
    _sha256_file,
    _write_json,
    _write_jsonl,
)
from mrna_editflow.core.constants import (
    SYNONYMOUS_CODONS,
    is_valid_cds,
    translate,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.prepare_codon import gc3_fraction
from mrna_editflow.eval.metrics import (
    cai,
    codon_weights_from_reference,
    gc_fraction,
)


CLAIM_POLICY = (
    "This artifact measures synonymous constrained generation from the public "
    "naniltx/codonGPT Hugging Face checkpoint. The public repository provides "
    "the pretrained model but not the paper's task-specific RL policies or "
    "reward-training artifacts. Do not call this an RL reproduction, wet-lab "
    "validation, full-length generation result, or MEF superiority claim."
)
MODEL_NAME = "codonGPT"
TASK_FAMILY = "cds_protein_conditioned"
DEFAULT_HF_REPO = "naniltx/codonGPT"
DEFAULT_HF_REVISION = "ee7017c4bdd285206b87be2e65a28272ff4ac88e"
BatchGenerator = Callable[
    [Sequence[Mapping[str, object]], int],
    Sequence[str],
]


def _append_jsonl(row: Mapping[str, object], path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _load_module(path: str, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _executable_version(executable: str) -> str:
    proc = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    text = proc.stdout.strip()
    return text or f"unknown exit={proc.returncode}"


def _target_protein(row: Mapping[str, object]) -> str:
    protein = str(
        row.get("protein_target_with_stop")
        or row.get("protein_target")
        or ""
    )
    return protein if protein.endswith("*") else protein + "*"


def _codon_accuracy(designed_cds: str, native_cds: str) -> float:
    designed = _codons(designed_cds)
    native = _codons(native_cds)
    denominator = max(len(native), len(designed), 1)
    return float(sum(a == b for a, b in zip(designed, native)) / denominator)


def _score_generated_cds(
    row: Mapping[str, object],
    designed_cds: str,
    *,
    codon_weights: Mapping[str, float],
    wall_clock_s: float,
) -> dict[str, object]:
    native_cds = _normalise_rna(row.get("native_cds"))
    designed_cds = _normalise_rna(designed_cds)
    target = _target_protein(row)
    designed_protein = translate(designed_cds)
    identity = _protein_identity(target, designed_protein)
    valid = bool(is_valid_cds(designed_cds))
    return {
        "transcript_id": row.get("transcript_id"),
        "model_name": MODEL_NAME,
        "designed_cds": designed_cds,
        "native_cds": native_cds,
        "wall_clock_s": float(wall_clock_s),
        "valid_cds": valid,
        "protein_identity": float(identity),
        "protein_identity_exact_1": bool(identity == 1.0),
        "cai": float(cai(designed_cds, codon_weights)),
        "gc": float(gc_fraction(designed_cds)),
        "gc3": float(gc3_fraction(designed_cds)),
        "codon_usage_kl_vs_native": float(
            _codon_kl(designed_cds, native_cds)
        ),
        "codon_pair_kl_vs_native": float(
            _codon_kl(designed_cds, native_cds, pairs=True)
        ),
        "codon_accuracy_vs_native": _codon_accuracy(
            designed_cds,
            native_cds,
        ),
        "generation_protocol": (
            "official_hf_pretrained_checkpoint_synonymous_masked_sampling"
        ),
    }


class _CodonGPTRuntime:
    def __init__(self, model_dir: str, device: str) -> None:
        import torch
        from transformers import GPT2LMHeadModel

        tokenizer_module = _load_module(
            os.path.join(model_dir, "tokenizer.py"),
            "_codongpt_official_tokenizer",
        )
        tokenizer_class = getattr(tokenizer_module, "CodonTokenizer")
        self.tokenizer = tokenizer_class()
        self.aa_to_codon = {
            aa: [codon.replace("U", "T") for codon in codons]
            for aa, codons in SYNONYMOUS_CODONS.items()
        }
        self.torch = torch
        self.device = torch.device(device)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model.eval().to(self.device)
        self.allowed_ids = {
            aa: [
                int(token)
                for token in self.tokenizer.convert_tokens_to_ids(codons)
            ]
            for aa, codons in self.aa_to_codon.items()
        }

    def generate(
        self,
        rows: Sequence[Mapping[str, object]],
        batch_seed: int,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[str]:
        torch = self.torch
        targets = [_target_protein(row) for row in rows]
        unsupported = sorted(
            {
                aa
                for target in targets
                for aa in target
                if aa not in self.allowed_ids
            }
        )
        if unsupported:
            raise ValueError(
                f"unsupported target amino acids: {unsupported}"
            )
        tokenizer = self.tokenizer
        allowed_ids = self.allowed_ids
        eos_id = int(tokenizer.eos_token_id)

        class PositionSynonymMask:
            def __call__(self, input_ids: object, scores: object) -> object:
                step = int(input_ids.shape[1]) - 1
                mask = torch.full_like(scores, -float("inf"))
                for row_index, target in enumerate(targets):
                    ids = (
                        allowed_ids[target[step]]
                        if step < len(target)
                        else [eos_id]
                    )
                    mask[row_index, ids] = 0.0
                return scores + mask

        torch.manual_seed(int(batch_seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(batch_seed))
        input_ids = torch.full(
            (len(rows), 1),
            int(tokenizer.bos_token_id),
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max(len(target) for target in targets) + 1,
                do_sample=True,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                logits_processor=[PositionSynonymMask()],
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=eos_id,
                use_cache=True,
            )
        sequences: list[str] = []
        for row_index, target in enumerate(targets):
            codons: list[str] = []
            for token_id in generated[row_index, 1:].tolist():
                if token_id == eos_id:
                    break
                if token_id == int(tokenizer.pad_token_id):
                    continue
                token = str(tokenizer.decode([token_id])).upper()
                if len(token) != 3:
                    raise ValueError(
                        f"decoded non-codon token {token!r}"
                    )
                codons.append(token)
            if len(codons) != len(target):
                raise ValueError(
                    "generated codon count mismatch: "
                    f"{len(codons)} != {len(target)}"
                )
            sequences.append("".join(codons))
        return sequences


def run_codongpt_adapter(
    *,
    input_pack_summary: str,
    model_dir: str,
    executable: str,
    out_dir: str,
    limit: Optional[int] = None,
    batch_size: int = 64,
    seed: int = 0,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
    resume: bool = False,
    progress_jsonl: Optional[str] = None,
    batch_generator: Optional[BatchGenerator] = None,
) -> dict[str, object]:
    """Generate one synonym-constrained codonGPT candidate per input row."""
    start = time.perf_counter()
    input_pack = _load_json(input_pack_summary)
    cds_inputs_path = _resolve_pack_file(
        input_pack_summary,
        input_pack,
        "cds_protein_jsonl",
        "cds_protein_inputs.jsonl",
    )
    input_rows = _load_jsonl(cds_inputs_path)
    selected = input_rows[:limit] if limit is not None else input_rows
    reference_records = [
        MRNARecord(
            str(row.get("transcript_id")),
            "",
            _normalise_rna(row.get("native_cds")),
            "",
        )
        for row in selected
    ]
    weights = codon_weights_from_reference(reference_records)
    os.makedirs(out_dir, exist_ok=True)
    outputs_path = os.path.join(out_dir, "cds_outputs.jsonl")
    failures_path = os.path.join(out_dir, "failures.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    if progress_jsonl is None:
        progress_jsonl = os.path.join(out_dir, "progress.jsonl")
    if not resume:
        for path in (outputs_path, failures_path, progress_jsonl):
            if os.path.exists(path):
                os.remove(path)
    existing = _load_jsonl(outputs_path) if resume and os.path.exists(outputs_path) else []
    outputs_by_id = {
        str(row.get("transcript_id")): row
        for row in existing
    }
    failures: list[dict[str, object]] = []
    runtime: Optional[_CodonGPTRuntime] = None
    if batch_generator is None:
        runtime = _CodonGPTRuntime(model_dir, device)

        def generate_batch(
            rows: Sequence[Mapping[str, object]],
            batch_seed: int,
        ) -> Sequence[str]:
            assert runtime is not None
            return runtime.generate(
                rows,
                batch_seed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        batch_generator = generate_batch

    _append_jsonl(
        {
            "event": "start",
            "time": time.time(),
            "n_inputs": len(selected),
            "n_resumed": len(outputs_by_id),
            "batch_size": int(batch_size),
            "seed": int(seed),
        },
        progress_jsonl,
    )
    for batch_start in range(0, len(selected), max(1, int(batch_size))):
        original_batch = selected[
            batch_start : batch_start + max(1, int(batch_size))
        ]
        batch = [
            row
            for row in original_batch
            if str(row.get("transcript_id")) not in outputs_by_id
        ]
        if not batch:
            continue
        batch_started = time.perf_counter()
        try:
            generated = list(
                batch_generator(batch, int(seed) + batch_start)
            )
            if len(generated) != len(batch):
                raise ValueError(
                    f"batch output count {len(generated)} != {len(batch)}"
                )
            amortized_s = float(
                (time.perf_counter() - batch_started) / len(batch)
            )
            pending_outputs: list[tuple[str, dict[str, object]]] = []
            for row, designed_cds in zip(batch, generated):
                scored = _score_generated_cds(
                    row,
                    designed_cds,
                    codon_weights=weights,
                    wall_clock_s=amortized_s,
                )
                if (
                    scored["valid_cds"] is not True
                    or scored["protein_identity_exact_1"] is not True
                ):
                    raise ValueError(
                        "generated CDS failed exact hard constraints for "
                        f"{row.get('transcript_id')}"
                    )
                pending_outputs.append(
                    (str(row.get("transcript_id")), scored)
                )
            for transcript_id, scored in pending_outputs:
                outputs_by_id[transcript_id] = scored
        except Exception as exc:
            for row in batch:
                failures.append(
                    {
                        "record_index": row.get("record_index"),
                        "transcript_id": row.get("transcript_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        ordered_outputs = [
            outputs_by_id[str(row.get("transcript_id"))]
            for row in selected
            if str(row.get("transcript_id")) in outputs_by_id
        ]
        _write_jsonl(ordered_outputs, outputs_path)
        _write_jsonl(failures, failures_path)
        _append_jsonl(
            {
                "event": "batch_complete",
                "time": time.time(),
                "batch_start": batch_start,
                "batch_rows": len(batch),
                "n_outputs": len(ordered_outputs),
                "n_failures": len(failures),
            },
            progress_jsonl,
        )

    outputs = [
        outputs_by_id[str(row.get("transcript_id"))]
        for row in selected
        if str(row.get("transcript_id")) in outputs_by_id
    ]
    elapsed_s = float(time.perf_counter() - start)

    def mean(key: str) -> float:
        values = [
            float(row[key])
            for row in outputs
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
        ]
        return float(sum(values) / len(values)) if values else 0.0

    input_pack_outputs = input_pack.get("outputs", {})
    input_pack_outputs = (
        input_pack_outputs
        if isinstance(input_pack_outputs, Mapping)
        else {}
    )
    dataset = input_pack.get("dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    model_manifest = (
        _load_json(os.path.join(model_dir, "model_manifest.json"))
        if os.path.exists(os.path.join(model_dir, "model_manifest.json"))
        else {}
    )
    hardware: dict[str, object] = {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "requested_device": device,
    }
    if runtime is not None and runtime.device.type == "cuda":
        hardware["cuda_device_name"] = runtime.torch.cuda.get_device_name(
            runtime.device
        )
    summary: dict[str, object] = {
        "artifact_kind": "external_sota_real_run_summary",
        "claim_policy": CLAIM_POLICY,
        "model_name": MODEL_NAME,
        "task_family": TASK_FAMILY,
        "protocol_fidelity": (
            "official_hf_pretrained_checkpoint_synonymous_masked_sampling"
        ),
        "protocol_fidelity_sufficient_for_sota_reproduction": False,
        "protocol_limitations": [
            "Public HF checkpoint is pretrained codonGPT, not a released task-specific RL policy.",
            "One sampled candidate is generated per target protein.",
            "Metrics are offline sequence proxies, not expression measurements.",
        ],
        "input_pack": {
            "summary_sha256": _sha256_file(input_pack_summary),
            "cds_protein_jsonl_sha256": input_pack_outputs.get(
                "cds_protein_jsonl_sha256"
            ),
            "utr5_jsonl_sha256": input_pack_outputs.get(
                "utr5_jsonl_sha256"
            ),
        },
        "dataset": {
            "records_jsonl_sha256": dataset.get("records_jsonl_sha256"),
            "split_name": dataset.get("split_name"),
            "seed": dataset.get("seed"),
        },
        "runtime": {
            "elapsed_s": elapsed_s,
            "batch_size": int(batch_size),
            "seed": int(seed),
            "resume": bool(resume),
        },
        "hardware": hardware,
        "executable": {
            "path": os.path.abspath(executable),
            "version": _executable_version(executable),
            "sha256": _sha256_file(executable),
        },
        "checkpoint": {
            "hf_repo": model_manifest.get("hf_repo", DEFAULT_HF_REPO),
            "hf_revision": model_manifest.get(
                "hf_revision",
                DEFAULT_HF_REVISION,
            ),
            "model_dir": os.path.abspath(model_dir),
            "model_manifest_sha256": _sha256_file(
                os.path.join(model_dir, "model_manifest.json")
            ),
            "weights_sha256": _sha256_file(
                os.path.join(model_dir, "pytorch_model.bin")
            ),
            "license": model_manifest.get(
                "license",
                "free_for_research_use_model_card",
            ),
            "redistribution_rights_assumed": False,
        },
        "sampling": {
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "synonymous_mask": True,
            "terminal_stop_generated_under_mask": True,
        },
        "n_inputs": len(selected),
        "n_outputs": len(outputs),
        "n_failures": len(selected) - len(outputs),
        "mean_wall_clock_s": mean("wall_clock_s"),
        "valid_cds_fraction": (
            float(sum(bool(row.get("valid_cds")) for row in outputs) / len(outputs))
            if outputs
            else 0.0
        ),
        "protein_identity_exact_1_fraction": (
            float(
                sum(
                    bool(row.get("protein_identity_exact_1"))
                    for row in outputs
                )
                / len(outputs)
            )
            if outputs
            else 0.0
        ),
        "mean_cai": mean("cai"),
        "mean_gc": mean("gc"),
        "mean_gc3": mean("gc3"),
        "mean_codon_accuracy_vs_native": mean(
            "codon_accuracy_vs_native"
        ),
        "mean_codon_usage_kl_vs_native": mean(
            "codon_usage_kl_vs_native"
        ),
        "mean_codon_pair_kl_vs_native": mean(
            "codon_pair_kl_vs_native"
        ),
        "outputs": {
            "cds_outputs_jsonl": outputs_path,
            "cds_outputs_sha256": _sha256_file(outputs_path),
            "failures_jsonl": failures_path,
            "failures_sha256": _sha256_file(failures_path),
            "progress_jsonl": progress_jsonl,
        },
    }
    _write_json(summary, summary_path)
    _append_jsonl(
        {
            "event": "complete",
            "time": time.time(),
            "n_inputs": len(selected),
            "n_outputs": len(outputs),
            "n_failures": len(selected) - len(outputs),
            "elapsed_s": elapsed_s,
        },
        progress_jsonl,
    )
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack-summary", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-jsonl", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_codongpt_adapter(
        input_pack_summary=args.input_pack_summary,
        model_dir=args.model_dir,
        executable=args.executable,
        out_dir=args.out_dir,
        limit=args.limit,
        batch_size=args.batch_size,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        resume=args.resume,
        progress_jsonl=args.progress_jsonl,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "DEFAULT_HF_REPO",
    "DEFAULT_HF_REVISION",
    "run_codongpt_adapter",
    "main",
]
