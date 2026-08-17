#!/usr/bin/env python3
"""Screen PyTorch SDPA backends against mRNABERT's official ALiBi attention.

This is an operator-level benchmark.  It neither loads project records nor
changes the frozen encoder.  A successful result is only permission to run a
full-encoder cache-alignment benchmark later; it is not an automatic backend
switch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


BACKEND_NAMES = {
    "FLASH_ATTENTION": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    "EFFICIENT_ATTENTION": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    "MATH": torch.nn.attention.SDPBackend.MATH,
}


def alibi_slopes(number_of_heads: int) -> list[float]:
    """Reproduce the slope construction used by the official mRNABERT code."""

    def power_of_two_slopes(heads: int) -> list[float]:
        start = 2 ** (-2 ** -(math.log2(heads) - 3))
        return [start * start**index for index in range(heads)]

    if math.log2(number_of_heads).is_integer():
        return power_of_two_slopes(number_of_heads)
    closest_power = 2 ** math.floor(math.log2(number_of_heads))
    first = power_of_two_slopes(closest_power)
    second = alibi_slopes(2 * closest_power)[0::2][
        : number_of_heads - closest_power
    ]
    return first + second


def build_alibi_attention_bias(
    *,
    sequence_length: int,
    valid_lengths: torch.Tensor,
    number_of_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence_length, device=device)
    relative = torch.abs(positions[None, :] - positions[:, None])
    slopes = torch.tensor(
        alibi_slopes(number_of_heads), dtype=torch.float32, device=device
    )
    alibi = -slopes[:, None, None] * relative[None, :, :]
    key_is_valid = positions[None, :] < valid_lengths[:, None]
    padding_bias = (~key_is_valid)[:, None, None, :].to(torch.float32) * -10000.0
    bias = (alibi[None, :, :, :] + padding_bias).to(dtype)
    query_is_valid = positions[None, :] < valid_lengths[:, None]
    return bias, query_is_valid


def official_manual_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    probabilities = torch.softmax(scores + bias, dim=-1)
    return torch.matmul(probabilities, value)


def sdpa_context(backend_name: str):
    if backend_name == "AUTO":
        return nullcontext()
    return torch.nn.attention.sdpa_kernel(BACKEND_NAMES[backend_name])


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    backend_name: str,
) -> torch.Tensor:
    with sdpa_context(backend_name):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=0.0,
            is_causal=False,
        )


def error_metrics(
    reference: torch.Tensor, candidate: torch.Tensor, query_is_valid: torch.Tensor
) -> dict[str, float]:
    mask = query_is_valid[:, None, :, None].expand_as(reference)
    reference_values = reference[mask].float()
    candidate_values = candidate[mask].float()
    absolute = (reference_values - candidate_values).abs()
    cosine = F.cosine_similarity(
        reference_values.reshape(1, -1), candidate_values.reshape(1, -1)
    ).item()
    return {
        "maximum_absolute_difference": absolute.max().item(),
        "mean_absolute_difference": absolute.mean().item(),
        "cosine_similarity": cosine,
    }


def benchmark_callable(
    callable_object,
    *,
    warmup_iterations: int,
    measured_iterations: int,
    device: torch.device,
) -> float:
    for _ in range(warmup_iterations):
        callable_object()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(measured_iterations):
        callable_object()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / measured_iterations


def detect_auto_backend(callable_object) -> dict[str, Any]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        callable_object()
        torch.cuda.synchronize()
    event_names = sorted({event.key for event in profile.key_averages()})
    joined = "\n".join(event_names).lower()
    if "flash_attention" in joined:
        selected = "FLASH_ATTENTION"
    elif "efficient_attention" in joined or "mem_efficient" in joined:
        selected = "EFFICIENT_ATTENTION"
    elif "scaled_dot_product_attention_math" in joined:
        selected = "MATH"
    else:
        selected = "OTHER_OR_UNRESOLVED"
    relevant = [
        name
        for name in event_names
        if "attention" in name.lower() or "scaled_dot" in name.lower()
    ]
    return {"selected_backend": selected, "relevant_profiler_events": relevant}


def screening_passed(metrics: dict[str, float], tolerances: dict[str, float]) -> bool:
    return (
        metrics["maximum_absolute_difference"]
        <= tolerances["maximum_absolute_difference"]
        and metrics["mean_absolute_difference"]
        <= tolerances["mean_absolute_difference"]
        and metrics["cosine_similarity"] >= tolerances["minimum_cosine_similarity"]
    )


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "device",
        "dtype",
        "seed",
        "hidden_size",
        "num_attention_heads",
        "sequence_lengths",
        "batch_sizes",
        "warmup_iterations",
        "measured_iterations",
        "candidate_backends",
        "screening_tolerances",
        "output_path",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if config["hidden_size"] % config["num_attention_heads"]:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    unknown = sorted(set(config["candidate_backends"]) - {"AUTO", *BACKEND_NAMES})
    if unknown:
        raise ValueError(f"unknown attention backends: {unknown}")


def run(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this attention backend benchmark")
    device = torch.device(config["device"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    heads = int(config["num_attention_heads"])
    head_dimension = int(config["hidden_size"]) // heads
    tolerances = config["screening_tolerances"]
    rows: list[dict[str, Any]] = []

    for batch_size in config["batch_sizes"]:
        for sequence_length in config["sequence_lengths"]:
            valid_lengths = torch.tensor(
                [max(1, sequence_length - index * 3) for index in range(batch_size)],
                dtype=torch.long,
                device=device,
            )
            qkv = torch.randn(
                batch_size,
                sequence_length,
                3,
                heads,
                head_dimension,
                device=device,
                dtype=dtype,
            )
            query = qkv[:, :, 0].permute(0, 2, 1, 3)
            key = qkv[:, :, 1].permute(0, 2, 1, 3)
            value = qkv[:, :, 2].permute(0, 2, 1, 3)
            bias, query_is_valid = build_alibi_attention_bias(
                sequence_length=sequence_length,
                valid_lengths=valid_lengths,
                number_of_heads=heads,
                dtype=dtype,
                device=device,
            )
            reference_callable = lambda: official_manual_attention(query, key, value, bias)
            with torch.inference_mode():
                reference = reference_callable()
                reference_ms = benchmark_callable(
                    reference_callable,
                    warmup_iterations=config["warmup_iterations"],
                    measured_iterations=config["measured_iterations"],
                    device=device,
                )
                for backend_name in config["candidate_backends"]:
                    candidate_callable = lambda name=backend_name: sdpa_attention(
                        query, key, value, bias, name
                    )
                    row: dict[str, Any] = {
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "backend": backend_name,
                        "reference_milliseconds": reference_ms,
                    }
                    try:
                        candidate = candidate_callable()
                        metrics = error_metrics(reference, candidate, query_is_valid)
                        candidate_ms = benchmark_callable(
                            candidate_callable,
                            warmup_iterations=config["warmup_iterations"],
                            measured_iterations=config["measured_iterations"],
                            device=device,
                        )
                        row.update(metrics)
                        row.update(
                            {
                                "status": "SUPPORTED",
                                "screening_passed": screening_passed(metrics, tolerances),
                                "candidate_milliseconds": candidate_ms,
                                "speedup_over_official_manual": reference_ms / candidate_ms,
                            }
                        )
                        if backend_name == "AUTO" and batch_size == max(config["batch_sizes"]) and sequence_length == max(config["sequence_lengths"]):
                            row["auto_backend_detection"] = detect_auto_backend(candidate_callable)
                    except RuntimeError as error:
                        row.update(
                            {
                                "status": "UNSUPPORTED",
                                "screening_passed": False,
                                "error": str(error).splitlines()[0],
                            }
                        )
                    rows.append(row)

    supported_auto = [
        row for row in rows if row["backend"] == "AUTO" and row["status"] == "SUPPORTED"
    ]
    all_auto_equivalent = bool(supported_auto) and all(
        row["screening_passed"] for row in supported_auto
    )
    auto_speedups = [row["speedup_over_official_manual"] for row in supported_auto]
    report = {
        "schema_version": "route_a_v3_route2_mrnabert_attention_backend_benchmark.v1",
        "benchmark_id": config.get("benchmark_id"),
        "recorded_at_unix": time.time(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "official_position_encoding": "BIDIRECTIONAL_ALIBI",
        "official_fallback": "EXPLICIT_QK_SOFTMAX_V",
        "rows": rows,
        "summary": {
            "all_auto_shapes_numerically_equivalent": all_auto_equivalent,
            "minimum_auto_speedup": min(auto_speedups) if auto_speedups else None,
            "median_auto_speedup": (
                sorted(auto_speedups)[len(auto_speedups) // 2] if auto_speedups else None
            ),
            "maximum_auto_speedup": max(auto_speedups) if auto_speedups else None,
            "decision": (
                "ELIGIBLE_FOR_FULL_ENCODER_CACHE_ALIGNMENT_BENCHMARK"
                if all_auto_equivalent
                else "KEEP_OFFICIAL_PYTORCH_FALLBACK"
            ),
            "formal_encoder_backend_changed": False,
        },
        "activation_policy": config.get("activation_policy"),
        "required_followup_before_switch": config.get("required_followup_before_switch"),
    }
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).read_text())
    report = run(config)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
