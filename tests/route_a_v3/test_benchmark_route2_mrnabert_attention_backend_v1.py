import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "route_a_v3"
    / "benchmark_route2_mrnabert_attention_backend_v1.py"
)
SPEC = importlib.util.spec_from_file_location("attention_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_alibi_slopes_match_expected_head_count_and_are_decreasing():
    slopes = MODULE.alibi_slopes(12)
    assert len(slopes) == 12
    assert all(value > 0 for value in slopes)
    assert len(set(slopes)) == 12


def test_cpu_sdpa_math_matches_official_manual_attention():
    torch.manual_seed(17)
    batch, heads, length, dimension = 2, 4, 7, 8
    query = torch.randn(batch, heads, length, dimension)
    key = torch.randn(batch, heads, length, dimension)
    value = torch.randn(batch, heads, length, dimension)
    valid_lengths = torch.tensor([7, 5])
    bias, query_is_valid = MODULE.build_alibi_attention_bias(
        sequence_length=length,
        valid_lengths=valid_lengths,
        number_of_heads=heads,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    reference = MODULE.official_manual_attention(query, key, value, bias)
    candidate = MODULE.sdpa_attention(query, key, value, bias, "MATH")
    metrics = MODULE.error_metrics(reference, candidate, query_is_valid)
    assert metrics["maximum_absolute_difference"] < 1e-5
    assert metrics["mean_absolute_difference"] < 1e-6
    assert metrics["cosine_similarity"] > 0.999999


def test_validation_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown attention backends"):
        MODULE.validate_config(
            {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "seed": 1,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "sequence_lengths": [50],
                "batch_sizes": [1],
                "warmup_iterations": 1,
                "measured_iterations": 1,
                "candidate_backends": ["NOT_A_BACKEND"],
                "screening_tolerances": {},
                "output_path": "/tmp/report.json",
            }
        )
