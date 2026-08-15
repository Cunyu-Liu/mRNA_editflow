#!/usr/bin/env python3
"""Future exactly-one DEC028 source-relative critic implementation.

The checked-in candidate is inactive.  ``--validate-only`` inspects static
configuration and exits before data, PyTorch, CUDA, model, optimizer, checkpoint
or output access.  A later config-only activation must provide all five frozen
activation requirements before ``--run`` can reach the private input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_gse200304_source_relative_critic_g1_implementation_candidate_v1.json"
IMPLEMENTATION_ID = "ROUTE_A_V3_GSE200304_SOURCE_RELATIVE_CRITIC_G1_IMPLEMENTATION_CANDIDATE_V1"
INACTIVE = "INACTIVE_FAIL_BEFORE_DATA_MODEL_CUDA_OUTPUT"
ACTIVE = "ACTIVE_FOR_THIS_G1_ONE_RUN_ONLY"


class ContractError(RuntimeError):
    pass


class InactiveAuthorityError(ContractError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load implementation config: {path}") from exc
    if type(value) is not dict:
        raise ContractError("implementation config root must be an object")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("implementation_id") != IMPLEMENTATION_ID:
        raise ContractError("implementation ID differs")
    if config.get("decision_id") != "V3-DEC-028" or config.get("runtime_event_id") != "A1-EVT-061":
        raise ContractError("authority context differs")
    if config.get("run_role") != "GSE200304_SOURCE_RELATIVE_CRITIC_G1":
        raise ContractError("run role differs")
    if config.get("activation_state") not in {INACTIVE, ACTIVE}:
        raise ContractError("activation state is invalid")

    inputs = config["input_contract"]
    if inputs["study_unit"] != "GSE200304_SUPERSERIES_ONE_STUDY" or inputs["required_member_count"] != 6547:
        raise ContractError("input membership contract differs")
    if inputs["missing_or_nonfinite_policy"] != "REJECT_NEVER_IMPUTE_ZERO":
        raise ContractError("missing/nonfinite policy differs")
    if inputs["one_source_group_one_vote"] is not True:
        raise ContractError("source-group weighting differs")

    model = config["model_contract"]
    if model["fixed_prefix_truncation_allowed"] is not False or model["full_length_dynamic_padding_required"] is not True:
        raise ContractError("model can hide supported edits")
    if model["mean_construction"] != "HALF_FORWARD_MINUS_REVERSE_PAIR_SCORE":
        raise ContractError("antisymmetric mean construction differs")
    if model["random_initialization_only"] is not True or model["external_learned_input_count"] != 0:
        raise ContractError("scratch-only model route differs")
    if model["deterministic_algorithms_required"] is not True:
        raise ContractError("deterministic algorithm policy differs")
    if model["flash_attention_allowed"] is not False or model["memory_efficient_attention_allowed"] is not False:
        raise ContractError("nondeterministic attention backend is enabled")

    single = config["single_fit_contract"]
    expected_counts = {
        "authorized_execution_count": 1,
        "optimizer_fit_count": 1,
        "fold_model_count": 1,
        "checkpoint_count": 1,
        "final_refit_count": 0,
        "seed_count": 1,
    }
    for key, expected in expected_counts.items():
        if single[key] != expected:
            raise ContractError(f"single-fit count differs: {key}")
    for key in (
        "early_stopping_allowed",
        "best_checkpoint_selection_allowed",
        "hyperparameter_search_allowed",
        "automatic_retry_allowed",
    ):
        if single[key] is not False:
            raise ContractError(f"selection/retry is enabled: {key}")
    if single["terminal_checkpoint_only"] is not True:
        raise ContractError("terminal checkpoint policy differs")

    if config["gate_bundle"]["any_nonpass_action"] != "STOP_WITH_EVIDENCE_NO_RETRY":
        raise ContractError("gate failure action differs")
    evaluator = config["evaluator_and_baseline_contract"]
    if evaluator["fit_role"] != "TRAIN" or evaluator["calibration_role"] != "CALIBRATION":
        raise ContractError("baseline fit or calibration role differs")
    if evaluator["terminal_evaluation_role"] != "TEST":
        raise ContractError("terminal evaluator role differs")
    if evaluator["one_source_group_one_vote"] is not True:
        raise ContractError("evaluator source-group weighting differs")
    if evaluator["primary_metric"] != "WITHIN_STUDY_SOURCE_GROUP_EQUAL_WEIGHT_SPEARMAN":
        raise ContractError("primary evaluator metric differs")
    if evaluator["baseline_set"] != [
        "TRAIN_SOURCE_GROUP_EQUAL_WEIGHT_GLOBAL_MEAN",
        "TRAIN_DIRECTED_EDIT_TYPE_MEAN",
        "TRAIN_GC_AND_LENGTH_LINEAR_RIDGE_FIXED_ALPHA_1",
        "TRAIN_15MER_COUNT_RIDGE_FIXED_ALPHA_10",
    ]:
        raise ContractError("frozen baseline set differs")
    if evaluator["kmer_length"] != 15 or evaluator["kmer_feature_dimension"] != 4096:
        raise ContractError("15-mer baseline geometry differs")
    if evaluator["kmer_feature_map"] != "FNV1A64_SIGNED_FEATURE_HASH_OF_CANDIDATE_MINUS_SOURCE_15MER_COUNTS":
        raise ContractError("15-mer feature map differs")
    if evaluator["guide_or_model_selection_output_allowed"] is not False or evaluator["test_feedback_allowed"] is not False:
        raise ContractError("evaluator isolation differs")
    input_contract = config["input_contract"]
    expected_definitions = {
        "context_vector_definition": "SIXTEEN_CONTIGUOUS_201NT_POSITION_BINS_TIMES_ACGT_PAIR_MEAN_FRACTIONS_SOURCE_CANDIDATE_SWAP_INVARIANT",
        "edit_feature_definition": "THREE_POSITIONS_CENTER_MINUS_ONE_CENTER_CENTER_PLUS_ONE_TIMES_ACGT_CANDIDATE_ONEHOT_MINUS_SOURCE_ONEHOT",
        "effect_definition": "MEAN_OVER_SIX_PAIRED_BIOLOGICAL_REPLICATES_OF_LOG2_SUM_HIGH_LOW_MINUS_TOTAL_RNA_MUTANT_MINUS_WT",
        "standard_error_definition": "SAMPLE_STANDARD_DEVIATION_OF_SIX_PAIRED_REPLICATE_DELTAS_DIVIDED_BY_SQRT_SIX_FINITE_POSITIVE",
    }
    for key, expected in expected_definitions.items():
        if input_contract.get(key) != expected:
            raise ContractError(f"row-contract definition differs: {key}")
    requirements = config["future_activation_requirements"]
    required_count = sum(
        bool(requirements[key])
        for key in (
            "separate_run_authority_required",
            "materialization_conformance_pass_required",
            "real_split_evaluator_baseline_pass_required",
            "implementation_and_gate_review_pass_required",
            "cuda_owner_device_binding_required",
        )
    )
    if required_count != 5:
        raise ContractError("future activation requirement set differs")
    if config["activation_state"] == INACTIVE and requirements["current_requirement_count_satisfied"] != 0:
        raise ContractError("inactive candidate claims satisfied activation requirements")
    if config["activation_state"] == ACTIVE and requirements["current_requirement_count_satisfied"] != 5:
        raise ContractError("active run is missing activation requirements")
    activation = config["activation_binding"]
    expected_activation_keys = {
        "run_authority_id",
        "materialization_conformance_id",
        "materialized_rows_path",
        "materialized_rows_bytes",
        "materialized_rows_sha256",
        "split_evaluator_baseline_id",
        "split_assignments_path",
        "split_assignments_bytes",
        "split_assignments_sha256",
        "implementation_gate_review_id",
        "implementation_commit",
        "cuda_device",
        "cuda_physical_index",
        "cuda_uuid",
        "python_executable",
        "torch_version",
        "torch_cuda_version",
        "output_directory",
    }
    if set(activation) != expected_activation_keys:
        raise ContractError("activation binding closure differs")
    if config["activation_state"] == INACTIVE:
        if any(value is not None for value in activation.values()):
            raise ContractError("inactive implementation has activation bindings")
    else:
        integer_keys = {"materialized_rows_bytes", "split_assignments_bytes", "cuda_physical_index"}
        if any(
            (not isinstance(value, int) or isinstance(value, bool) or value < 0)
            if key in integer_keys
            else (not isinstance(value, str) or not value)
            for key, value in activation.items()
        ):
            raise ContractError("active implementation has incomplete activation bindings")
        for key in ("materialized_rows_sha256", "split_assignments_sha256"):
            value = activation[key]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ContractError(f"active file digest is invalid: {key}")
        implementation = activation["implementation_commit"]
        if len(implementation) != 40 or any(character not in "0123456789abcdef" for character in implementation):
            raise ContractError("reviewed implementation commit is invalid")
        for key in ("materialized_rows_path", "split_assignments_path", "python_executable", "output_directory"):
            if not Path(activation[key]).is_absolute():
                raise ContractError(f"active path is not absolute: {key}")
        if not activation["cuda_uuid"].startswith("GPU-"):
            raise ContractError("active CUDA UUID is invalid")

    truth = config["current_truth"]
    for key, value in truth.items():
        if key in {"scientific_claim_status"}:
            if value != "NOT_ESTABLISHED":
                raise ContractError("scientific claim was promoted")
        elif isinstance(value, bool):
            if value is not False:
                raise ContractError(f"inactive truth is not false: {key}")
        elif value != 0:
            raise ContractError(f"inactive truth is not zero: {key}")


def validate_only(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "status": (
            "PASS_ACTIVE_EXACTLY_ONE_RUN_AUTHORITY_STATIC_VALIDATION_NOT_RUN"
            if config["activation_state"] == ACTIVE
            else "PASS_STATIC_IMPLEMENTATION_CONTRACT_NOT_ACTIVE_NOT_RUN"
        ),
        "activation_state": config["activation_state"],
        "data_rows_read": 0,
        "model_constructions": 0,
        "cuda_touches": 0,
        "parameter_updates": 0,
        "outputs_written": 0,
    }


def require_active_before_operational_io(config: Mapping[str, Any]) -> None:
    validate_config(config)
    if config["activation_state"] != ACTIVE:
        raise InactiveAuthorityError(
            "critic implementation is inactive; stop before data, model, CUDA and output"
        )


def _git(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ContractError(f"repository audit failed: {' '.join(args)}")
    return result.stdout.strip()


def _repository_audit(config: Mapping[str, Any], repository_root: Path) -> None:
    """Bind the active config-only authority to the reviewed implementation."""

    activation = config["activation_binding"]
    expected_script = repository_root / "scripts/route_a_v3/gse200304_source_relative_critic_g1.py"
    if Path(__file__).resolve() != expected_script.resolve():
        raise ContractError("executing critic script is outside the bound repository path")
    if _git(repository_root, "status", "--porcelain"):
        raise ContractError("repository worktree is not clean")
    head = _git(repository_root, "rev-parse", "HEAD")
    if head != _git(repository_root, "rev-parse", "@{u}"):
        raise ContractError("HEAD differs from its configured upstream")
    implementation = activation["implementation_commit"]
    if _git(repository_root, "rev-parse", "HEAD^") != implementation:
        raise ContractError("active run authority is not the direct child of the reviewed implementation")
    changed = _git(repository_root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    if changed != ["configs/route_a_v3_gse200304_source_relative_critic_g1_implementation_candidate_v1.json"]:
        raise ContractError("active run authority is not config-only")
    reviewed_script = subprocess.run(
        ["git", "-C", str(repository_root), "show", f"{implementation}:scripts/route_a_v3/gse200304_source_relative_critic_g1.py"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if reviewed_script.returncode != 0 or reviewed_script.stdout != expected_script.read_bytes():
        raise ContractError("executing critic bytes differ from the reviewed implementation")


def _cuda_binding_audit(config: Mapping[str, Any]) -> None:
    activation = config["activation_binding"]
    index = str(activation["cuda_physical_index"])
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != activation["cuda_uuid"]:
        raise ContractError("bound CUDA physical device identity differs")
    if activation["cuda_device"] != f"cuda:{index}":
        raise ContractError("logical and physical CUDA bindings differ")
    if Path(sys.executable).resolve() != Path(activation["python_executable"]).resolve():
        raise ContractError("Python runtime differs from active authority")


def _read_bound_file(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read bound private input: {path.name}") from exc
    if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ContractError(f"bound private input identity differs: {path.name}")
    return payload


def _load_rows_and_split(
    config: Mapping[str, Any],
    rows_payload: Optional[bytes] = None,
    split_payload: Optional[bytes] = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Private input reader, reachable only after active authority."""

    input_contract = config["input_contract"]
    activation = config["activation_binding"]
    if rows_payload is None:
        rows_payload = _read_bound_file(
            Path(activation["materialized_rows_path"]),
            activation["materialized_rows_bytes"],
            activation["materialized_rows_sha256"],
        )
    if split_payload is None:
        split_payload = _read_bound_file(
            Path(activation["split_assignments_path"]),
            activation["split_assignments_bytes"],
            activation["split_assignments_sha256"],
        )
    try:
        rows = [json.loads(line) for line in rows_payload.decode("utf-8").splitlines() if line]
        split = json.loads(split_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("bound private input is not valid JSON") from exc
    if len(rows) != input_contract["required_member_count"]:
        raise ContractError("materialized row count differs")
    expected_keys = set(input_contract["row_schema_keys_exactly"])
    seen: set[str] = set()
    for row in rows:
        if set(row) != expected_keys:
            raise ContractError("materialized row schema differs")
        key = row["record_key"]
        if not isinstance(key, str) or not key or key in seen:
            raise ContractError("record key is missing or duplicated")
        seen.add(key)
        if len(row["context_vector"]) != input_contract["context_vector_width"]:
            raise ContractError("context vector width differs")
        if len(row["edit_features"]) != input_contract["edit_feature_width"]:
            raise ContractError("edit feature width differs")
        for field in ("direction_normalized_effect", "biological_standard_error"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContractError(f"{field} is missing or nonfinite")
        if float(row["biological_standard_error"]) <= 0.0:
            raise ContractError("biological standard error must be positive")
    if set(split) != seen or set(split.values()) != set(input_contract["split_roles_exactly"]):
        raise ContractError("split assignment universe or roles differ")
    return rows, split


def _torch_components(config: Mapping[str, Any]):
    """Create the model type only after active authority and input closure."""

    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    model_cfg = config["model_contract"]
    hidden = int(model_cfg["hidden_width"])

    class FullLengthEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(6, hidden, padding_idx=5)
            self.local = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=4)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=int(model_cfg["attention_heads"]),
                dim_feedforward=int(model_cfg["feed_forward_width"]),
                dropout=float(model_cfg["dropout"]),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(model_cfg["encoder_layers"]))
            self.norm = nn.LayerNorm(hidden)

        @staticmethod
        def _position(length: int, device, dtype):
            positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
            frequencies = torch.exp(
                torch.arange(0, hidden, 2, device=device, dtype=dtype)
                * (-math.log(10000.0) / hidden)
            )
            values = torch.zeros(length, hidden, device=device, dtype=dtype)
            values[:, 0::2] = torch.sin(positions * frequencies)
            values[:, 1::2] = torch.cos(positions * frequencies)
            return values.unsqueeze(0)

        def forward(self, tokens, mask):
            x = self.embedding(tokens) + self._position(tokens.shape[1], tokens.device, self.embedding.weight.dtype)
            x = x + self.local(x.transpose(1, 2)).transpose(1, 2)
            x = self.encoder(x, src_key_padding_mask=~mask)
            x = self.norm(x) * mask.unsqueeze(-1)
            return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    class SourceRelativeCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = FullLengthEncoder()
            pair_width = hidden + int(model_cfg["edit_feature_width"] if "edit_feature_width" in model_cfg else 12) + 64
            self.pair = nn.Sequential(
                nn.Linear(pair_width, hidden), nn.GELU(), nn.Dropout(float(model_cfg["dropout"])), nn.Linear(hidden, 2)
            )

        def _raw(self, source_tokens, source_mask, candidate_tokens, candidate_mask, edit_features, context_vector):
            source = self.encoder(source_tokens, source_mask)
            candidate = self.encoder(candidate_tokens, candidate_mask)
            return self.pair(torch.cat([candidate - source, edit_features, context_vector], dim=-1))

        def forward(self, source_tokens, source_mask, candidate_tokens, candidate_mask, edit_features, context_vector):
            forward = self._raw(source_tokens, source_mask, candidate_tokens, candidate_mask, edit_features, context_vector)
            reverse = self._raw(candidate_tokens, candidate_mask, source_tokens, source_mask, -edit_features, context_vector)
            mean = 0.5 * (forward[:, 0] - reverse[:, 0])
            scale = torch.nn.functional.softplus(0.5 * (forward[:, 1] + reverse[:, 1]))
            scale = scale + float(model_cfg["predictive_scale_minimum"])
            return {"mean": mean, "scale": scale}

    return torch, SourceRelativeCritic


def _tokenize(sequence: str) -> list[int]:
    table = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
    if not isinstance(sequence, str) or not sequence:
        raise ContractError("sequence must be a nonempty string")
    try:
        return [table[base] for base in sequence.upper()]
    except KeyError as exc:
        raise ContractError(f"unsupported sequence base: {exc.args[0]}") from exc


def _batch_tensors(torch, batch: list[dict[str, Any]], device):
    source_tokens = [_tokenize(row["source_sequence"]) for row in batch]
    candidate_tokens = [_tokenize(row["candidate_sequence"]) for row in batch]
    max_length = max(max(map(len, source_tokens)), max(map(len, candidate_tokens)))

    def padded(values: list[list[int]]):
        tokens = torch.full((len(values), max_length), 5, dtype=torch.long)
        mask = torch.zeros((len(values), max_length), dtype=torch.bool)
        for index, sequence in enumerate(values):
            tokens[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[index, : len(sequence)] = True
        return tokens.to(device), mask.to(device)

    source, source_mask = padded(source_tokens)
    candidate, candidate_mask = padded(candidate_tokens)
    edit = torch.tensor([row["edit_features"] for row in batch], dtype=torch.float32, device=device)
    context = torch.tensor([row["context_vector"] for row in batch], dtype=torch.float32, device=device)
    effect = torch.tensor([row["direction_normalized_effect"] for row in batch], dtype=torch.float32, device=device)
    standard_error = torch.tensor([row["biological_standard_error"] for row in batch], dtype=torch.float32, device=device)
    return {
        "source_tokens": source,
        "source_mask": source_mask,
        "candidate_tokens": candidate,
        "candidate_mask": candidate_mask,
        "edit_features": edit,
        "context_vector": context,
    }, effect, standard_error


def _group_weights(torch, batch: list[dict[str, Any]], group_counts: Mapping[str, int], device):
    weights = torch.tensor(
        [1.0 / group_counts[row["source_group"]] for row in batch],
        dtype=torch.float32,
        device=device,
    )
    return weights


def _predict_rows(torch, model, rows: list[dict[str, Any]], batch_size: int, device):
    predictions: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            tensors, effect, standard_error = _batch_tensors(torch, batch, device)
            output = model(**tensors)
            for index, row in enumerate(batch):
                predictions.append(
                    {
                        "record_key": row["record_key"],
                        "source_group": row["source_group"],
                        "observed": float(effect[index].cpu()),
                        "standard_error": float(standard_error[index].cpu()),
                        "predicted_mean": float(output["mean"][index].cpu()),
                        "predicted_scale": float(output["scale"][index].cpu()),
                    }
                )
    return predictions


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _spearman(observed: list[float], predicted: list[float]) -> Optional[float]:
    if len(observed) < 2:
        return None
    left, right = _rank(observed), _rank(predicted)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 0.0 else None


def _group_means(predictions: list[dict[str, Any]], predicted_field: str) -> list[dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for item in predictions:
        group = item["source_group"]
        state = grouped.setdefault(group, {"count": 0.0, "observed": 0.0, "predicted": 0.0, "scale": 0.0})
        state["count"] += 1.0
        state["observed"] += float(item["observed"])
        state["predicted"] += float(item[predicted_field])
        state["scale"] += float(item.get("calibrated_scale", item["predicted_scale"]))
    result = []
    for state in grouped.values():
        count = state["count"]
        result.append(
            {
                "observed": state["observed"] / count,
                "predicted": state["predicted"] / count,
                "scale": state["scale"] / count,
            }
        )
    return result


def _calibration_line(predictions: list[dict[str, Any]]) -> tuple[float, float]:
    groups = _group_means(predictions, "predicted_mean")
    if len(groups) < 2:
        raise ContractError("calibration split has fewer than two source groups")
    x = [item["predicted"] for item in groups]
    y = [item["observed"] for item in groups]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0.0:
        raise ContractError("constant-rank calibration prediction")
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator
    intercept = y_mean - slope * x_mean
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ContractError("nonfinite calibration slope or intercept")
    return float(slope), float(intercept)


def _apply_calibration(predictions: list[dict[str, Any]], slope: float, intercept: float) -> None:
    for item in predictions:
        item["calibrated_mean"] = intercept + slope * float(item["predicted_mean"])
        item["calibrated_scale"] = max(abs(slope) * float(item["predicted_scale"]), 1e-6)


def _calibration_quantile(predictions: list[dict[str, Any]], quantile: float = 0.9) -> float:
    groups = _group_means(predictions, "calibrated_mean")
    scores = sorted(
        abs(item["observed"] - item["predicted"])
        / max(item["scale"], 1e-6)
        for item in groups
    )
    if not scores:
        raise ContractError("calibration split is empty")
    index = min(math.ceil((len(scores) + 1) * quantile) - 1, len(scores) - 1)
    return float(scores[max(index, 0)])


def _group_equal_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _group_means(predictions, "calibrated_mean")
    observed = [item["observed"] for item in groups]
    predicted = [item["predicted"] for item in groups]
    spearman = _spearman(observed, predicted)
    if spearman is None:
        raise ContractError("constant-rank terminal main prediction")
    return {
        "source_group_count": len(groups),
        "spearman": spearman,
        "mae": sum(abs(a - b) for a, b in zip(observed, predicted)) / len(groups),
    }


def _group_equal_interval_coverage(predictions: list[dict[str, Any]], quantile: float) -> float:
    by_group: dict[str, list[bool]] = {}
    for item in predictions:
        covered = abs(float(item["observed"]) - float(item["calibrated_mean"])) <= quantile * float(item["calibrated_scale"])
        by_group.setdefault(item["source_group"], []).append(covered)
    return sum(sum(values) / len(values) for values in by_group.values()) / len(by_group)


def _coverage_risk(
    calibration: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
    retained_fractions: list[float],
) -> list[dict[str, Any]]:
    calibration_groups = sorted(item["scale"] for item in _group_means(calibration, "calibrated_mean"))
    terminal_groups = _group_means(terminal, "calibrated_mean")
    result: list[dict[str, Any]] = []
    for fraction in retained_fractions:
        index = max(0, min(len(calibration_groups) - 1, math.ceil(len(calibration_groups) * fraction) - 1))
        threshold = calibration_groups[index]
        retained = [item for item in terminal_groups if item["scale"] <= threshold]
        result.append(
            {
                "target_retained_fraction": fraction,
                "calibration_scale_threshold": threshold,
                "terminal_retained_group_count": len(retained),
                "terminal_realized_coverage": len(retained) / len(terminal_groups),
                "terminal_mae": (
                    sum(abs(item["observed"] - item["predicted"]) for item in retained) / len(retained)
                    if retained else None
                ),
            }
        )
    return result


def _directed_edit_type(row: Mapping[str, Any]) -> str:
    source = row["source_sequence"].upper().replace("T", "U")
    candidate = row["candidate_sequence"].upper().replace("T", "U")
    if len(source) != len(candidate):
        raise ContractError("directed-edit baseline requires equal-length source and candidate")
    changes = [(a, b) for a, b in zip(source, candidate) if a != b]
    if len(changes) != 1:
        raise ContractError("directed-edit baseline requires exactly one substitution")
    return f"{changes[0][0]}>{changes[0][1]}"


def _group_row_weights(rows: list[dict[str, Any]]) -> list[float]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["source_group"]] = counts.get(row["source_group"], 0) + 1
    return [1.0 / counts[row["source_group"]] for row in rows]


def _gc_length_features(row: Mapping[str, Any]) -> list[float]:
    source = row["source_sequence"].upper().replace("T", "U")
    candidate = row["candidate_sequence"].upper().replace("T", "U")
    source_gc = sum(base in {"G", "C"} for base in source) / len(source)
    candidate_gc = sum(base in {"G", "C"} for base in candidate) / len(candidate)
    return [1.0, source_gc, candidate_gc, candidate_gc - source_gc, len(source) / 201.0, len(candidate) / 201.0, (len(candidate) - len(source)) / 201.0]


def _fnv1a64(value: str) -> int:
    state = 14695981039346656037
    for byte in value.encode("ascii"):
        state ^= byte
        state = (state * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return state


def _kmer_delta_features(row: Mapping[str, Any], k: int, dimension: int) -> dict[int, float]:
    source = row["source_sequence"].upper().replace("T", "U")
    candidate = row["candidate_sequence"].upper().replace("T", "U")
    if len(source) != len(candidate):
        raise ContractError("15-mer baseline requires equal-length source and candidate")
    changed = [index for index, (a, b) in enumerate(zip(source, candidate)) if a != b]
    if len(changed) != 1:
        raise ContractError("15-mer baseline requires exactly one substitution")
    position = changed[0]
    result: dict[int, float] = {}
    start_min = max(0, position - k + 1)
    start_max = min(position, len(source) - k)
    for start in range(start_min, start_max + 1):
        for sequence, coefficient in ((candidate, 1.0), (source, -1.0)):
            hashed = _fnv1a64(sequence[start : start + k])
            bucket = hashed % dimension
            sign = 1.0 if (hashed >> 63) == 0 else -1.0
            result[bucket] = result.get(bucket, 0.0) + coefficient * sign
    return {key: value for key, value in result.items() if value != 0.0}


def _ridge_fit(torch, features, outcomes, weights, alpha: float, device):
    matrix = torch.tensor(features, dtype=torch.float32, device=device)
    target = torch.tensor(outcomes, dtype=torch.float32, device=device)
    weight = torch.tensor(weights, dtype=torch.float32, device=device)
    gram = matrix.transpose(0, 1) @ (matrix * weight.unsqueeze(1))
    right = matrix.transpose(0, 1) @ (target * weight)
    penalty = torch.eye(matrix.shape[1], dtype=matrix.dtype, device=device) * alpha
    penalty[0, 0] = 0.0
    return torch.linalg.solve(gram + penalty, right)


def _fit_and_evaluate_baselines(torch, train_rows, test_rows, config, device) -> dict[str, Any]:
    evaluator = config["evaluator_and_baseline_contract"]
    weights = _group_row_weights(train_rows)
    outcomes = [float(row["direction_normalized_effect"]) for row in train_rows]
    weight_sum = sum(weights)
    global_mean = sum(w * y for w, y in zip(weights, outcomes)) / weight_sum

    edit_sums: dict[str, float] = {}
    edit_weights: dict[str, float] = {}
    for row, weight, outcome in zip(train_rows, weights, outcomes):
        key = _directed_edit_type(row)
        edit_sums[key] = edit_sums.get(key, 0.0) + weight * outcome
        edit_weights[key] = edit_weights.get(key, 0.0) + weight
    edit_means = {key: edit_sums[key] / edit_weights[key] for key in edit_sums}

    gc_beta = _ridge_fit(
        torch,
        [_gc_length_features(row) for row in train_rows],
        outcomes,
        weights,
        float(evaluator["gc_length_ridge_alpha"]),
        device,
    )

    dimension = int(evaluator["kmer_feature_dimension"])
    k = int(evaluator["kmer_length"])
    kmer_matrix = torch.zeros((len(train_rows), dimension + 1), dtype=torch.float32, device=device)
    kmer_matrix[:, 0] = 1.0
    for row_index, row in enumerate(train_rows):
        for bucket, value in _kmer_delta_features(row, k, dimension).items():
            kmer_matrix[row_index, bucket + 1] = value
    target = torch.tensor(outcomes, dtype=torch.float32, device=device)
    weight = torch.tensor(weights, dtype=torch.float32, device=device)
    gram = kmer_matrix.transpose(0, 1) @ (kmer_matrix * weight.unsqueeze(1))
    right = kmer_matrix.transpose(0, 1) @ (target * weight)
    penalty = torch.eye(dimension + 1, dtype=torch.float32, device=device) * float(evaluator["kmer_ridge_alpha"])
    penalty[0, 0] = 0.0
    kmer_beta = torch.linalg.solve(gram + penalty, right)

    predictions: dict[str, list[float]] = {name: [] for name in evaluator["baseline_set"]}
    with torch.no_grad():
        for row in test_rows:
            predictions[evaluator["baseline_set"][0]].append(global_mean)
            predictions[evaluator["baseline_set"][1]].append(edit_means.get(_directed_edit_type(row), global_mean))
            gc = torch.tensor(_gc_length_features(row), dtype=torch.float32, device=device)
            predictions[evaluator["baseline_set"][2]].append(float((gc @ gc_beta).cpu()))
            vector = torch.zeros(dimension + 1, dtype=torch.float32, device=device)
            vector[0] = 1.0
            for bucket, value in _kmer_delta_features(row, k, dimension).items():
                vector[bucket + 1] = value
            predictions[evaluator["baseline_set"][3]].append(float((vector @ kmer_beta).cpu()))

    result: dict[str, Any] = {}
    for name, values in predictions.items():
        records = [
            {
                "source_group": row["source_group"],
                "observed": float(row["direction_normalized_effect"]),
                "predicted_mean": value,
                "calibrated_mean": value,
                "predicted_scale": 1.0,
            }
            for row, value in zip(test_rows, values)
        ]
        groups = _group_means(records, "calibrated_mean")
        observed = [item["observed"] for item in groups]
        predicted = [item["predicted"] for item in groups]
        result[name] = {
            "source_group_count": len(groups),
            "spearman": _spearman(observed, predicted),
            "mae": sum(abs(a - b) for a, b in zip(observed, predicted)) / len(groups),
        }
    return result


def _write_terminal_outputs(
    torch,
    model,
    config: Mapping[str, Any],
    output_dir: Path,
    test_predictions: list[dict[str, Any]],
    calibration_quantile: float,
    aggregate: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise ContractError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        outputs = config["output_contract"]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "implementation_id": IMPLEMENTATION_ID,
                "calibration_quantile": calibration_quantile,
            },
            temporary / outputs["private_checkpoint_filename"],
        )
        prediction_path = temporary / outputs["private_predictions_filename"]
        with prediction_path.open("w", encoding="utf-8") as handle:
            for item in test_predictions:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        (temporary / outputs["aggregate_report_filename"]).write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lcb_manifest = {
            "implementation_id": IMPLEMENTATION_ID,
            "run_authority_id": aggregate["run_authority_id"],
            "status": "FROZEN_CALIBRATION_LCB_MANIFEST_CANDIDATE_NOT_ACCEPTED_FOR_A6",
            "mean_calibration": {
                "slope": aggregate["calibration_slope"],
                "intercept": aggregate["calibration_intercept"],
            },
            "interval_quantile": calibration_quantile,
            "lcb_formula": "CALIBRATED_MEAN_MINUS_INTERVAL_QUANTILE_TIMES_CALIBRATED_SCALE",
            "source_role": "CALIBRATION",
            "terminal_role": "TEST",
            "a6_learned_base_value_authorized": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        (temporary / outputs["calibration_lcb_manifest_filename"]).write_text(
            json.dumps(lcb_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_failure(config: Mapping[str, Any], output_dir: Path, error: Exception) -> None:
    if output_dir.exists():
        raise ContractError("failure output directory already exists") from error
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.failure.", dir=output_dir.parent))
    try:
        failure = {
            "implementation_id": IMPLEMENTATION_ID,
            "run_authority_id": config["activation_binding"]["run_authority_id"],
            "status": "TERMINATED_SAFELY_WITH_EVIDENCE_NO_RETRY",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "member_payload_included": False,
            "retry_authorized": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        (temporary / config["output_contract"]["failure_record_filename"]).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_once(config: Mapping[str, Any], repository_root: Path, output_dir: Path) -> dict[str, Any]:
    """Train one fixed critic and publish only its terminal result."""

    require_active_before_operational_io(config)
    activation = config["activation_binding"]
    if output_dir.resolve() != Path(activation["output_directory"]).resolve():
        raise ContractError("output directory differs from active authority")
    _repository_audit(config, repository_root)
    _cuda_binding_audit(config)
    rows, split = _load_rows_and_split(config)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch, model_type = _torch_components(config)
    if torch.__version__ != activation["torch_version"] or torch.version.cuda != activation["torch_cuda_version"]:
        raise ContractError("PyTorch or CUDA runtime version differs from active authority")
    device_name = activation["cuda_device"]
    if not device_name.startswith("cuda:") or not torch.cuda.is_available():
        raise ContractError("bound CUDA device is unavailable")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    seed = int(config["single_fit_contract"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    by_role = {
        role: [row for row in rows if split[row["record_key"]] == role]
        for role in config["input_contract"]["split_roles_exactly"]
    }
    if any(not values for values in by_role.values()):
        raise ContractError("train, calibration and test splits must all be nonempty")

    model = model_type().to(device)
    single = config["single_fit_contract"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(single["learning_rate"]),
        weight_decay=float(single["weight_decay"]),
    )
    batch_size = int(single["batch_size"])
    train_rows = by_role["TRAIN"]
    train_group_counts: dict[str, int] = {}
    for row in train_rows:
        train_group_counts[row["source_group"]] = train_group_counts.get(row["source_group"], 0) + 1
    model.train()
    final_loss = None
    update_count = 0
    for _epoch in range(int(single["epochs"])):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for offset in range(0, len(order), batch_size):
            batch = [train_rows[index] for index in order[offset : offset + batch_size]]
            tensors, effect, standard_error = _batch_tensors(torch, batch, device)
            output = model(**tensors)
            total_variance = output["scale"].square() + standard_error.square()
            losses = 0.5 * (
                torch.log(total_variance)
                + (effect - output["mean"]).square() / total_variance
            )
            weights = _group_weights(torch, batch, train_group_counts, device)
            loss = (losses * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise ContractError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(single["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise ContractError("nonfinite gradient norm")
            optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise ContractError("nonfinite parameter state")
            update_count += 1
            final_loss = float(loss.detach().cpu())

    baseline_metrics = _fit_and_evaluate_baselines(torch, train_rows, by_role["TEST"], config, device)
    calibration = _predict_rows(torch, model, by_role["CALIBRATION"], batch_size, device)
    calibration_slope, calibration_intercept = _calibration_line(calibration)
    _apply_calibration(calibration, calibration_slope, calibration_intercept)
    calibration_quantile = _calibration_quantile(calibration)
    test_predictions = _predict_rows(torch, model, by_role["TEST"], batch_size, device)
    _apply_calibration(test_predictions, calibration_slope, calibration_intercept)
    terminal_metrics = _group_equal_metrics(test_predictions)
    coverage = _group_equal_interval_coverage(test_predictions, calibration_quantile)
    coverage_risk = _coverage_risk(
        calibration,
        test_predictions,
        list(config["evaluator_and_baseline_contract"]["coverage_risk_retained_fractions"]),
    )
    aggregate = {
        "implementation_id": IMPLEMENTATION_ID,
        "run_authority_id": activation["run_authority_id"],
        "status": "DEVELOPMENT_RESULT_NOT_SCIENTIFIC_CLAIM",
        "study_unit": "GSE200304_SUPERSERIES_ONE_STUDY",
        "member_count": len(rows),
        "source_group_count": len({row["source_group"] for row in rows}),
        "train_count": len(by_role["TRAIN"]),
        "calibration_count": len(by_role["CALIBRATION"]),
        "test_count": len(by_role["TEST"]),
        "optimizer_fit_count": 1,
        "baseline_fit_count": 4,
        "parameter_update_count": update_count,
        "checkpoint_count": 1,
        "final_refit_count": 0,
        "seed_count": 1,
        "terminal_training_loss": final_loss,
        "test_source_group_count": terminal_metrics["source_group_count"],
        "test_within_study_source_group_equal_weight_spearman": terminal_metrics["spearman"],
        "test_source_group_equal_weight_mae": terminal_metrics["mae"],
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "calibration_quantile": calibration_quantile,
        "test_source_group_equal_weight_interval_coverage": coverage,
        "coverage_risk_abstention": coverage_risk,
        "baseline_metrics": baseline_metrics,
        "baseline_names": config["evaluator_and_baseline_contract"]["baseline_set"],
        "guide_or_model_selection_feedback_count": 0,
        "test_feedback_count": 0,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "a6_learned_base_value_authorized": False,
        "a7_allowed": False,
    }
    _write_terminal_outputs(
        torch,
        model,
        config,
        output_dir,
        test_predictions,
        calibration_quantile,
        aggregate,
    )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.validate_only:
        print(json.dumps(validate_only(config), sort_keys=True))
        return 0
    if args.output_dir is None:
        raise ContractError("--run requires --output-dir")
    try:
        result = run_once(config, args.repository_root, args.output_dir)
    except InactiveAuthorityError:
        raise
    except Exception as exc:
        _write_failure(config, args.output_dir, exc)
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
