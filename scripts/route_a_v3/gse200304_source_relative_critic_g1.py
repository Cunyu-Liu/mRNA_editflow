#!/usr/bin/env python3
"""Future exactly-one DEC028 source-relative critic implementation.

The checked-in candidate is inactive.  ``--validate-only`` inspects static
configuration and exits before data, PyTorch, CUDA, model, optimizer, checkpoint
or output access.  A later config-only activation must provide all five frozen
activation requirements before ``--run`` can reach the private input contract.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import shutil
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
        "split_evaluator_baseline_id",
        "implementation_gate_review_id",
        "cuda_device",
        "output_directory",
    }
    if set(activation) != expected_activation_keys:
        raise ContractError("activation binding closure differs")
    if config["activation_state"] == INACTIVE:
        if any(value is not None for value in activation.values()):
            raise ContractError("inactive implementation has activation bindings")
    else:
        if any(not isinstance(value, str) or not value for value in activation.values()):
            raise ContractError("active implementation has incomplete activation bindings")

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
        "status": "PASS_STATIC_IMPLEMENTATION_CONTRACT_NOT_ACTIVE_NOT_RUN",
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


def _load_rows_and_split(config: Mapping[str, Any], asset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Private input reader, reachable only after active authority."""

    input_contract = config["input_contract"]
    rows_path = asset_dir / input_contract["materialized_rows_filename"]
    split_path = asset_dir / input_contract["split_assignments_filename"]
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    split = json.loads(split_path.read_text(encoding="utf-8"))
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


def _calibration_quantile(predictions: list[dict[str, Any]], quantile: float = 0.9) -> float:
    scores = sorted(
        abs(item["observed"] - item["predicted_mean"])
        / max(item["predicted_scale"], 1e-6)
        for item in predictions
    )
    if not scores:
        raise ContractError("calibration split is empty")
    index = min(math.ceil((len(scores) + 1) * quantile) - 1, len(scores) - 1)
    return float(scores[max(index, 0)])


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


def run_once(config: Mapping[str, Any], asset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Train one fixed critic and publish only its terminal result."""

    require_active_before_operational_io(config)
    activation = config["activation_binding"]
    if output_dir.resolve() != Path(activation["output_directory"]).resolve():
        raise ContractError("output directory differs from active authority")
    rows, split = _load_rows_and_split(config, asset_dir)
    torch, model_type = _torch_components(config)
    device_name = activation["cuda_device"]
    if not device_name.startswith("cuda:") or not torch.cuda.is_available():
        raise ContractError("bound CUDA device is unavailable")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
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
            loss = (losses * weights).mean()
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
            update_count += 1
            final_loss = float(loss.detach().cpu())

    calibration = _predict_rows(torch, model, by_role["CALIBRATION"], batch_size, device)
    calibration_quantile = _calibration_quantile(calibration)
    test_predictions = _predict_rows(torch, model, by_role["TEST"], batch_size, device)
    observed = [item["observed"] for item in test_predictions]
    predicted = [item["predicted_mean"] for item in test_predictions]
    covered = [
        abs(item["observed"] - item["predicted_mean"])
        <= calibration_quantile * item["predicted_scale"]
        for item in test_predictions
    ]
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
        "parameter_update_count": update_count,
        "checkpoint_count": 1,
        "final_refit_count": 0,
        "seed_count": 1,
        "terminal_training_loss": final_loss,
        "test_within_study_spearman": _spearman(observed, predicted),
        "test_mae": sum(abs(a - b) for a, b in zip(observed, predicted)) / len(observed),
        "calibration_quantile": calibration_quantile,
        "test_interval_coverage": sum(covered) / len(covered),
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
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.validate_only:
        print(json.dumps(validate_only(config), sort_keys=True))
        return 0
    if args.asset_dir is None or args.output_dir is None:
        raise ContractError("--run requires --asset-dir and --output-dir")
    try:
        result = run_once(config, args.asset_dir, args.output_dir)
    except InactiveAuthorityError:
        raise
    except Exception as exc:
        _write_failure(config, args.output_dir, exc)
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
