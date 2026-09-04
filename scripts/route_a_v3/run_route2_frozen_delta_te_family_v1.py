#!/usr/bin/env python3
"""Task 5.3 (P0): frozen-delta evaluation of RNA-FM / UTR-LM on the three
no-external-target tasks (GSE200304 / GSE149487 / GSE186455), backfilling the
"待 frozen 评测" slots of TASK6_leaderboard_freeze_20260903.md §1.5.

Existing-protocol reuse (no new caliber):
- Encoders: official pretrained weights, zero tuning (frozen). RNA-FM =
  multimolecule conversion (external_model_assets/rnafm); UTR-LM = official
  SISS checkpoint loaded via the official modified ESM source
  (run_route2_utrlm_baseline_v1.load_official_encoder, unchanged).
- Readout: the linear probe of the existing MRL frozen leaderboard rows
  (UTR-LM MRL 0.1107 / RNA-FM MRL 0.134-0.137): features =
  embedding(candidate) - embedding(source) z-scored by fit statistics;
  Linear(embed_dim, 1); AdamW lr 1e-3 wd 1e-4; 100 full-batch epochs;
  seed 20260816; source-group-weighted MSE. The readout scores each sequence
  absolutely (y = w . emb + b); because it is linear,
  delta_hat = y_cand - y_src == linear(emb_cand - emb_src) (bias cancels) -
  the exact functional form of those rows.
- Tasks with an in-study TRAIN split (GSE200304, GSE186455): probe fit on
  task TRAIN, epoch selection by task-VALIDATION source-group-weighted MSE
  (the HPO_VALIDATION_ONLY protocol behind the MRL leaderboard rows),
  evaluated on task VALIDATION.
- GSE149487 has no TRAIN split: LOSO probe per the repo LOSO convention
  (run_route2_classical_prediction_baselines_v1.py): fit on the pooled TRAIN
  split of the other Development studies, bridge-excluded via
  connected_source_component_id, STUDY_THEN_SOURCE_GROUP_EQUAL weighting,
  frozen HPO budget with final-epoch predictions; zero GSE149487 labels used.
- Evaluation: frozen Task-1 evaluator (evaluate_route2_prediction_v1.py),
  VALIDATION only, K=10; per-task Spearman + top-1 + NDCG@10 (null when
  every source group in the stratum is single-candidate: GSE200304 1614
  singletons, GSE149487 48 singletons per endpoint - decision metrics are
  undefined in-stratum, matching leaderboard §1.5 presentation).
- Optional port-validation task gse114002_mrl (explicit --tasks only)
  must reproduce the existing MRL frozen rows end-to-end.

Input adaptation (recorded per model in frozen_delta_results.json):
- RNA-FM: T->U, tokenizer padding, fp32 forward, mean-pool over non-special
  tokens (run_route2_external_prediction_baselines_v1 policy); sequences
  longer than 1000 nt are split into 1000-nt chunks with a length-weighted
  mean of chunk embeddings (build_route2_rnafm_feature_cache_v1 policy);
  length-sorted batches of <=32 sequences / <=8192 tokens.
- UTR-LM: BOS token representation of layer 6 (encode_bos_embeddings
  policy); rotary position embeddings impose no hard length limit;
  length-sorted batches of <=128 sequences / <=16384 tokens (memory
  adaptation only - padding + attention mask keep results identical).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_REPO = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
)
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
RNAFM_MODEL_PATH = MNT / "external_model_assets/rnafm"
UTRLM_ASSET_ROOT = MNT / "external_model_assets/utrlm"
UTRLM_CHECKPOINT = UTRLM_ASSET_ROOT / (
    "Model/Pretrained/ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_"
    "lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl"
)

K = 10
SEED = 20260816
PROBE_EPOCHS = 100
PROBE_LEARNING_RATE = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
RNAFM_CHUNK_NUCLEOTIDES = 1000
RNAFM_MAX_SEQUENCES_PER_BATCH = 32
RNAFM_BATCH_TOKEN_BUDGET = 8192
UTRLM_MAX_SEQUENCES_PER_BATCH = 128
UTRLM_BATCH_TOKEN_BUDGET = 16384
BASES = set("ACGT")

TASKS = {
    "gse200304_te": {
        "study": "GSE200304",
        "region": "3UTR",
        "endpoint": "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY",
        "mode": "IN_STUDY_PROBE",
    },
    "gse149487_te": {
        "study": "GSE149487",
        "region": "5UTR",
        "endpoint": "te_log2_polysome_over_totalrna",
        "mode": "LOSO_PROBE",
    },
    "gse149487_rna": {
        "study": "GSE149487",
        "region": "5UTR",
        "endpoint": "transcript_log2_totalrna_over_dna",
        "mode": "LOSO_PROBE",
    },
    "gse186455": {
        "study": "GSE186455",
        "region": "3UTR",
        "endpoint": "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE",
        "mode": "IN_STUDY_PROBE",
    },
    # Port-validation only (explicit --tasks): must reproduce the MRL rows.
    "gse114002_mrl": {
        "study": "GSE114002",
        "region": "5UTR",
        "endpoint": "MEAN_RIBOSOME_LOAD",
        "mode": "IN_STUDY_PROBE",
    },
}
DEFAULT_TASKS = ("gse200304_te", "gse149487_te", "gse149487_rna", "gse186455")

# References for the aligned interpretation table (leaderboard §1.5 caliber).
REFERENCE = {
    "internal_target_global_scaled_spearman": {
        "gse200304_te": -0.0266,
        "gse149487_te": 0.1747,
        "gse149487_rna": 0.2230,
        "gse186455": -0.0052,
    },
    "critic_v5_spearman": {
        "gse200304_te": 0.0579,
        "gse149487_te": 0.1953,
        "gse149487_rna": 0.0500,
        "gse186455": 0.0639,
    },
    "mrl_frozen_row_spearman": {
        "utrlm": 0.1107267878538859,
        "rnafm": 0.13693329073357266,
    },
    "mrl_frozen_row_top_1": {
        "utrlm": 0.3939393939393939,
        "rnafm": 0.37662337662337664,
    },
    "source": (
        "TASK6_leaderboard_freeze_20260903.md §1.5; "
        "analysis_task1_alignment_20260902/task1_internal_controls_per_task.json; "
        "runs/development_hpo/utrlm_lr1e3_wd1e4_replay_gpu5_v1 and "
        "runs/development_hpo/external_lr1e3_wd1e4_replay_gpu5_v1 (MRL frozen rows)"
    ),
}

CALIBER_DECLARATIONS = [
    "Official pretrained encoder weights are never updated (frozen); the only trained "
    "parameters are the linear readout (RNA-FM 641 / UTR-LM 129 parameters) - the exact "
    "protocol of the existing UTR-LM / RNA-FM MRL frozen leaderboard rows.",
    "The linear readout scores each sequence absolutely (y = w . emb + b); delta_hat = "
    "y_cand - y_src is computed per VALIDATION record and evaluated against "
    "direction_normalized_delta with the frozen Task-1 evaluator (K=10).",
    "VALIDATION split only; Evaluation pool (protected) reads = 0.",
    "GSE149487 has no TRAIN split: LOSO probe fit on the pooled TRAIN split of the other "
    "Development studies (bridge-excluded via connected_source_component_id), "
    "STUDY_THEN_SOURCE_GROUP_EQUAL weighting, final-epoch predictions; zero GSE149487 "
    "labels used - the same information boundary as the internal global controls "
    "(global_scaled / critic V5) whose GSE149487 rows are zero-shot transfers.",
    "top-1 / NDCG@10 are null where every source group in the stratum is single-candidate "
    "(GSE200304: 1614 singleton groups; GSE149487: 48 singleton groups per endpoint).",
]


class FrozenDeltaError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenDeltaError(message)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_module(
    "route2_frozen_eval", EVAL_REPO / "scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
utrlm_lib = _load_module(
    "route2_utrlm_lib", REPO_ROOT / "scripts/route_a_v3/run_route2_utrlm_baseline_v1.py"
)


@dataclass(frozen=True)
class PairRecord:
    record_id: str
    source_id: str
    study: str
    source: str
    candidate: str
    target: float


def canonical_path_for(study: str) -> Path:
    for name in ("canonical_records.private.jsonl", "canonical_records.jsonl"):
        path = MNT / "canonical" / study / "v1" / name
        if path.is_file():
            return path
    raise FrozenDeltaError(f"canonical records absent for {study}")


def load_manifest_rows() -> list[dict]:
    rows = []
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    _require(rows, "development manifest is empty")
    return rows


def task_data(
    task_key: str, manifest_rows: list[dict], smoke_limit: int
) -> tuple[list[PairRecord], list[PairRecord]]:
    """Return (fit_records, eval_records) for one task under the frozen protocol."""
    spec = TASKS[task_key]
    stratum = (spec["study"], spec["region"], spec["endpoint"])
    split_of: dict[str, str] = {}
    study_of: dict[str, str] = {}
    eval_ids: list[str] = []
    fit_ids: list[str] = []
    holdout_components: set[str] = set()
    for row in manifest_rows:
        record_id = str(row["canonical_record_id"])
        split_of[record_id] = str(row["split"])
        study_of[record_id] = str(row["study_unit_id"])
        if spec["mode"] == "LOSO_PROBE" and row["study_unit_id"] == spec["study"]:
            holdout_components.add(str(row["connected_source_component_id"]))
        if tuple(row["stratum"]) == stratum:
            if row["split"] == "VALIDATION":
                eval_ids.append(record_id)
            elif row["split"] == "TRAIN":
                fit_ids.append(record_id)
    if spec["mode"] == "LOSO_PROBE":
        for row in manifest_rows:
            if (
                row["split"] == "TRAIN"
                and row["study_unit_id"] != spec["study"]
                and str(row["connected_source_component_id"]) not in holdout_components
            ):
                fit_ids.append(str(row["canonical_record_id"]))
    _require(eval_ids, f"task {task_key} has no VALIDATION rows")
    _require(fit_ids, f"task {task_key} probe fit pool is empty")
    eval_ids.sort()
    fit_ids.sort()
    if smoke_limit:
        eval_ids = eval_ids[:smoke_limit]
        fit_ids = fit_ids[: max(4 * smoke_limit, 1)]
    needed = set(eval_ids) | set(fit_ids)
    by_study: dict[str, set[str]] = defaultdict(set)
    for record_id in needed:
        by_study[study_of[record_id]].add(record_id)
    records: dict[str, dict] = {}
    for study in sorted(by_study):
        want = by_study[study]
        with canonical_path_for(study).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                record_id = str(row["canonical_record_id"])
                if record_id in want:
                    _require(record_id not in records, f"canonical record duplicated: {record_id}")
                    records[record_id] = row
    _require(set(records) == needed, "canonical coverage mismatch against manifest selection")

    def to_pairs(ids: list[str]) -> list[PairRecord]:
        pairs = []
        for record_id in ids:
            row = records[record_id]
            source = str(row["source_sequence"]).upper()
            candidate = str(row["candidate_sequence"]).upper()
            _require(not ((set(source) | set(candidate)) - BASES), f"non-ACGT sequence in {record_id}")
            target = float(row["direction_normalized_delta"])
            _require(np.isfinite(target), f"non-finite target in {record_id}")
            pairs.append(
                PairRecord(
                    record_id=record_id,
                    source_id=str(row["source_id"]),
                    study=study_of[record_id],
                    source=source,
                    candidate=candidate,
                    target=target,
                )
            )
        return pairs

    return to_pairs(fit_ids), to_pairs(eval_ids)


def _length_budgeted_batches(
    chunks: list[tuple[str, str]], max_sequences: int, token_budget: int
):
    """Length-sorted batching policy of build_route2_rnafm_feature_cache_v1."""
    ordered = sorted(chunks, key=lambda value: (len(value[1]), value[0]))
    current: list[tuple[str, str]] = []
    longest = 0
    for item in ordered:
        proposed = max(longest, len(item[1]))
        if current and (
            len(current) >= max_sequences or proposed * (len(current) + 1) > token_budget
        ):
            yield current
            current = []
            longest = 0
        current.append(item)
        longest = max(longest, len(item[1]))
    if current:
        yield current


def embed_rnafm(sequences: list[str], cache: dict[str, torch.Tensor], device, stats: dict) -> None:
    try:
        from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer
    except ImportError as exc:
        raise FrozenDeltaError("MultiMolecule RNA-FM classes are unavailable") from exc
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    missing = [sequence for sequence in sequences if sequence not in cache]
    if not missing:
        return
    tokenizer = RnaTokenizer.from_pretrained(RNAFM_MODEL_PATH, local_files_only=True)
    model = RnaFmModel.from_pretrained(RNAFM_MODEL_PATH, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(parameter_count > 90_000_000, "RNA-FM pretrained model geometry changed")
    stats["pretrained_parameter_count"] = parameter_count
    chunks: list[tuple[str, str]] = []
    for sequence in sorted(set(missing)):
        for start in range(0, len(sequence), RNAFM_CHUNK_NUCLEOTIDES):
            chunks.append((sequence, sequence[start : start + RNAFM_CHUNK_NUCLEOTIDES]))
    sums: dict[str, torch.Tensor] = {}
    lengths: dict[str, int] = {}
    batch_count = 0
    with torch.no_grad():
        for batch in _length_budgeted_batches(
            chunks, RNAFM_MAX_SEQUENCES_PER_BATCH, RNAFM_BATCH_TOKEN_BUDGET
        ):
            tokens = tokenizer(
                [chunk.replace("T", "U") for _sequence, chunk in batch],
                padding=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens).last_hidden_state
            attention = tokens["attention_mask"].bool()
            special = torch.zeros_like(attention)
            special[:, 0] = True
            token_lengths = attention.sum(dim=1)
            special[torch.arange(len(batch), device=device), token_lengths - 1] = True
            keep = attention & ~special
            pooled = (
                (output * keep.unsqueeze(-1)).sum(dim=1)
                / keep.sum(dim=1, keepdim=True).clamp_min(1)
            )
            _require(
                pooled.is_cuda and torch.isfinite(pooled).all().item(),
                "RNA-FM embedding left CUDA or became nonfinite",
            )
            for (sequence, chunk), embedding in zip(batch, pooled):
                if len(sequence) <= RNAFM_CHUNK_NUCLEOTIDES:
                    cache[sequence] = embedding.detach()
                else:
                    weight = len(chunk)
                    if sequence in sums:
                        sums[sequence] = sums[sequence] + weight * embedding
                    else:
                        sums[sequence] = weight * embedding
                    lengths[sequence] = lengths.get(sequence, 0) + weight
            batch_count += 1
            if batch_count % 200 == 0:
                print(f"[rnafm] embedding batches done: {batch_count}/{len(chunks)} chunks", flush=True)
    for sequence, total in sums.items():
        cache[sequence] = total / lengths[sequence]
    stats["chunked_sequence_count"] = len(sums)
    del model
    torch.cuda.empty_cache()


def embed_utrlm(
    sequences: list[str],
    cache: dict[str, torch.Tensor],
    device,
    model,
    alphabet,
    stats: dict,
) -> None:
    stats["pretrained_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    missing = sorted(set(sequence for sequence in sequences if sequence not in cache))
    if not missing:
        return
    converter = alphabet.get_batch_converter()
    batch_count = 0
    with torch.no_grad():
        for batch in _length_budgeted_batches(
            [(sequence, sequence) for sequence in missing],
            UTRLM_MAX_SEQUENCES_PER_BATCH,
            UTRLM_BATCH_TOKEN_BUDGET,
        ):
            raw = [
                (str(index), sequence, sequence, [])
                for index, (_key, sequence) in enumerate(batch)
            ]
            tokens = converter(raw)[3].to(device)
            output = model(
                tokens,
                repr_layers=[6],
                need_head_weights=False,
                return_contacts=False,
                return_representation=True,
            )
            bos = output["representations"][6][:, 0]
            _require(
                bos.is_cuda and torch.isfinite(bos).all().item(),
                "UTR-LM embedding left CUDA or became nonfinite",
            )
            for (_key, sequence), embedding in zip(batch, bos):
                cache[sequence] = embedding.detach()
            batch_count += 1
            if batch_count % 200 == 0:
                print(f"[utrlm] embedding batches done: {batch_count}", flush=True)


def source_group_weights(records: list[PairRecord], device) -> torch.Tensor:
    """Source-group-equal weights (existing probe convention); mean = 1."""
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.source_id] += 1
    scale = len(records) / len(counts)
    return torch.tensor(
        [scale / counts[record.source_id] for record in records],
        dtype=torch.float32,
        device=device,
    )


def study_then_source_group_weights(records: list[PairRecord], device) -> torch.Tensor:
    """STUDY_THEN_SOURCE_GROUP_EQUAL weighting (classical LOSO convention); mean = 1."""
    by_study: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        by_study[record.study].append(record)
    values: list[float] = []
    for study in sorted(by_study):
        rows = by_study[study]
        values.extend(source_group_weights(rows, device).tolist())
    return torch.tensor(values, dtype=torch.float32, device=device)


def fit_probe(
    fit_records: list[PairRecord],
    selection_records: list[PairRecord] | None,
    embeddings: dict[str, torch.Tensor],
    device,
    weighting: str,
) -> tuple[callable, dict]:
    """Linear probe replica of the MRL frozen-row protocol (see module docstring)."""
    fit_features = torch.stack(
        [embeddings[record.candidate] - embeddings[record.source] for record in fit_records]
    )
    _require(
        fit_features.is_cuda and torch.isfinite(fit_features).all().item(),
        "probe features left CUDA or became nonfinite",
    )
    mean = fit_features.mean(dim=0)
    std = fit_features.std(dim=0).clamp_min(1e-6)
    fit_features = (fit_features - mean) / std
    fit_targets = torch.tensor(
        [record.target for record in fit_records], dtype=torch.float32, device=device
    )
    if weighting == "SOURCE_GROUP_EQUAL":
        fit_weights = source_group_weights(fit_records, device)
    else:
        fit_weights = study_then_source_group_weights(fit_records, device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    head = torch.nn.Linear(fit_features.shape[1], 1).to(device)
    _require(next(head.parameters()).is_cuda, "probe head left CUDA")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=PROBE_LEARNING_RATE, weight_decay=PROBE_WEIGHT_DECAY
    )
    selection_features = None
    selection_targets = None
    selection_weights = None
    if selection_records is not None:
        selection_features = torch.stack(
            [embeddings[r.candidate] - embeddings[r.source] for r in selection_records]
        )
        selection_features = (selection_features - mean) / std
        selection_targets = torch.tensor(
            [r.target for r in selection_records], dtype=torch.float32, device=device
        )
        selection_weights = source_group_weights(selection_records, device)
    best_state = None
    best_validation = float("inf")
    best_epoch = None
    history = []
    for epoch in range(PROBE_EPOCHS):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = head(fit_features).squeeze(1)
        loss = ((prediction - fit_targets) ** 2 * fit_weights).mean()
        _require(
            loss.is_cuda and torch.isfinite(loss).item(),
            "probe loss left CUDA or became nonfinite",
        )
        loss.backward()
        optimizer.step()
        head.eval()
        validation = None
        if selection_features is not None:
            with torch.no_grad():
                error = head(selection_features).squeeze(1) - selection_targets
                validation = float(((error ** 2) * selection_weights).mean())
            if validation < best_validation:
                best_validation = validation
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone() for key, value in head.state_dict().items()
                }
        if epoch == 0 or (epoch + 1) % 10 == 0:
            history.append(
                {
                    "epoch": epoch + 1,
                    "fit_weighted_mse": float(loss.detach()),
                    "selection_weighted_mse": validation,
                }
            )
    if best_state is None:
        best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        best_epoch = PROBE_EPOCHS
        best_validation = None
    head.load_state_dict(best_state)
    head.eval()

    def predict(records: list[PairRecord]) -> dict[str, float]:
        with torch.no_grad():
            features = torch.stack(
                [embeddings[r.candidate] - embeddings[r.source] for r in records]
            )
            features = (features - mean) / std
            values = head(features).squeeze(1)
            _require(
                values.is_cuda and torch.isfinite(values).all().item(),
                "probe prediction left CUDA or became nonfinite",
            )
            return {r.record_id: float(value) for r, value in zip(records, values.cpu())}

    meta = {
        "probe_parameter_count": sum(p.numel() for p in head.parameters()),
        "probe_epochs": PROBE_EPOCHS,
        "probe_learning_rate": PROBE_LEARNING_RATE,
        "probe_weight_decay": PROBE_WEIGHT_DECAY,
        "probe_seed": SEED,
        "probe_weighting": weighting,
        "epoch_selection": (
            "TASK_VALIDATION_SOURCE_GROUP_WEIGHTED_MSE"
            if selection_records is not None
            else "FIXED_FINAL_EPOCH"
        ),
        "selected_epoch": best_epoch,
        "best_selection_weighted_mse": best_validation,
        "history": history,
    }
    return predict, meta


def evaluate_task(
    task_key: str, predictions: dict[str, float], eval_ids: list[str]
) -> dict:
    spec = TASKS[task_key]
    observations = ev.load_observations(
        [canonical_path_for(spec["study"])], set(eval_ids)
    )
    return ev.evaluate(observations, predictions, K)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=sorted(TASKS))
    parser.add_argument("--models", nargs="+", default=["rnafm", "utrlm"], choices=["rnafm", "utrlm"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MNT / "experiments/analysis_frozen_delta_te_family_20260904",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=0,
        help="cap each task to N eval rows / 4N fit rows for fast chain validation",
    )
    args = parser.parse_args()

    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance",
    )
    _require(torch.cuda.is_available(), "CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    _require(0 <= device.index < torch.cuda.device_count(), "physical GPU index is unavailable")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    provenance = {
        "device": str(device),
        "physical_gpu_index": args.physical_gpu_index,
        "cuda_device_name": properties.name,
        "cuda_total_memory_mb": properties.total_memory / (1024 ** 2),
        "cuda_device_uuid": str(properties.uuid),
    }

    manifest_rows = load_manifest_rows()
    data = {task: task_data(task, manifest_rows, args.smoke_limit) for task in args.tasks}
    for task in args.tasks:
        fit_records, eval_records = data[task]
        print(
            f"[data] {task}: fit {len(fit_records)} rows | eval {len(eval_records)} rows | "
            f"mode {TASKS[task]['mode']}",
            flush=True,
        )

    _require(not args.output_dir.exists(), f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    results: dict[str, dict] = {task: {} for task in args.tasks}
    model_stats: dict[str, dict] = {}
    for model_key in args.models:
        all_sequences: set[str] = set()
        for task in args.tasks:
            fit_records, eval_records = data[task]
            for record in fit_records + eval_records:
                all_sequences.add(record.source)
                all_sequences.add(record.candidate)
        ordered_sequences = sorted(all_sequences)
        lengths = [len(sequence) for sequence in ordered_sequences]
        stats: dict = {
            "model_path": str(RNAFM_MODEL_PATH if model_key == "rnafm" else UTRLM_CHECKPOINT),
            "unique_sequence_count": len(ordered_sequences),
            "sequence_length_min_median_max": [
                min(lengths),
                sorted(lengths)[len(lengths) // 2],
                max(lengths),
            ],
        }
        embeddings: dict[str, torch.Tensor] = {}
        print(f"[embed] {model_key}: {len(ordered_sequences)} unique sequences", flush=True)
        if model_key == "rnafm":
            embed_rnafm(ordered_sequences, embeddings, device, stats)
        else:
            model, alphabet = utrlm_lib.load_official_encoder(
                UTRLM_ASSET_ROOT, UTRLM_CHECKPOINT, device
            )
            embed_utrlm(ordered_sequences, embeddings, device, model, alphabet, stats)
            del model
            torch.cuda.empty_cache()
        stats["input_adaptation"] = (
            {
                "tokenizer": "multimolecule RnaTokenizer, T->U, dynamic padding",
                "pooling": "mean over non-special tokens (fp32)",
                "length_limit": "max_position_embeddings 1026 / model_max_length 1024",
                "chunk_policy": (
                    "sequences > 1000 nt split into 1000-nt chunks, "
                    "length-weighted mean of chunk embeddings "
                    "(build_route2_rnafm_feature_cache_v1 policy)"
                ),
                "batching": "length-sorted, <=32 sequences, <=8192 tokens",
            }
            if model_key == "rnafm"
            else {
                "checkpoint": UTRLM_CHECKPOINT.name,
                "official_git_revision": "b77b589bf182eb9de6a1a5024fa09d44294d94fc",
                "embedding": "BOS ([cls]) token representation, layer 6",
                "position_embeddings": "rotary - no hard length limit",
                "batching": (
                    "length-sorted, <=128 sequences, <=16384 tokens "
                    "(memory adaptation; results identical to count batching)"
                ),
            }
        )
        model_stats[model_key] = stats
        for task in args.tasks:
            spec = TASKS[task]
            fit_records, eval_records = data[task]
            if spec["mode"] == "IN_STUDY_PROBE":
                selection_records = eval_records
                weighting = "SOURCE_GROUP_EQUAL"
            else:
                selection_records = None
                weighting = "STUDY_THEN_SOURCE_GROUP_EQUAL"
            predict, probe_meta = fit_probe(
                fit_records, selection_records, embeddings, device, weighting
            )
            predictions = predict(eval_records)
            metrics = evaluate_task(task, predictions, [r.record_id for r in eval_records])
            run_dir = args.output_dir / f"{task}__{model_key}"
            _require(not run_dir.exists(), f"run dir already exists: {run_dir}")
            run_dir.mkdir()
            with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for record_id in sorted(predictions):
                    handle.write(
                        json.dumps(
                            {
                                "canonical_record_id": record_id,
                                "predicted_direction_normalized_delta": predictions[record_id],
                            }
                        )
                        + "\n"
                    )
            prediction_values = np.asarray(list(predictions.values()), dtype=float)
            entry = {
                "mode": "FROZEN_ENCODER_LINEAR_PROBE_DELTA",
                "probe_mode": (
                    "IN_STUDY_PROBE_TRAIN_FIT_VALIDATION_EPOCH_SELECTION"
                    if spec["mode"] == "IN_STUDY_PROBE"
                    else "LOSO_PROBE_POOLED_TRAIN_FINAL_EPOCH"
                ),
                "stratum": f"{spec['study']}|{spec['region']}|{spec['endpoint']}",
                "record_count": len(eval_records),
                "fit_record_count": len(fit_records),
                "task_macro_spearman": metrics.get("task_macro_spearman"),
                "within_source": metrics.get("source_macro_within_source_spearman"),
                "top_1": metrics.get("source_macro_top_1_accuracy"),
                "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
                "source_group_count": metrics.get("source_group_count"),
                "rankable_source_group_count": metrics.get("rankable_source_group_count"),
                "prediction_std": float(prediction_values.std()) if len(prediction_values) else None,
                "metrics_full": metrics,
                "probe": probe_meta,
            }
            if metrics.get("rankable_source_group_count") == 0:
                entry["decision_metric_note"] = (
                    "top-1/NDCG@10 undefined: every source group in this stratum is "
                    "single-candidate"
                )
            with (run_dir / "run_detail.json").open("w", encoding="utf-8") as handle:
                json.dump(entry, handle, indent=1, sort_keys=True)
            results[task][model_key] = {
                key: entry[key]
                for key in (
                    "mode",
                    "probe_mode",
                    "record_count",
                    "fit_record_count",
                    "task_macro_spearman",
                    "within_source",
                    "top_1",
                    "ndcg_at_10",
                    "source_group_count",
                    "rankable_source_group_count",
                    "prediction_std",
                    "decision_metric_note",
                )
                if key in entry
            }
            results[task][model_key]["selected_epoch"] = probe_meta["selected_epoch"]
            print(
                f"[result] {task} x {model_key}: spearman "
                f"{entry['task_macro_spearman'] if entry['task_macro_spearman'] is None else round(entry['task_macro_spearman'], 4)}"
                f" | top-1 {entry['top_1'] if entry['top_1'] is None else round(entry['top_1'], 4)}"
                f" | ndcg@10 {entry['ndcg_at_10'] if entry['ndcg_at_10'] is None else round(entry['ndcg_at_10'], 4)}"
                f" | epoch {probe_meta['selected_epoch']}",
                flush=True,
            )
            if task == "gse114002_mrl":
                expected_spearman = REFERENCE["mrl_frozen_row_spearman"][model_key]
                delta = abs(entry["task_macro_spearman"] - expected_spearman)
                print(
                    f"[port-validation] gse114002_mrl x {model_key}: "
                    f"spearman {entry['task_macro_spearman']:.6f} vs existing row "
                    f"{expected_spearman:.6f} (|delta| {delta:.2e})",
                    flush=True,
                )
        del embeddings
        torch.cuda.empty_cache()

    summary = {
        "schema_version": "route_a_v3_route2_frozen_delta_te_family.v1",
        "mode": "FROZEN_ENCODER_LINEAR_PROBE_DELTA",
        "smoke_limit": args.smoke_limit or None,
        "record_scope": "DEVELOPMENT_VALIDATION_ONLY",
        "protected_reads": 0,
        "k": K,
        "caliber_declarations": CALIBER_DECLARATIONS,
        "probe_protocol": {
            "functional_form": (
                "linear(emb(candidate) - emb(source)); the linear readout scores each "
                "sequence absolutely and delta_hat = y_cand - y_src (bias cancels)"
            ),
            "hyperparameters": {
                "epochs": PROBE_EPOCHS,
                "learning_rate": PROBE_LEARNING_RATE,
                "weight_decay": PROBE_WEIGHT_DECAY,
                "seed": SEED,
            },
            "pretrained_weights_tuned": False,
            "source": (
                "protocol of runs/development_hpo/utrlm_lr1e3_wd1e4_replay_gpu5_v1 and "
                "external_lr1e3_wd1e4_replay_gpu5_v1 (UTR-LM / RNA-FM MRL frozen rows)"
            ),
        },
        "models": model_stats,
        "record_count": {task: results[task][next(iter(results[task]))]["record_count"] for task in args.tasks if results[task]},
        "reference": REFERENCE,
        "results": results,
        "cuda_provenance": provenance,
    }
    output_path = args.output_dir / "frozen_delta_results.json"
    _require(not output_path.exists(), f"output already exists: {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
