"""Frozen XEditCritic V5 guidance critic for SetFlow V5 guided generation (B2).

Pre-authorized substitution: the session task list froze the guided-arm critic
choice to xeditcritic V5 ("SetFlow guided generation (B2/B3, 已授权): 冻结 critic
选 V5") because the previously wired Critic V2 refit checkpoint
(all_development_refit_v1/seed20260823) was never executed and cannot be
produced without an ~80h serial refit.  V5 is the gate-passing critic family
and the spec accepts coarse-grained guidance, so V5 replaces V2 as the frozen
potential with the frozen G0 reward policy otherwise unchanged.

The potential follows the frozen reward policy verbatim
(configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json):
STANDARDIZED_PREDICTED_MEAN_DELTA, i.e. the XEditCriticV4 ``mean`` output
(which predicts the task-robustly scaled source-relative delta), clipped by
the CLIPPED_IDENTITY transform to [-5, 5], with constant beta applied by the
caller exactly as for the V2 critic.

Input assembly reuses the frozen V4 tooling end to end rather than inventing
a parallel path: candidate states are scored through
records_from_projection_rows -> assemble_frozen_bottom_encoder_chunk_cache_v4
(online bottom-six encoding through the frozen
FrozenMRNABERTBottomSixEncoderV4 model/tokenizer and the shared
forward_bottom_six_hidden_v4, the sole bottom-six forward used by both cache
construction and online encoding) -> FrozenBottomEncoderChunkCacheViewV4 ->
XEditCriticDatasetV4 -> XEditCriticCollatorV4 -> the frozen XEditCriticV4
model.  The model itself is built by the reference constructor
train_route2_xeditcritic_v4._build_model (the V5 screen checkpoint keeps the
V4-FULL constructor scope; V5 changed only the training objective), loaded
strict, fully frozen, eval, CUDA, BF16 autocast, and verified against the
checkpoint capacity block.

Throughput contract (the ~14M guided-state cohort of the full 891x32 run):
every unique candidate sequence of a source is bottom-six encoded and scored
exactly once (sequence-hash memoization across trajectories and decision
rounds); candidates are processed in groups of ``round_chunk_size``
(encode -> assemble once per group, model forwards in
``scoring_batch_size`` batches >= 64); the bottom-six host transfer is one
tensor per packed GPU batch instead of per chunk (bit-exact: row slicing
before or after the elementwise ``.float().cpu()`` transfer yields identical
tensors, and the frozen pooling/validation helpers run unchanged on the same
host tensors); intra-op CPU threads are pinned to a fixed count so replays
are deterministic.  The source identity state (candidate == source) receives
the model's exact identity potential of 0.0 without a forward.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from core.route2_bottom_encoder_chunk_cache_v4 import (
    BottomEncodedChunkV4,
    BottomEncodedSequenceV4,
    assemble_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_development_projection_v3 import load_projection_rows
from core.route2_legal_xeditflow import FlowState
from core.route2_mrnabert_edit_site_features_v3 import (
    CHUNK_NUCLEOTIDES,
    CHUNK_OVERLAP,
    format_utr_chunk,
    legacy_global_chunk_spans,
    official_masked_chunk_mean,
    overlapping_chunk_spans,
    validate_token_layout,
)
from core.route2_xeditcritic_batch_v4 import (
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
    XEditCriticDatasetV4,
)
from core.route2_xeditcritic_training_data_v3 import (
    build_vocabs,
    records_from_projection_rows,
)
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    FrozenMRNABERTBottomSixEncoderV4,
    _ChunkRequestV4,
    forward_bottom_six_hidden_v4,
    request_batches_v4,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import (
    TaskRobustScalerV3,
)
from scripts.route_a_v3.train_route2_xeditcritic_v4 import (
    CPU_RAGGED_STRUCTURE_KEYS_V4,
    _build_model,
    screen_run_spec_v4,
)


DEFAULT_CRITIC_V4_SCREEN_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
)
REGION = {"5UTR": 0, "3UTR": 1}
V5_SCREEN_SEED = 20260907
V5_SELECTION_POLICY = "FINAL_PASS_8_FIXED_NO_VALIDATION_PEAK_RESELECTION"
PINNED_CPU_THREADS = 8


class FrozenXEditCriticV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenXEditCriticV5Error(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _scaler(payload: Mapping[str, Any]) -> TaskRobustScalerV3:
    _require(
        payload.get("schema_version")
        == "route_a_v3_route2_xeditcritic_task_robust_scaler.v3",
        "V5 critic checkpoint scaler identity changed",
    )
    return TaskRobustScalerV3(
        scales={
            str(key): float(value) for key, value in payload["task_scales"].items()
        },
        region_scales={
            int(key): float(value) for key, value in payload["region_scales"].items()
        },
        global_scale=float(payload["global_scale"]),
        floor=float(payload["floor"]),
        training_record_count=int(payload["training_record_count"]),
    )


class FrozenXEditCriticV5:
    """Score per-task source-relative deltas for one source's generated states."""

    def __init__(
        self,
        checkpoint_path: Path,
        model_path: Path,
        device: torch.device,
        *,
        critic_config_path: Path = DEFAULT_CRITIC_V4_SCREEN_CONFIG,
        potential_minimum: float = -5.0,
        potential_maximum: float = 5.0,
        encoding_maximum_sequences_per_batch: int = 256,
        encoding_batch_token_budget: int = 65536,
        round_chunk_size: int = 8192,
        scoring_batch_size: int = 512,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        model_path = Path(model_path)
        _require(
            checkpoint_path.is_file(),
            f"V5 critic checkpoint is absent: {checkpoint_path}",
        )
        _require(
            model_path.is_dir(), f"mRNABERT model directory is absent: {model_path}"
        )
        _require(device.type == "cuda", "frozen V5 critic guidance requires CUDA")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        minimum = float(potential_minimum)
        maximum = float(potential_maximum)
        _require(
            math.isfinite(minimum) and math.isfinite(maximum) and minimum < maximum,
            "potential clip is invalid",
        )
        _require(scoring_batch_size >= 64, "V5 critic scoring batch must be at least 64")
        _require(
            round_chunk_size >= scoring_batch_size,
            "V5 critic round chunk must cover at least one scoring batch",
        )
        _require(
            encoding_maximum_sequences_per_batch >= 8
            and encoding_batch_token_budget >= 4096,
            "V5 critic online encoding batch limits are invalid",
        )
        # Pin intra-op CPU threads: the per-record host assembly is dominated
        # by small tensor ops that thrash at the 96-core default, and a fixed
        # count keeps replays deterministic.
        torch.set_num_threads(PINNED_CPU_THREADS)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.minimum = minimum
        self.maximum = maximum
        self.scoring_batch_size = int(scoring_batch_size)
        self.round_chunk_size = int(round_chunk_size)
        config = _read_json(Path(critic_config_path))
        _require(
            config.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_screen_config.v1",
            "unexpected Critic V4 screen config schema",
        )
        _require(
            Path(config["mrnabert_model_path"]) == model_path,
            "runner mRNABERT model directory differs from the frozen critic config",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(
            checkpoint.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_screen_checkpoint.v2",
            "V5 critic checkpoint schema changed",
        )
        _require(
            checkpoint.get("run_stage") == "SCREEN"
            and checkpoint.get("run_id") == "v5_full"
            and checkpoint.get("selected_pass") == 8
            and checkpoint.get("selection_policy") == V5_SELECTION_POLICY
            and int(checkpoint.get("seed", -1)) == V5_SCREEN_SEED
            and int(config["training"]["screen_seed"]) == V5_SCREEN_SEED,
            "checkpoint is not the terminal V5 screen final-pass-8 run",
        )
        _require(
            checkpoint.get("cpu_fallback_used") is False
            and checkpoint.get("a100_device_verified") is True
            and checkpoint.get("bf16_supported") is True,
            "V5 critic checkpoint lacks GPU training provenance",
        )
        spec = screen_run_spec_v4(config, "v4_full")
        _require(
            checkpoint.get("model_kind") == spec.model_kind
            and checkpoint.get("control_mode") == spec.control_mode
            and checkpoint.get("mechanism_mode") == spec.mechanism_mode
            and checkpoint.get("candidate_bundle_permutation")
            == spec.candidate_bundle_permutation,
            "V5 checkpoint constructor scope differs from the frozen V4-FULL spec",
        )
        self.vocabs = checkpoint["vocabs"]
        self.scaler = _scaler(checkpoint["target_scaler"])
        self._build_guidance_registry(config)
        model, capacity = _build_model(config, spec, self.vocabs, device=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval().requires_grad_(False)
        self.model = model
        _require(
            capacity["module_counts"] == checkpoint["capacity"]["module_counts"],
            "built model module counts differ from the V5 checkpoint capacity",
        )
        self.attention_backend = str(config["memory_preflight"]["attention_backend"])
        self._bottom_six = FrozenMRNABERTBottomSixEncoderV4(
            model_path,
            device,
            maximum_sequences_per_batch=int(encoding_maximum_sequences_per_batch),
            batch_token_budget=int(encoding_batch_token_budget),
            attention_backend=self.attention_backend,
        )
        self._encoding_maximum_sequences_per_batch = int(
            encoding_maximum_sequences_per_batch
        )
        self._encoding_batch_token_budget = int(encoding_batch_token_budget)
        self._minimum_physical_batch = int(
            config["memory_preflight"]["minimum_physical_batch"]
        )
        self._checkpoint_identity = {
            key: checkpoint.get(key)
            for key in (
                "schema_version",
                "run_stage",
                "run_id",
                "model_kind",
                "control_mode",
                "mechanism_mode",
                "candidate_bundle_permutation",
                "seed",
                "selected_pass",
                "selection_policy",
                "training_git_head",
                "physical_gpu_index",
                "precision",
            )
        }
        self._validation_metrics = checkpoint.get("validation_metrics")
        self._record_id_counter = itertools.count()
        self._potential_memo: dict[str, float] = {}
        self.model_batch_forward_count = 0
        self.candidate_forward_equivalent_count = 0
        self.encoded_sequence_count = 0
        self.potential_query_count = 0
        self.potential_newly_scored_count = 0
        self.scoring_batch_count = 0

    def _build_guidance_registry(self, config: Mapping[str, Any]) -> None:
        """Rebuild the V5 training vocab/task registry from the frozen projections."""

        rows = load_projection_rows(
            [Path(path) for path in config["projection_paths"]],
            allowed_splits=("TRAIN", "VALIDATION"),
        )
        records = records_from_projection_rows(rows)
        _require(
            len(records) == int(config["data_geometry"]["expected_record_count"]),
            "V5 critic projection record count changed",
        )
        _require(
            build_vocabs(records) == self.vocabs,
            "V5 checkpoint vocabulary differs from the projection rebuild",
        )
        descriptors: dict[str, dict[str, Any]] = {}
        for row in rows:
            endpoint = str(row["endpoint_id"])
            descriptor = row["endpoint_descriptor"]
            if endpoint in descriptors:
                _require(
                    descriptors[endpoint] == descriptor,
                    f"endpoint descriptor is ambiguous: {endpoint}",
                )
            else:
                descriptors[endpoint] = descriptor
        self._endpoint_descriptors = descriptors
        self._task_ids = {str(record.task) for record in records}
        self._registry_record_count = len(records)

    @property
    def cached_potential_count(self) -> int:
        return len(self._potential_memo)

    def clear_source_caches(self) -> None:
        """Start an independently budgeted source cohort without stale scores."""

        self._potential_memo.clear()

    def guidance_provenance(self) -> dict[str, Any]:
        return {
            "critic_kind": "XEDITCRITIC_V5_FROZEN_GUIDANCE",
            "preauthorization": (
                "SESSION_TASK_LIST_FROZEN_CRITIC_CHOICE_V5_CRITIC_V2_REFIT_ABSENT"
            ),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_identity": dict(self._checkpoint_identity),
            "validation_task_macro_spearman": (
                None
                if self._validation_metrics is None
                else self._validation_metrics.get("task_macro_spearman")
            ),
            "reward_signal": "STANDARDIZED_PREDICTED_MEAN_DELTA",
            "potential_transform": {
                "kind": "CLIPPED_IDENTITY",
                "minimum": self.minimum,
                "maximum": self.maximum,
            },
            "attention_backend": self.attention_backend,
            "online_encoding_path": (
                "FrozenMRNABERTBottomSixEncoderV4 model/tokenizer with the shared "
                "forward_bottom_six_hidden_v4 and batched host transfers, "
                "assembled through "
                "assemble_frozen_bottom_encoder_chunk_cache_v4 and the frozen "
                "V4 collator"
            ),
            "registry_record_count": self._registry_record_count,
            "encoding_batching": {
                "maximum_sequences_per_batch": (
                    self._encoding_maximum_sequences_per_batch
                ),
                "batch_token_budget": self._encoding_batch_token_budget,
                "host_transfer": "ONE_TENSOR_PER_PACKED_BATCH",
                "note": (
                    "batch packing affects throughput only; the bottom-six "
                    "forward is the frozen unpadded varlen path shared with "
                    "cache construction and stays deterministic per run"
                ),
            },
            "scoring_batch_size": self.scoring_batch_size,
            "round_chunk_size": self.round_chunk_size,
            "pinned_cpu_threads": PINNED_CPU_THREADS,
        }

    def _move(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            and key not in CPU_RAGGED_STRUCTURE_KEYS_V4
            else value
            for key, value in batch.items()
        }

    def _encode_sequences_fast(
        self, sequences: Mapping[int, str]
    ) -> dict[int, BottomEncodedSequenceV4]:
        """Bit-exact batched-host-transfer variant of encode_sequences.

        Identical tokenizer input, chunk-span policy, request packing, and
        forward_bottom_six_hidden_v4 call as
        FrozenMRNABERTBottomSixEncoderV4.encode_sequences; the only change is
        that each packed GPU batch returns to the host as one
        ``.float().cpu()`` tensor pair instead of per-chunk synchronizations
        (elementwise cast and row slicing commute), after which the frozen
        layout validation and masked-mean pooling run unchanged on the host.
        """

        encoder = self._bottom_six
        _require(bool(sequences), "no sequences were supplied")
        local_spans = {
            index: tuple(
                overlapping_chunk_spans(
                    len(sequence),
                    chunk_nucleotides=CHUNK_NUCLEOTIDES,
                    overlap=CHUNK_OVERLAP,
                )
            )
            for index, sequence in sequences.items()
        }
        global_spans = {
            index: tuple(
                legacy_global_chunk_spans(
                    len(sequence), chunk_nucleotides=CHUNK_NUCLEOTIDES
                )
            )
            for index, sequence in sequences.items()
        }
        requests = {
            _ChunkRequestV4(index, span)
            for index in sequences
            for span in {*local_spans[index], *global_spans[index]}
        }
        batches = request_batches_v4(
            requests,
            maximum_sequences=self._encoding_maximum_sequences_per_batch,
            token_budget=self._encoding_batch_token_budget,
        )
        local_hidden: dict[
            tuple[int, Any], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        global_sums: dict[int, torch.Tensor] = {}
        global_lengths: dict[int, int] = {}
        global_span_sets = {
            index: set(spans) for index, spans in global_spans.items()
        }
        with torch.inference_mode():
            for batch in batches:
                chunks = [
                    sequences[request.sequence_index][
                        request.span.start : request.span.end
                    ]
                    for request in batch
                ]
                tokenized = encoder.tokenizer(
                    [format_utr_chunk(chunk) for chunk in chunks],
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                tokenized = {
                    key: value.to(encoder.device)
                    for key, value in tokenized.items()
                }
                token_type_ids = tokenized.get("token_type_ids")
                if token_type_ids is None:
                    token_type_ids = torch.zeros_like(tokenized["input_ids"])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = forward_bottom_six_hidden_v4(
                        encoder.model,
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                        token_type_ids=token_type_ids,
                    )
                hidden_host = hidden.float().cpu()
                mask_host = tokenized["attention_mask"].cpu()
                for row_index, request in enumerate(batch):
                    mask = mask_host[row_index]
                    validate_token_layout(mask, chunk_length=request.span.length)
                    active_token_count = int(mask.sum().item())
                    active_hidden = hidden_host[row_index, :active_token_count]
                    active_mask = mask[:active_token_count].to(torch.bool)
                    key = request.sequence_index
                    if request.span in set(local_spans[key]):
                        local_hidden[(key, request.span)] = (
                            active_hidden,
                            active_mask,
                        )
                    if request.span in global_span_sets[key]:
                        pooled = official_masked_chunk_mean(
                            active_hidden, active_mask
                        )
                        weight = request.span.length
                        global_sums[key] = (
                            pooled * weight
                            if key not in global_sums
                            else global_sums[key] + pooled * weight
                        )
                        global_lengths[key] = (
                            global_lengths.get(key, 0) + weight
                        )
        result: dict[int, BottomEncodedSequenceV4] = {}
        for index in sorted(sequences):
            _require(
                global_lengths.get(index) == len(sequences[index]),
                "global bottom-six residual coverage changed",
            )
            chunks = tuple(
                BottomEncodedChunkV4(
                    span=span,
                    hidden=local_hidden[(index, span)][0],
                    attention_mask=local_hidden[(index, span)][1],
                )
                for span in local_spans[index]
            )
            result[index] = BottomEncodedSequenceV4(
                chunks=chunks,
                global_residual=global_sums[index] / global_lengths[index],
            )
        _require(
            set(result) == set(sequences),
            "bottom-six sequence encoding is incomplete",
        )
        return result

    def _score_candidate_group(
        self,
        source: str,
        source_row: Mapping[str, Any],
        endpoint_id: str,
        region_id: int,
        task_id: str,
        candidates: Sequence[str],
    ) -> None:
        descriptor = self._endpoint_descriptors[str(endpoint_id)]
        rows: list[dict[str, Any]] = []
        candidate_by_record_id: dict[str, str] = {}
        for candidate in candidates:
            _require(
                len(candidate) == len(source),
                "V5 critic supports source-relative SUB candidates only",
            )
            record_id = f"GUIDED_ONLINE_{next(self._record_id_counter)}"
            rows.append(
                {
                    "canonical_record_id": record_id,
                    "split": "VALIDATION",
                    "source_sequence": source,
                    "candidate_sequence": candidate,
                    "source_relative_edits": [
                        {
                            "position": position,
                            "source_base": source[position],
                            "candidate_base": candidate[position],
                        }
                        for position in range(len(source))
                        if source[position] != candidate[position]
                    ],
                    "direction_normalized_delta": 0.0,
                    "task_id": task_id,
                    "study_unit_id": str(source_row["study_unit_id"]),
                    "source_group_id": str(source_row["source_key"]),
                    "assay_id": str(source_row["assay_id"]),
                    "biological_context_id": str(
                        source_row["biological_context_id"]
                    ),
                    "region_id": int(region_id),
                    "endpoint_descriptor": descriptor,
                }
            )
            candidate_by_record_id[record_id] = candidate
        sequences = sorted({source, *candidates})
        sequence_to_index = {
            sequence: index for index, sequence in enumerate(sequences)
        }
        encoded = self._encode_sequences_fast(
            {index: sequence for sequence, index in sequence_to_index.items()}
        )
        self.encoded_sequence_count += len(sequences)
        cache_payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
            rows,
            sequence_to_index=sequence_to_index,
            encoded=encoded,
            model_id="GUIDED_ONLINE_V5_BOTTOM_SIX",
            pretrained_parameter_count=self._bottom_six.parameter_count,
            attention_backend=self._bottom_six.attention_backend,
        )
        view = FrozenBottomEncoderChunkCacheViewV4(
            cache_payload,
            {str(row["canonical_record_id"]) for row in rows},
            validate_payload=False,
        )
        records = records_from_projection_rows(rows)
        record_by_id = {record.record_id: record for record in records}
        dataset = XEditCriticDatasetV4(
            records,
            all_records=record_by_id,
            vocabs=self.vocabs,
            target_scaler=self.scaler,
            cache=None,
        )
        collator = XEditCriticCollatorV4(
            view,
            minimum_physical_batch=self._minimum_physical_batch,
        )
        record_count = len(records)
        for start in range(0, record_count, self.scoring_batch_size):
            valid_count = min(self.scoring_batch_size, record_count - start)
            indices = list(range(start, start + valid_count))
            while len(indices) < self._minimum_physical_batch:
                indices.append(start)
            batch = self._move(collator([dataset[index] for index in indices]))
            self.model.eval()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = self.model(batch)
            self.model_batch_forward_count += 1
            self.candidate_forward_equivalent_count += valid_count
            self.scoring_batch_count += 1
            values = output["mean"].float()[:valid_count].tolist()
            record_ids = list(batch["record_ids"][:valid_count])
            for record_id, value in zip(record_ids, values, strict=True):
                _require(
                    math.isfinite(float(value)), "V5 critic mean is nonfinite"
                )
                self._potential_memo[candidate_by_record_id[str(record_id)]] = min(
                    self.maximum, max(self.minimum, float(value))
                )

    def potentials(
        self,
        states: Iterable[FlowState],
        *,
        endpoint_id: str,
        region: str,
        source_row: Mapping[str, Any] | None = None,
    ) -> list[float]:
        ordered = list(states)
        _require(bool(ordered), "potential state batch is empty")
        _require(
            source_row is not None,
            "V5 critic potentials requires the runner source row",
        )
        source = ordered[0].source_sequence
        assay = ordered[0].assay_id
        context = ordered[0].context_id
        for state in ordered:
            _require(
                state.source_sequence == source
                and state.assay_id == assay
                and state.context_id == context,
                "batched potentials must share source and biological context",
            )
        _require(
            str(source_row["source_sequence"]).upper().replace("T", "U") == source
            and str(source_row["assay_id"]) == assay
            and str(source_row["biological_context_id"]) == context,
            "source row does not match the queried flow states",
        )
        region_key = str(region).replace("′", "").replace("'", "")
        _require(region_key in REGION, "V5 critic region is unsupported")
        region_id = REGION[region_key]
        endpoint = str(endpoint_id)
        task_id = f"{endpoint}::region={region_id}"
        _require(
            endpoint in self._endpoint_descriptors,
            "V5 critic endpoint is unknown",
        )
        _require(
            task_id in self._task_ids,
            "V5 critic task is outside its training registry",
        )
        sequences = [state.current_sequence for state in ordered]
        self.potential_query_count += len(ordered)
        if source not in self._potential_memo:
            # The frozen model zeroes the identity (candidate == source) row
            # exactly and the study calibration is multiplicative, so the
            # clipped identity potential is exactly zero.
            self._potential_memo[source] = 0.0
        missing = [
            sequence
            for sequence in dict.fromkeys(sequences)
            if sequence not in self._potential_memo
        ]
        self.potential_newly_scored_count += len(missing)
        for start in range(0, len(missing), self.round_chunk_size):
            self._score_candidate_group(
                source,
                source_row,
                endpoint,
                region_id,
                task_id,
                missing[start : start + self.round_chunk_size],
            )
        return [self._potential_memo[sequence] for sequence in sequences]

    def potential(
        self,
        state: FlowState,
        *,
        endpoint_id: str,
        region: str,
        source_row: Mapping[str, Any] | None = None,
    ) -> float:
        return self.potentials(
            [state], endpoint_id=endpoint_id, region=region, source_row=source_row
        )[0]
