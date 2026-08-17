#!/usr/bin/env python3
"""Route 2 legal search/generation baseline adapters with matched budgets."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence


ALPHABET = ("A", "C", "G", "U")
METHODS = (
    "random_legal",
    "exhaustive",
    "greedy",
    "beam",
    "genetic",
    "local_search",
    "generate_then_rerank",
)
REGION = {"5UTR": 0, "3UTR": 1}


class SearchBaselineError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SearchBaselineError(message)


def edit_count(source: str, candidate: str) -> int:
    _require(len(source) == len(candidate), "candidate length differs from source")
    _require(set(source) <= set(ALPHABET) and set(candidate) <= set(ALPHABET), "sequence is outside RNA alphabet")
    return sum(left != right for left, right in zip(source, candidate))


def legal_candidate(source: str, candidate: str, edit_budget: int) -> bool:
    try:
        return edit_count(source, candidate) <= edit_budget
    except SearchBaselineError:
        return False


def legal_space_size(source_length: int, edit_budget: int) -> int:
    return sum(math.comb(source_length, edits) * (len(ALPHABET) - 1) ** edits for edits in range(min(source_length, edit_budget) + 1))


def enumerate_legal_candidates(source: str, edit_budget: int) -> Iterable[str]:
    """Enumerate each source-relative candidate once, including immediate STOP."""

    yield source
    for count in range(1, min(len(source), edit_budget) + 1):
        for positions in itertools.combinations(range(len(source)), count):
            choices = [tuple(base for base in ALPHABET if base != source[position]) for position in positions]
            for alts in itertools.product(*choices):
                candidate = list(source)
                for position, alt in zip(positions, alts):
                    candidate[position] = alt
                yield "".join(candidate)


def monotone_neighbors(source: str, candidate: str, edit_budget: int) -> tuple[str, ...]:
    """One legal SUB at an untouched source position; no re-edit or revert."""

    if edit_count(source, candidate) >= edit_budget:
        return ()
    neighbors = []
    for position, (source_base, current_base) in enumerate(zip(source, candidate)):
        if current_base != source_base:
            continue
        for alt in ALPHABET:
            if alt == source_base:
                continue
            child = list(candidate)
            child[position] = alt
            neighbors.append("".join(child))
    return tuple(neighbors)


def random_candidate(source: str, edit_budget: int, rng: random.Random) -> str:
    count = rng.randint(0, min(len(source), edit_budget))
    positions = rng.sample(range(len(source)), count)
    candidate = list(source)
    for position in positions:
        candidate[position] = rng.choice([base for base in ALPHABET if base != source[position]])
    return "".join(candidate)


class BudgetedScorer:
    """Cache exact sequence scores and count only actual critic evaluations."""

    def __init__(self, score_function: Callable[[str], float], max_forwards: int):
        _require(max_forwards > 0, "critic forward budget must be positive")
        self.score_function = score_function
        self.max_forwards = max_forwards
        self.forward_count = 0
        self.cache: dict[str, float] = {}

    @property
    def remaining(self) -> int:
        return self.max_forwards - self.forward_count

    def score(self, sequence: str) -> float:
        if sequence in self.cache:
            return self.cache[sequence]
        if self.remaining <= 0:
            raise SearchBaselineError("critic forward-equivalent budget exhausted")
        value = float(self.score_function(sequence))
        _require(math.isfinite(value), "critic score is not finite")
        self.cache[sequence] = value
        self.forward_count += 1
        return value

    def score_available(self, sequences: Iterable[str]) -> list[tuple[str, float]]:
        ordered = list(dict.fromkeys(sequences))
        missing = [sequence for sequence in ordered if sequence not in self.cache]
        missing = missing[: self.remaining]
        score_many = getattr(self.score_function, "score_many", None)
        if missing and callable(score_many):
            values = list(score_many(missing))
            _require(len(values) == len(missing), "batched critic score count differs")
            for sequence, value in zip(missing, values):
                number = float(value)
                _require(math.isfinite(number), "critic score is not finite")
                self.cache[sequence] = number
            self.forward_count += len(missing)
        elif missing:
            for sequence in missing:
                self.score(sequence)
        result = []
        for sequence in ordered:
            if sequence not in self.cache:
                break
            result.append((sequence, self.cache[sequence]))
        return result


@dataclass(frozen=True)
class SearchResult:
    method_id: str
    candidates: tuple[str, ...]
    scores: tuple[float, ...]
    generator_nfe: int
    proposal_count: int
    critic_forwards: int
    source_score: float


def validate_frozen_checkpoint_provenance(provenance: Mapping[str, object]) -> None:
    physical_index = provenance.get("physical_gpu_index")
    total_memory = provenance.get("cuda_total_memory_mb")
    _require(
        provenance.get("result_stage") == "FROZEN_DEVELOPMENT_VALIDATION"
        and isinstance(provenance.get("optimizer_steps"), int)
        and int(provenance["optimizer_steps"]) > 0
        and provenance.get("parameter_changed") is True
        and provenance.get("cpu_fallback_used") is False
        and provenance.get("cuda_training_tensors_verified") is True
        and isinstance(physical_index, int)
        and not isinstance(physical_index, bool)
        and physical_index >= 0
        and provenance.get("device", provenance.get("torch_device")) == f"cuda:{physical_index}"
        and provenance.get("cuda_device_index") == physical_index
        and isinstance(provenance.get("cuda_device_uuid"), str)
        and bool(provenance.get("cuda_device_uuid"))
        and isinstance(total_memory, (int, float))
        and not isinstance(total_memory, bool)
        and math.isfinite(float(total_memory))
        and float(total_memory) > 0.0,
        "checkpoint does not prove a frozen learned GPU parameter update",
    )


class TorchCheckpointScorer:
    """Frozen Route 2 Delta checkpoint scorer for dynamically proposed candidates."""

    def __init__(self, checkpoint_path: Path, device_text: str):
        import torch
        from core.route2_delta_predictor import (
            ROUTE2_DELTA_MODEL_KIND,
            ROUTE2_EDIT_CENTERED_MODEL_KIND,
            ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND,
            Route2DeltaPredictor,
            Route2EditCenteredDeltaPredictor,
            Route2NeuralBaseline,
        )
        from core.route2_target_scaling import target_scaler_from_checkpoint

        _require(device_text.startswith("cuda"), "checkpoint search scoring requires CUDA")
        _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
        _require(torch.cuda.is_available(), "CUDA is unavailable for checkpoint search scoring")
        self.torch = torch
        self.device = torch.device(device_text)
        torch.cuda.set_device(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        provenance = checkpoint.get("training_provenance", {})
        validate_frozen_checkpoint_provenance(provenance)
        model_kind = checkpoint.get("model_kind")
        if model_kind == ROUTE2_DELTA_MODEL_KIND:
            self.model = Route2DeltaPredictor(**checkpoint["model_config"])
        elif model_kind in {ROUTE2_EDIT_CENTERED_MODEL_KIND, ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND}:
            self.model = Route2EditCenteredDeltaPredictor(**checkpoint["model_config"])
        else:
            _require(model_kind in Route2NeuralBaseline.MODES, f"unsupported checkpoint model kind: {model_kind}")
            self.model = Route2NeuralBaseline(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.vocabs = checkpoint["vocabs"]
        self.target_scaler = target_scaler_from_checkpoint(checkpoint)
        self.source = ""
        self.endpoint = ""
        self.category_ids: dict[str, int] = {}
        self.region_id = 0

    def bind_source(self, source_row: Mapping[str, object]):
        self.source = str(source_row["source_sequence"]).upper().replace("T", "U")
        region_text = str(source_row["region"]).replace("′", "").replace("'", "")
        _require(region_text in REGION, f"unsupported region: {source_row['region']}")
        values = {
            "study": str(source_row["study_unit_id"]),
            "assay": str(source_row["assay_id"]),
            "context": str(source_row["biological_context_id"]),
            "endpoint": str(source_row["endpoint_id"]),
        }
        self.category_ids = {
            field: int(self.vocabs[field].get(value, 0)) for field, value in values.items()
        }
        self.endpoint = values["endpoint"]
        self.region_id = REGION[region_text]
        return self

    @property
    def peak_vram_mb(self) -> float:
        return self.torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def __call__(self, candidate: str) -> float:
        torch = self.torch
        candidate = candidate.upper().replace("T", "U")
        _require(legal_candidate(self.source, candidate, len(self.source)), "checkpoint scorer received an invalid candidate")
        token = {"A": 0, "C": 1, "G": 2, "U": 3}
        source_tokens = torch.tensor([[token[base] for base in self.source]], device=self.device)
        candidate_tokens = torch.tensor([[token[base] for base in candidate]], device=self.device)
        padding = torch.zeros_like(source_tokens, dtype=torch.bool)
        with torch.no_grad():
            output = self.model(
                source_tokens,
                candidate_tokens,
                padding,
                torch.tensor([self.category_ids["study"]], device=self.device),
                torch.tensor([self.category_ids["assay"]], device=self.device),
                torch.tensor([self.category_ids["context"]], device=self.device),
                torch.tensor([self.category_ids["endpoint"]], device=self.device),
                torch.tensor([self.region_id], device=self.device),
            )
        _require(
            output["mean"].is_cuda
            and output["mean"].device == self.device
            and torch.isfinite(output["mean"]).all().item(),
            "checkpoint search prediction left CUDA or became nonfinite",
        )
        scale, _scale_source = self.target_scaler.scale(self.endpoint, self.region_id)
        return float(output["mean"].item()) * scale


class MRNABERTCheckpointScorer:
    """Final-refit mRNABERT scorer with batched online candidate encoding."""

    def __init__(
        self,
        checkpoint_path: Path,
        model_path: Path,
        reward_policy_path: Path,
        device_text: str,
    ):
        from scripts.route_a_v3.route2_mrnabert_guided_critic_v1 import (
            FrozenRoute2MRNABERTCritic,
        )

        _require(device_text.startswith("cuda"), "mRNABERT search scoring requires CUDA")
        _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA remapping is forbidden")
        _require(torch.cuda.is_available(), "CUDA is unavailable for mRNABERT search scoring")
        self.device = torch.device(device_text)
        torch.cuda.set_device(self.device)
        policy = json.loads(reward_policy_path.read_text(encoding="utf-8"))
        _require(
            policy.get("status") == "PROSPECTIVELY_FROZEN_BEFORE_GUIDED_GENERATION"
            and policy.get("uncertainty_in_guidance") == "DISABLED_DIAGNOSTIC_ONLY"
            and policy.get("evaluation_records_used_for_training_hpo_threshold_or_reward") == 0,
            "mRNABERT search reward policy differs",
        )
        transform = policy["potential_transform"]
        self.critic = FrozenRoute2MRNABERTCritic(
            checkpoint_path,
            model_path,
            self.device,
            potential_minimum=float(transform["minimum"]),
            potential_maximum=float(transform["maximum"]),
        )
        self.source_row: Mapping[str, object] | None = None

    def bind_source(self, source_row: Mapping[str, object]):
        self.critic.clear_source_caches()
        self.source_row = source_row
        return self

    @property
    def peak_vram_mb(self) -> float:
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def score_many(self, candidates: Iterable[str]) -> list[float]:
        _require(self.source_row is not None, "mRNABERT scorer is not bound to a source")
        row = self.source_row
        assert row is not None
        return self.critic.score_candidates(
            str(row["source_sequence"]),
            list(candidates),
            assay_id=str(row["assay_id"]),
            context_id=str(row["biological_context_id"]),
            endpoint_id=str(row["endpoint_id"]),
            region=str(row["region"]),
        )

    def __call__(self, candidate: str) -> float:
        return self.score_many([candidate])[0]


def _rank_unique(items: Iterable[tuple[str, float]], candidate_budget: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
    best: dict[str, float] = {}
    for sequence, score in items:
        best[sequence] = max(score, best.get(sequence, -math.inf))
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:candidate_budget]
    return tuple(sequence for sequence, _ in ranked), tuple(score for _, score in ranked)


def _random_legal(source: str, edit_budget: int, candidate_budget: int, scorer: BudgetedScorer, rng: random.Random):
    generated: list[str] = []
    attempts = 0
    max_attempts = max(100, candidate_budget * 30)
    target = min(candidate_budget, legal_space_size(len(source), edit_budget), scorer.max_forwards)
    while len(set(generated)) < target and attempts < max_attempts:
        generated.append(random_candidate(source, edit_budget, rng))
        attempts += 1
    unique = list(dict.fromkeys(generated))
    scored = scorer.score_available(unique)
    return scored[:candidate_budget], attempts


def _exhaustive(source: str, edit_budget: int, candidate_budget: int, scorer: BudgetedScorer, max_space: int):
    size = legal_space_size(len(source), edit_budget)
    _require(size <= max_space, f"legal space {size} exceeds exhaustive limit {max_space}")
    _require(size <= scorer.max_forwards, "exhaustive search requires a forward budget covering the legal space")
    candidates = list(enumerate_legal_candidates(source, edit_budget))
    return scorer.score_available(candidates), len(candidates)


def _beam(source: str, edit_budget: int, scorer: BudgetedScorer, beam_width: int):
    _require(beam_width > 0, "beam width must be positive")
    beam = [source]
    all_scored = scorer.score_available(beam)
    nfe = 1
    for _ in range(edit_budget):
        proposals = sorted({child for parent in beam for child in monotone_neighbors(source, parent, edit_budget)})
        nfe += len(proposals)
        scored = scorer.score_available(proposals)
        if not scored:
            break
        all_scored.extend(scored)
        beam = [sequence for sequence, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:beam_width]]
    return all_scored, nfe


def _mutate(source: str, sequence: str, edit_budget: int, rng: random.Random) -> str:
    candidate = list(sequence)
    edited = [index for index, (left, right) in enumerate(zip(source, sequence)) if left != right]
    unedited = [index for index in range(len(source)) if index not in edited]
    moves = []
    if unedited and len(edited) < edit_budget:
        moves.append("add")
    if edited:
        moves.extend(("change", "remove"))
    if not moves:
        return sequence
    move = rng.choice(moves)
    if move == "add":
        position = rng.choice(unedited)
        candidate[position] = rng.choice([base for base in ALPHABET if base != source[position]])
    elif move == "remove":
        position = rng.choice(edited)
        candidate[position] = source[position]
    else:
        position = rng.choice(edited)
        candidate[position] = rng.choice([base for base in ALPHABET if base not in {source[position], candidate[position]}])
    return "".join(candidate)


def _genetic(source: str, edit_budget: int, scorer: BudgetedScorer, rng: random.Random, population_size: int):
    _require(population_size >= 2, "genetic population must contain at least two candidates")
    population = {random_candidate(source, edit_budget, rng) for _ in range(population_size * 3)}
    population.add(source)
    all_scored: list[tuple[str, float]] = []
    nfe = len(population)
    while scorer.remaining > 0 and population:
        scored = scorer.score_available(sorted(population))
        all_scored.extend(scored)
        parents = [sequence for sequence, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[: max(2, population_size // 2)]]
        if len(parents) < 2:
            break
        children: set[str] = set()
        for _ in range(population_size * 2):
            left, right = rng.sample(parents, 2)
            edits = [(index, base) for index, base in enumerate(left) if base != source[index]]
            edits.extend((index, base) for index, base in enumerate(right) if base != source[index] and (index, base) not in edits)
            rng.shuffle(edits)
            child = list(source)
            used: set[int] = set()
            for index, base in edits:
                if index not in used and len(used) < edit_budget:
                    child[index] = base
                    used.add(index)
            children.add(_mutate(source, "".join(child), edit_budget, rng))
        population = {child for child in children if child not in scorer.cache}
        nfe += len(children)
    return all_scored, nfe


def _local_search(source: str, edit_budget: int, scorer: BudgetedScorer, rng: random.Random):
    current = random_candidate(source, edit_budget, rng)
    current_score = scorer.score(current)
    scored = [(current, current_score)]
    nfe = 1
    step = 0
    while scorer.remaining > 0:
        proposal = _mutate(source, current, edit_budget, rng)
        nfe += 1
        if proposal in scorer.cache:
            step += 1
            if step > scorer.max_forwards * 20:
                break
            continue
        score = scorer.score(proposal)
        scored.append((proposal, score))
        temperature = max(0.01, 1.0 - scorer.forward_count / scorer.max_forwards)
        if score >= current_score or rng.random() < math.exp((score - current_score) / temperature):
            current, current_score = proposal, score
    return scored, nfe


def _generate_then_rerank(
    source: str,
    edit_budget: int,
    candidate_budget: int,
    scorer: BudgetedScorer,
    rng: random.Random,
    oversample_factor: int,
):
    _require(oversample_factor >= 2, "rerank oversample factor must be at least two")
    target = min(candidate_budget * oversample_factor, legal_space_size(len(source), edit_budget), scorer.max_forwards)
    generated: set[str] = set()
    attempts = 0
    while len(generated) < target and attempts < target * 30:
        generated.add(random_candidate(source, edit_budget, rng))
        attempts += 1
    return scorer.score_available(sorted(generated)), attempts


def run_search_method(
    method_id: str,
    source: str,
    *,
    edit_budget: int,
    candidate_budget: int,
    max_critic_forwards: int,
    score_function: Callable[[str], float],
    seed: int,
    beam_width: int = 8,
    population_size: int = 16,
    oversample_factor: int = 8,
    exhaustive_space_limit: int = 100_000,
) -> SearchResult:
    _require(method_id in METHODS, f"unknown method: {method_id}")
    _require(source and set(source) <= set(ALPHABET), "invalid RNA source")
    _require(edit_budget >= 0 and candidate_budget > 0, "budgets must be nonnegative/positive")
    scorer = BudgetedScorer(score_function, max_critic_forwards)
    rng = random.Random(seed)
    source_score = scorer.score(source)
    if method_id == "random_legal":
        scored, nfe = _random_legal(source, edit_budget, candidate_budget, scorer, rng)
    elif method_id == "exhaustive":
        scored, nfe = _exhaustive(source, edit_budget, candidate_budget, scorer, exhaustive_space_limit)
    elif method_id == "greedy":
        scored, nfe = _beam(source, edit_budget, scorer, beam_width=1)
    elif method_id == "beam":
        scored, nfe = _beam(source, edit_budget, scorer, beam_width=beam_width)
    elif method_id == "genetic":
        scored, nfe = _genetic(source, edit_budget, scorer, rng, population_size)
    elif method_id == "local_search":
        scored, nfe = _local_search(source, edit_budget, scorer, rng)
    else:
        scored, nfe = _generate_then_rerank(
            source, edit_budget, candidate_budget, scorer, rng, oversample_factor
        )
    candidates, scores = _rank_unique(scored, candidate_budget)
    _require(candidates, f"method produced no candidates: {method_id}")
    _require(all(legal_candidate(source, sequence, edit_budget) for sequence in candidates), "method produced an illegal candidate")
    return SearchResult(method_id, candidates, scores, 0, nfe, scorer.forward_count, source_score)


def validated_search_hyperparameters(
    *,
    beam_width: int,
    genetic_population_size: int,
    oversample_factor: int,
    exhaustive_space_limit: int,
) -> dict[str, int]:
    values = {
        "beam_width": beam_width,
        "genetic_population_size": genetic_population_size,
        "oversample_factor": oversample_factor,
        "exhaustive_space_limit": exhaustive_space_limit,
    }
    _require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values.values()), "search hyperparameters must be positive integers")
    return values


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_critic_budgets_by_source(path: Path) -> dict[str, int]:
    rows = _read_jsonl(path)
    budgets: dict[str, int] = {}
    for row in rows:
        source_key = str(row["source_key"])
        _require(source_key not in budgets, f"duplicate guided budget source: {source_key}")
        value = row.get("matched_search_critic_forward_budget")
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"guided critic budget is invalid: {source_key}",
        )
        budgets[source_key] = value
    _require(bool(budgets), "guided critic budget table is empty")
    return budgets


def scoring_execution_provenance(
    checkpoint_used: bool,
    device_text: str | None,
    physical_gpu_index: int | None,
) -> dict[str, object]:
    if checkpoint_used:
        _require(device_text is not None and device_text.startswith("cuda"), "checkpoint scoring requires CUDA")
        _require(
            physical_gpu_index is not None and 0 <= physical_gpu_index < torch.cuda.device_count(),
            "checkpoint scoring physical GPU is unavailable",
        )
        _require(device_text == f"cuda:{physical_gpu_index}", "CUDA device index differs from declared physical GPU")
        return {
            "critic_scoring_execution": "CUDA_CHECKPOINT",
            "device": device_text,
            "physical_gpu_index": physical_gpu_index,
            "cpu_fallback_used": False,
            **cuda_device_observation(physical_gpu_index, require_physical_index_match=True),
        }
    return {
        "critic_scoring_execution": "PRECOMPUTED_SCORE_TABLE",
        "device": None,
        "physical_gpu_index": None,
        "cpu_fallback_used": None,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    scorer = parser.add_mutually_exclusive_group(required=True)
    scorer.add_argument("--score-table", type=Path)
    scorer.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mrnabert-model-path", type=Path)
    parser.add_argument("--reward-policy", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--method", choices=METHODS, required=True)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--max-critic-forwards", type=int)
    budget.add_argument("--critic-budget-by-source", type=Path)
    parser.add_argument("--beam-width", type=int, required=True)
    parser.add_argument("--genetic-population-size", type=int, required=True)
    parser.add_argument("--oversample-factor", type=int, required=True)
    parser.add_argument("--exhaustive-space-limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    search_hyperparameters = validated_search_hyperparameters(
        beam_width=args.beam_width,
        genetic_population_size=args.genetic_population_size,
        oversample_factor=args.oversample_factor,
        exhaustive_space_limit=args.exhaustive_space_limit,
    )
    sources = _read_jsonl(args.source_manifest)
    critic_budgets = (
        load_critic_budgets_by_source(args.critic_budget_by_source)
        if args.critic_budget_by_source is not None
        else None
    )
    if critic_budgets is not None:
        source_keys = [str(row["source_key"]) for row in sources]
        _require(
            len(source_keys) == len(set(source_keys)),
            "source manifest contains duplicate source keys",
        )
        _require(
            set(source_keys) == set(critic_budgets),
            "guided critic budgets do not exactly cover the source manifest",
        )
    score_table = None
    if args.score_table:
        score_rows = _read_jsonl(args.score_table)
        score_table = {
            (str(row["source_key"]), str(row["candidate_sequence"]).upper().replace("T", "U")): float(row["critic_score"])
            for row in score_rows
        }
    else:
        _require(
            args.device and args.physical_gpu_index is not None
            and 0 <= args.physical_gpu_index < torch.cuda.device_count(),
            "checkpoint scoring requires an available device and physical GPU",
        )
        _require(str(args.device) == f"cuda:{args.physical_gpu_index}", "CUDA device index differs from declared physical GPU")
    execution_provenance = scoring_execution_provenance(
        args.checkpoint is not None,
        str(args.device) if args.device is not None else None,
        args.physical_gpu_index,
    )
    _require(
        (args.mrnabert_model_path is None) == (args.reward_policy is None),
        "mRNABERT model path and reward policy must be supplied together",
    )
    if args.checkpoint and args.mrnabert_model_path:
        shared_checkpoint_scorer = MRNABERTCheckpointScorer(
            args.checkpoint,
            args.mrnabert_model_path,
            args.reward_policy,
            str(args.device),
        )
    elif args.checkpoint:
        shared_checkpoint_scorer = TorchCheckpointScorer(
            args.checkpoint, str(args.device)
        )
    else:
        shared_checkpoint_scorer = None
    output_rows = []
    for source_index, source_row in enumerate(sources):
        source_key = str(source_row["source_key"])
        source = str(source_row["source_sequence"]).upper().replace("T", "U")
        max_critic_forwards = (
            critic_budgets[source_key]
            if critic_budgets is not None
            else int(args.max_critic_forwards)
        )
        source_seed = int(args.seed) + source_index * 1_000_003

        checkpoint_scorer = (
            shared_checkpoint_scorer.bind_source(source_row) if shared_checkpoint_scorer is not None else None
        )

        def score_function(sequence: str, *, key=source_key, dynamic=checkpoint_scorer) -> float:
            if dynamic is not None:
                return dynamic(sequence)
            try:
                assert score_table is not None
                return score_table[(key, sequence)]
            except KeyError as exc:
                raise SearchBaselineError(f"score table does not cover generated candidate: {key}/{sequence}") from exc

        result = run_search_method(
            args.method,
            source,
            edit_budget=int(source_row["edit_budget"]),
            candidate_budget=int(source_row["candidate_budget"]),
            max_critic_forwards=max_critic_forwards,
            score_function=score_function,
            seed=source_seed,
            beam_width=args.beam_width,
            population_size=args.genetic_population_size,
            oversample_factor=args.oversample_factor,
            exhaustive_space_limit=args.exhaustive_space_limit,
        )
        for index, (candidate, score) in enumerate(zip(result.candidates, result.scores)):
            distance = edit_count(source, candidate)
            output_rows.append({
                "method_id": result.method_id,
                "source_key": source_key,
                "candidate_sequence": candidate,
                "critic_score": score,
                "source_critic_score": result.source_score,
                "critic_forward_budget": max_critic_forwards,
                "critic_forward_budget_rule": (
                    "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE"
                    if critic_budgets is not None
                    else "FIXED_NUMERIC_PER_SOURCE"
                ),
                "terminal_cause": "BUDGET_EXHAUSTED" if distance == int(source_row["edit_budget"]) else "EXPLICIT_STOP",
                "generator_nfe": result.generator_nfe if index == 0 else 0,
                "proposal_count": result.proposal_count if index == 0 else 0,
                "critic_forwards": result.critic_forwards if index == 0 else 0,
                "independent_evaluator_forwards": 0,
                "seed": source_seed,
                "search_hyperparameters": search_hyperparameters,
                "peak_vram_mb": checkpoint_scorer.peak_vram_mb if checkpoint_scorer is not None and index == 0 else 0,
                **execution_provenance,
            })
    _require(not args.output.exists(), f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    return 0


def main() -> int:
    try:
        return _main()
    except Exception as exc:
        arguments = sys.argv[1:]
        if "--checkpoint" in arguments and "--output" in arguments:
            output = Path(arguments[arguments.index("--output") + 1])
            device = arguments[arguments.index("--device") + 1] if "--device" in arguments else None
            physical_index = (
                int(arguments[arguments.index("--physical-gpu-index") + 1])
                if "--physical-gpu-index" in arguments else None
            )
            write_gpu_failure_evidence(
                output.with_suffix(output.suffix + ".failed.json"),
                {"device": device, "physical_gpu_index": physical_index},
                exc,
                entrypoint="run_route2_search_generation_baselines_v1",
                evaluation_outcomes_accessed=False,
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
