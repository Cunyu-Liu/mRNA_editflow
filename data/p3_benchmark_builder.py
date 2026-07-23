"""P3-01: local-edit benchmark construction (measured / proxy / unlabeled tiers).

Tier semantics (anti-fabrication contract):
    measured  : Sample2019 MPRA ``snv`` library — source-matched 5'UTR variants
                with wet-lab ribosome-load values. delta = variant - wild-type
                of the SAME mother sequence (unanchored variants excluded).
    proxy     : reconstructed human mRNA sources + controlled 5'UTR
                neighborhoods scored by the frozen P1-04 CNN-50mer cross-fitted
                ensemble (15 checkpoints). Proxy coverage is limited to the
                first 50 nt of the 5'UTR (the model's input window); edits
                outside that window are NEVER given proxy values — they fall
                into the unlabeled tier instead.
    unlabeled : legal edits (5'UTR beyond the proxy window; synonymous CDS
                scopes; joint 5'UTR+CDS) with ALL value fields null. These are
                data assets for gated tasks (task_b/c), not local-delta ground
                truth.

No GRPO training happens here. This module only builds/audits/freezes data.
"""
from __future__ import annotations

import gzip
import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mrna_editflow.data.p3_legality import (
    has_cryptic_splice,
    has_homopolymer_ge6,
    has_upstream_in_frame_start_codon,
    is_valid_cds,
    is_valid_rna,
    motif_policy_v1_guarded_risk_flags,
    motif_policy_v1_hard_forbidden_triggered,
    normalize_rna,
    protein_identical,
    translate,
)
from mrna_editflow.data.p3_local_edit_schema import (
    BenchmarkRecord,
    SCHEMA_VERSION,
    edits_from_alignment,
    hamming_distance,
    sha256_file,
    write_benchmark_jsonl,
)
from mrna_editflow.data.p3_neighborhood import (
    NeighborhoodConfig,
    apply_edits,
    assemble_doubles,
    cds_singles,
    five_utr_singles,
    joint_doubles,
    structure_disruption,
    REGION_FIVE_UTR,
    REGION_CDS_FIRST30,
    REGION_CDS_FIRST50,
    REGION_CDS_REMAINING,
    REGION_JOINT,
)
from mrna_editflow.data.p3_split import (
    SourceGroup,
    SplitConfig,
    assign_split_roles,
    audit_split,
    split_assignment_sha256,
)

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

SNV_CSV = "data/raw/sample2019_mpra/GSM3130443_designed_library.csv.gz"
RECORDS_JSONL = "data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl"
FAMILY_JSON = "data/reconstructed/p0_data_reconstruction_v1/combined/family_assignments.json"
RECON_MANIFEST = "data/reconstructed/p0_data_reconstruction_v1/combined/combined_reconstruction_manifest.json"
CKPT_DIR = "ckpts/p1_04_predictors"

MEASURED_DATA_SOURCE = "sample2019_mpra:GSM3130443:designed_library:snv"
MEASURED_ASSAY = "mpra_polysome_ribosome_load"
MEASURED_QUALIFIER = "wet-lab measured (MPRA ribosome load, Sample 2019)"
PROXY_DATA_SOURCE = "p0_data_reconstruction_v1+cnn50mer_p1_04"
PROXY_ASSAY = "internal_proxy_cnn50mer_mrl"
PROXY_QUALIFIER = "predicted/internal proxy (CNN-50mer cross-fitted ensemble, P1-04)"
UNLABELED_ASSAY = "none_unlabeled"
UNLABELED_QUALIFIER = "none (unlabeled data asset; not local-delta ground truth)"

PROXY_WINDOW = 50  # CNN-50mer input window (first 50 nt of 5'UTR)


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 20260723
    proxy_sources: int = 2000
    cds_sources: int = 300
    short_utr_fraction: float = 0.15  # fraction of proxy sources with 50<=len<128
    device: str = "cpu"
    torch_threads: int = 16
    smoke: bool = False
    neighborhood: NeighborhoodConfig = NeighborhoodConfig()

    def effective(self) -> "BuildConfig":
        if not self.smoke:
            return self
        return BuildConfig(
            seed=self.seed, proxy_sources=16, cds_sources=8,
            short_utr_fraction=self.short_utr_fraction,
            device=self.device, torch_threads=self.torch_threads,
            smoke=True, neighborhood=self.neighborhood,
        )


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------

def load_snv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh):
            if r.get("library") == "snv":
                rows.append(r)
    return rows


def load_reconstructed_sources(
    records_path: str, family_path: str
) -> List[Dict[str, Any]]:
    with open(family_path, "r") as fh:
        family = json.load(fh)
    sources: List[Dict[str, Any]] = []
    with open(records_path, "r") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            sources.append({
                "transcript_id": r["transcript_id"],
                "five_utr": normalize_rna(r.get("five_utr", "")),
                "cds": normalize_rna(r.get("cds", "")),
                "three_utr": normalize_rna(r.get("three_utr", "")),
                "family_cluster_id": str(family[i]),
            })
    return sources


# ---------------------------------------------------------------------------
# Measured tier (Sample2019 snv)
# ---------------------------------------------------------------------------

def _mother_family_clusters(mothers: Sequence[str], max_hamming: int = 5) -> Dict[str, str]:
    """Greedy near-duplicate clustering of 50nt mothers (hamming <= 5).

    Uses an 8-mer inverted index (hamming<=5 over 50nt guarantees a shared
    identical 8-mer block by the pigeonhole principle).
    """
    index: Dict[str, List[int]] = {}
    for i, m in enumerate(mothers):
        for k in range(0, len(m) - 7):
            index.setdefault(m[k:k + 8], []).append(i)
    parent = list(range(len(mothers)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_pairs = set()
    for ids in index.values():
        if len(ids) < 2:
            continue
        for a_pos in range(len(ids)):
            for b_pos in range(a_pos + 1, len(ids)):
                a, b = ids[a_pos], ids[b_pos]
                key = (a, b)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if hamming_distance(mothers[a], mothers[b]) <= max_hamming:
                    union(a, b)
    fam_of: Dict[str, str] = {}
    root_name: Dict[int, str] = {}
    for i, m in enumerate(mothers):
        root = find(i)
        if root not in root_name:
            root_name[root] = f"snvfam:{len(root_name):05d}"
        fam_of[m] = root_name[root]
    return fam_of


def build_measured_tier(
    rows: Sequence[Dict[str, str]],
) -> Tuple[List[BenchmarkRecord], Dict[str, Any]]:
    """Source-anchored measured records from the snv library."""
    by_mother: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        mother = normalize_rna(r["mother"])
        utr = normalize_rna(r["utr"])
        by_mother.setdefault(mother, []).append({**r, "mother_rna": mother, "utr_rna": utr})

    mothers = sorted(by_mother.keys())
    fam_of = _mother_family_clusters(mothers)

    stats: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_mothers": len(mothers),
        "n_anchored_mothers": 0,
        "n_wt_anchor_records": 0,
        "n_excluded_unanchored_variants": 0,
        "n_excluded_over_budget_variants": 0,
        "n_hard_forbidden_flagged_variants": 0,
    }
    records: List[BenchmarkRecord] = []

    for mother in mothers:
        entries = by_mother[mother]
        wt_rls = sorted(float(e["rl"]) for e in entries if e["utr_rna"] == mother)
        if not wt_rls:
            stats["n_excluded_unanchored_variants"] += len(entries)
            continue
        stats["n_anchored_mothers"] += 1
        anchor_value = wt_rls[len(wt_rls) // 2]  # median of wild-type replicates
        source_id = "snv:" + hashlib.sha1(mother.encode()).hexdigest()[:12]
        fam = fam_of[mother]

        anchor = BenchmarkRecord(
            source_id=source_id, cargo_id="EGFP", cell_context="HEK293T",
            source_sequence=mother, candidate_sequence=mother,
            edit_list=[], edit_count=0, edited_region=REGION_FIVE_UTR,
            protein_identity=True,
            measured_or_proxy_source_value=anchor_value,
            measured_or_proxy_candidate_value=anchor_value,
            delta=0.0,
            data_source=MEASURED_DATA_SOURCE, assay_type=MEASURED_ASSAY,
            confidence="measured", edit_type="wild_type_anchor",
            task_eligibility="task_a_active", value_qualifier=MEASURED_QUALIFIER,
            family_cluster_id=fam,
            internal_features={"n_wt_replicates": len(wt_rls), "wt_rl_values": wt_rls},
        ).finalize()
        records.append(anchor)
        stats["n_wt_anchor_records"] += 1

        for e in entries:
            if e["utr_rna"] == mother:
                continue
            dist = hamming_distance(mother, e["utr_rna"])
            if dist < 0 or dist > 10:
                stats["n_excluded_over_budget_variants"] += 1
                continue
            edit_list = edits_from_alignment(mother, e["utr_rna"], REGION_FIVE_UTR)
            edit_type = {1: "measured_single", 2: "measured_double"}.get(dist, "measured_multi")
            hard = motif_policy_v1_hard_forbidden_triggered(REGION_FIVE_UTR, mother, e["utr_rna"])
            flags = motif_policy_v1_guarded_risk_flags(mother, e["utr_rna"])
            if hard:
                stats["n_hard_forbidden_flagged_variants"] += 1
                flags = list(flags) + [f"hard_forbidden_in_action_space:{h}" for h in hard]
            cand_val = float(e["rl"])
            rec = BenchmarkRecord(
                source_id=source_id, cargo_id="EGFP", cell_context="HEK293T",
                source_sequence=mother, candidate_sequence=e["utr_rna"],
                edit_list=edit_list, edit_count=dist, edited_region=REGION_FIVE_UTR,
                protein_identity=True,
                measured_or_proxy_source_value=anchor_value,
                measured_or_proxy_candidate_value=cand_val,
                delta=cand_val - anchor_value,
                data_source=MEASURED_DATA_SOURCE, assay_type=MEASURED_ASSAY,
                confidence="measured", edit_type=edit_type,
                task_eligibility="task_a_active", value_qualifier=MEASURED_QUALIFIER,
                family_cluster_id=fam, motif_flags=flags,
                internal_features={
                    "variant_id": e.get("id", ""),
                    "rsid": e.get("info1", ""),
                    "locus": e.get("info2", ""),
                    "variant_class": e.get("info4", ""),
                },
            ).finalize()
            records.append(rec)
    return records, stats


# ---------------------------------------------------------------------------
# Source eligibility + stratified selection
# ---------------------------------------------------------------------------

def filter_eligible_sources(
    sources: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Eligibility for benchmark source sequences (fail-closed, documented)."""
    reasons: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []

    def reject(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for s in sources:
        utr, cds = s["five_utr"], s["cds"]
        if len(utr) < PROXY_WINDOW:
            reject("five_utr_shorter_than_50nt")
            continue
        if not is_valid_rna(utr) or not is_valid_rna(cds):
            reject("non_acgu_alphabet")
            continue
        if not is_valid_cds(cds):
            reject("invalid_cds")
            continue
        if has_upstream_in_frame_start_codon(utr):
            reject("source_has_upstream_in_frame_start_codon")
            continue
        if has_homopolymer_ge6(utr) or has_homopolymer_ge6(cds[:150]):
            reject("source_has_homopolymer_ge6")
            continue
        if has_cryptic_splice(utr):
            reject("source_has_cryptic_splice_consensus")
            continue
        out.append(s)
    return out, reasons


def select_proxy_sources(
    pool: Sequence[Dict[str, Any]], cfg: BuildConfig
) -> List[Dict[str, Any]]:
    """Deterministic stratified selection: short-5'UTR (50..127) fraction fixed
    at ``cfg.short_utr_fraction`` so the OOD length-shift axis is populated."""
    rng = random.Random(cfg.seed)
    long_pool = [s for s in pool if len(s["five_utr"]) >= 128]
    short_pool = [s for s in pool if len(s["five_utr"]) < 128]
    rng.shuffle(long_pool)
    rng.shuffle(short_pool)
    n = min(cfg.proxy_sources, len(pool))
    n_short = min(int(round(n * cfg.short_utr_fraction)), len(short_pool))
    n_long = min(n - n_short, len(long_pool))
    selected = long_pool[:n_long] + short_pool[:n_short]
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# CNN-50mer ensemble (batched)
# ---------------------------------------------------------------------------

def load_ensemble(repo_root: str, device: str, torch_threads: int) -> List[Any]:
    """Load the frozen P1-04 cross-fitted CNN-50mer ensemble (15 checkpoints)."""
    import torch  # lazy: data modules must stay torch-free for tests
    torch.set_num_threads(max(1, torch_threads))
    from mrna_editflow.models.predictors.cnn_50mer import CNN50merPredictor

    ckpt_dir = Path(repo_root) / CKPT_DIR
    paths = sorted(ckpt_dir.glob("cnn_50mer__sample2019_mpra__fold*_seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"no CNN-50mer checkpoints under {ckpt_dir}")
    models = []
    for p in paths:
        m = CNN50merPredictor.load(Path(str(p)[:-3]), device=device)
        # Checkpoints were trained on CUDA: the restored hyperparams carry
        # device="cuda", which ``load`` does not override. Force the runtime
        # device so predict() places inputs where the weights live.
        m.device = device
        m.hp.device = device
        m._model.to(device)
        models.append(m)
    return models


def ensemble_predict(models: Sequence[Any], seqs: Sequence[str]) -> Tuple[List[float], List[float]]:
    """(mean, std) across the ensemble for each sequence (first 50 nt used)."""
    import numpy as np  # lazy
    trimmed = [s[:PROXY_WINDOW] for s in seqs]
    per_model = [m.predict(trimmed) for m in models]
    stack = np.stack(per_model, axis=0)  # (n_models, n_seqs)
    means = stack.mean(axis=0)
    stds = stack.std(axis=0)
    return means.tolist(), stds.tolist()


def checkpoint_hashes(repo_root: str) -> Dict[str, str]:
    ckpt_dir = Path(repo_root) / CKPT_DIR
    return {
        p.name: sha256_file(str(p))
        for p in sorted(ckpt_dir.glob("cnn_50mer__sample2019_mpra__fold*_seed*.pt"))
    }


# ---------------------------------------------------------------------------
# Proxy tier (5'UTR neighborhoods + CNN-50mer ensemble)
# ---------------------------------------------------------------------------

def build_proxy_tier(
    sources: Sequence[Dict[str, Any]],
    models: Sequence[Any],
    cfg: BuildConfig,
) -> Tuple[List[BenchmarkRecord], List[BenchmarkRecord], Dict[str, Any]]:
    """Returns (proxy_records, unlabeled_five_utr_records, stats).

    The unlabeled byproduct covers 5'UTR edits OUTSIDE the first-50nt proxy
    window (legal task_a edits the proxy cannot score — never given values).
    """
    nbc = cfg.neighborhood
    stats: Dict[str, Any] = {"n_sources": len(sources)}

    # Phase A: enumerate legal singles inside the proxy window; collect seqs.
    per_source: List[Dict[str, Any]] = []
    seq_registry: Dict[str, int] = {}
    seq_list: List[str] = []

    def register(seq: str) -> int:
        key = seq[:PROXY_WINDOW]
        if key not in seq_registry:
            seq_registry[key] = len(seq_list)
            seq_list.append(key)
        return seq_registry[key]

    for s in sources:
        utr = s["five_utr"]
        singles = five_utr_singles(utr, (0, min(PROXY_WINDOW, len(utr))))
        entry = {"source": s, "singles": singles}
        register(utr)
        for e in singles:
            register(apply_edits(utr, [e]))
        per_source.append(entry)

    # Phase B: score sources + singles.
    means, stds = ensemble_predict(models, seq_list)
    value_of = {seq_list[i]: means[i] for i in range(len(seq_list))}
    std_of = {seq_list[i]: stds[i] for i in range(len(seq_list))}

    # Phase C: assemble doubles/negatives; register their sequences.
    for entry in per_source:
        s, utr, singles = entry["source"], entry["source"]["five_utr"], entry["singles"]
        src_val = value_of[utr[:PROXY_WINDOW]]
        scores = {}
        for e in singles:
            cand = apply_edits(utr, [e])
            scores[e] = abs(value_of[cand[:PROXY_WINDOW]] - src_val)
        rng = random.Random(f"{cfg.seed}:{s['transcript_id']}")
        entry["gen"] = assemble_doubles(singles, scores, utr, nbc, rng)
        entry["single_scores"] = scores
        for etype, groups in entry["gen"].items():
            for group in groups:
                register(apply_edits(utr, list(group)))

    # Phase D: score everything (cached registry; new seqs only appended).
    if len(seq_list) != len(means):
        means, stds = ensemble_predict(models, seq_list)
        value_of = {seq_list[i]: means[i] for i in range(len(seq_list))}
        std_of = {seq_list[i]: stds[i] for i in range(len(seq_list))}

    stats["n_scored_sequences"] = len(seq_list)

    # Phase E: emit proxy records.
    proxy_records: List[BenchmarkRecord] = []
    for entry in per_source:
        s, utr, singles = entry["source"], entry["source"]["five_utr"], entry["singles"]
        src_val = value_of[utr[:PROXY_WINDOW]]
        protein = translate(s["cds"])
        cargo_id = "prot:" + hashlib.sha1(protein.encode()).hexdigest()[:12]

        def make_rec(edit_group: Sequence[Any], edit_type: str) -> BenchmarkRecord:
            edits = list(edit_group)
            cand = apply_edits(utr, edits)
            cand_val = value_of[cand[:PROXY_WINDOW]]
            return BenchmarkRecord(
                source_id=s["transcript_id"], cargo_id=cargo_id,
                cell_context="HEK293T_proxy",
                source_sequence=utr, candidate_sequence=cand,
                edit_list=[e.to_dict() for e in edits], edit_count=len(edits),
                edited_region=REGION_FIVE_UTR, protein_identity=True,
                measured_or_proxy_source_value=src_val,
                measured_or_proxy_candidate_value=cand_val,
                delta=cand_val - src_val,
                data_source=PROXY_DATA_SOURCE, assay_type=PROXY_ASSAY,
                confidence="proxy", edit_type=edit_type,
                task_eligibility="task_a_active", value_qualifier=PROXY_QUALIFIER,
                value_std=std_of[cand[:PROXY_WINDOW]],
                family_cluster_id=s["family_cluster_id"],
                motif_flags=motif_policy_v1_guarded_risk_flags(utr, cand),
            ).finalize()

        for e in singles:
            proxy_records.append(make_rec([e], "all_legal_single"))
        for etype in ("random_double", "structure_guided_double",
                      "topranked_double", "matched_negative_single"):
            for group in entry["gen"][etype]:
                proxy_records.append(make_rec(group, etype))

    # Phase F: unlabeled 5'UTR edits OUTSIDE the proxy window.
    unlabeled_records: List[BenchmarkRecord] = []
    for entry in per_source:
        s, utr = entry["source"], entry["source"]["five_utr"]
        if len(utr) <= PROXY_WINDOW:
            continue
        singles = five_utr_singles(utr, (PROXY_WINDOW, len(utr)))
        if not singles:
            continue
        protein = translate(s["cds"])
        cargo_id = "prot:" + hashlib.sha1(protein.encode()).hexdigest()[:12]
        scores = {e: structure_disruption(utr, e, nbc.structure_window) for e in singles}
        rng = random.Random(f"{cfg.seed}:unlab:{s['transcript_id']}")
        gen = assemble_doubles(singles, scores, utr, nbc, rng)

        def make_unlab(edit_group: Sequence[Any], edit_type: str) -> BenchmarkRecord:
            edits = list(edit_group)
            cand = apply_edits(utr, edits)
            return BenchmarkRecord(
                source_id=s["transcript_id"], cargo_id=cargo_id,
                cell_context="none_unlabeled",
                source_sequence=utr, candidate_sequence=cand,
                edit_list=[e.to_dict() for e in edits], edit_count=len(edits),
                edited_region=REGION_FIVE_UTR, protein_identity=True,
                measured_or_proxy_source_value=None,
                measured_or_proxy_candidate_value=None, delta=None,
                data_source="p0_data_reconstruction_v1", assay_type=UNLABELED_ASSAY,
                confidence="unlabeled", edit_type=edit_type,
                task_eligibility="task_a_active", value_qualifier=UNLABELED_QUALIFIER,
                family_cluster_id=s["family_cluster_id"],
                motif_flags=motif_policy_v1_guarded_risk_flags(utr, cand),
                internal_features={"rank_feature": "structure_disruption"},
            ).finalize()

        for e in singles:
            unlabeled_records.append(make_unlab([e], "all_legal_single"))
        for etype in ("random_double", "structure_guided_double",
                      "topranked_double", "matched_negative_single"):
            for group in gen[etype]:
                unlabeled_records.append(make_unlab(group, etype))

    return proxy_records, unlabeled_records, stats


# ---------------------------------------------------------------------------
# Unlabeled CDS + joint tier (gated tasks; null values by contract)
# ---------------------------------------------------------------------------

def build_unlabeled_cds_tier(
    sources: Sequence[Dict[str, Any]],
    cfg: BuildConfig,
) -> Tuple[List[BenchmarkRecord], Dict[str, Any]]:
    nbc = cfg.neighborhood
    records: List[BenchmarkRecord] = []
    stats: Dict[str, Any] = {"n_sources": len(sources), "protein_identity_failures": 0}

    for s in sources:
        utr, cds = s["five_utr"], s["cds"]
        protein = translate(cds)
        cargo_id = "prot:" + hashlib.sha1(protein.encode()).hexdigest()[:12]
        rng = random.Random(f"{cfg.seed}:cds:{s['transcript_id']}")

        cds_scopes: List[Tuple[str, Tuple[int, int], Optional[int]]] = [
            (REGION_CDS_FIRST30, (0, 30), None),
            (REGION_CDS_FIRST50, (0, 50), None),
            (REGION_CDS_REMAINING, (50, len(cds) // 3), nbc.cds_remaining_max_singles),
        ]
        scope_singles: Dict[str, Tuple[List[Any], str]] = {}
        for region, window, cap in cds_scopes:
            if window[0] >= len(cds) // 3 - 1:
                continue  # CDS too short for this scope
            edits, scope_seq = cds_singles(cds, region, window, max_singles=cap)
            if not edits:
                continue
            scope_singles[region] = (edits, scope_seq)

        for region, (edits, scope_seq) in scope_singles.items():
            scores = {e: structure_disruption(scope_seq, e, nbc.structure_window) for e in edits}
            gen = assemble_doubles(edits, scores, scope_seq, nbc, rng)

            def make_cds_rec(edit_group: Sequence[Any], edit_type: str,
                             region: str = region, scope_seq: str = scope_seq
                             ) -> Optional[BenchmarkRecord]:
                group = list(edit_group)
                cand_scope = apply_edits(scope_seq, group)
                # Verify protein identity against the FULL CDS (fail-closed:
                # records that would alter the protein are never emitted).
                full_new = _apply_scope_edits_to_full_cds(cds, scope_seq, region, group)
                if not protein_identical(cds, full_new):
                    stats["protein_identity_failures"] += 1
                    return None
                return BenchmarkRecord(
                    source_id=s["transcript_id"], cargo_id=cargo_id,
                    cell_context="none_unlabeled",
                    source_sequence=scope_seq, candidate_sequence=cand_scope,
                    edit_list=[e.to_dict() for e in group], edit_count=len(group),
                    edited_region=region, protein_identity=True,
                    measured_or_proxy_source_value=None,
                    measured_or_proxy_candidate_value=None, delta=None,
                    data_source="p0_data_reconstruction_v1", assay_type=UNLABELED_ASSAY,
                    confidence="unlabeled", edit_type=edit_type,
                    task_eligibility="task_b_frozen_fallback",
                    value_qualifier=UNLABELED_QUALIFIER,
                    family_cluster_id=s["family_cluster_id"],
                    motif_flags=motif_policy_v1_guarded_risk_flags(scope_seq, cand_scope),
                    internal_features={"rank_feature": "structure_disruption"},
                ).finalize()

            for e in edits:
                rec = make_cds_rec([e], "all_legal_single")
                if rec is not None:
                    records.append(rec)
            for etype in ("random_double", "structure_guided_double",
                          "topranked_double", "matched_negative_single"):
                for group in gen[etype]:
                    rec = make_cds_rec(group, etype)
                    if rec is not None:
                        records.append(rec)

        # ---- joint 5'UTR + CDS (task_c_locked_extension) ----
        if (REGION_CDS_FIRST50 in scope_singles):
            utr_singles = five_utr_singles(utr, (0, len(utr)))
            cds50_edits, cds50_scope = scope_singles[REGION_CDS_FIRST50]
            if utr_singles and cds50_edits:
                joint_scope = utr + cds50_scope
                gen = joint_doubles(utr_singles, cds50_edits, utr, cds50_scope, nbc, rng)
                for etype in ("random_double", "structure_guided_double"):
                    for group in gen[etype]:
                        group = list(group)
                        cand_scope = apply_edits(joint_scope, group)
                        full_new = _apply_joint_edits_to_full_cds(
                            cds, utr, cds50_scope, group)
                        if not protein_identical(cds, full_new):
                            # fail-closed: never emitted (see make_cds_rec)
                            stats["protein_identity_failures"] += 1
                            continue
                        records.append(BenchmarkRecord(
                            source_id=s["transcript_id"], cargo_id=cargo_id,
                            cell_context="none_unlabeled",
                            source_sequence=joint_scope, candidate_sequence=cand_scope,
                            edit_list=[e.to_dict() for e in group], edit_count=len(group),
                            edited_region=REGION_JOINT, protein_identity=True,
                            measured_or_proxy_source_value=None,
                            measured_or_proxy_candidate_value=None, delta=None,
                            data_source="p0_data_reconstruction_v1",
                            assay_type=UNLABELED_ASSAY, confidence="unlabeled",
                            edit_type=etype, task_eligibility="task_c_locked_extension",
                            value_qualifier=UNLABELED_QUALIFIER,
                            family_cluster_id=s["family_cluster_id"],
                            motif_flags=motif_policy_v1_guarded_risk_flags(joint_scope, cand_scope),
                            internal_features={"rank_feature": "structure_disruption"},
                        ).finalize())

    return records, stats


def _scope_codon_offset(region: str) -> int:
    """Codon index at which a CDS scope starts within the full CDS.

    ``cds_singles`` clamps every window's lower bound to codon 1 (the start
    codon is never editable), so first30/first50 scopes begin at codon 1.
    """
    return {
        REGION_CDS_FIRST30: 1,
        REGION_CDS_FIRST50: 1,
        REGION_CDS_REMAINING: 50,
    }[region]


def _apply_scope_edits_to_full_cds(
    cds: str, scope_seq: str, region: str, edits: Sequence[Any]
) -> str:
    offset_nt = 3 * _scope_codon_offset(region)
    # sanity: scope must match the full CDS at the offset
    assert cds[offset_nt:offset_nt + len(scope_seq)] == scope_seq, "scope/full CDS mismatch"
    seq = list(cds)
    for e in edits:
        pos = offset_nt + e.pos
        assert seq[pos] == e.ref, "ref mismatch applying scope edit"
        seq[pos] = e.alt
    return "".join(seq)


def _apply_joint_edits_to_full_cds(
    cds: str, utr: str, cds50_scope: str, edits: Sequence[Any]
) -> str:
    shift = len(utr)
    # joint CDS edits are scoped to the cds_first50 scope, which starts at
    # codon 1 of the full CDS (start codon is never editable).
    scope_off = 3 * _scope_codon_offset(REGION_CDS_FIRST50)
    seq = list(cds)
    for e in edits:
        if e.pos >= shift:  # CDS edit (5'UTR edits don't touch the CDS)
            pos = e.pos - shift + scope_off
            assert seq[pos] == e.ref, "ref mismatch applying joint edit"
            seq[pos] = e.alt
    return "".join(seq)
