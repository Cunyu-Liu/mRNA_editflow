#!/usr/bin/env python3
"""P3-01 driver: build the local-edit benchmark, global leakage-safe split,
and frozen manifests.

Pipeline (no GRPO training; data assets only):
    1. measured tier   : Sample2019 MPRA snv library (wild-type anchored)
    2. proxy tier      : reconstructed human mRNA 5'UTR neighborhoods scored by
                         the frozen P1-04 CNN-50mer cross-fitted ensemble
                         (predicted/internal proxy; first-50nt window only)
    3. unlabeled tier  : legal edits with ALL value fields null
                         (5'UTR outside proxy window + synonymous CDS scopes
                         + joint 5'UTR+CDS; data assets for gated tasks b/c)
    4. GLOBAL split    : one group per source_id across ALL tiers (same source
                         never crosses roles); per-cohort OOD flagging; whole-
                         family test holdout; source-disjoint train/val
    5. freeze          : tier JSONL SHA-256, split assignment SHA-256, dataset
                         registry (Task 1 four-level), benchmark manifest
                         (Task 5), split audit (Task 4)

Usage (server, repo root):
    python3 scripts/p3_01_build_benchmark.py --repo-root . [--smoke] [--skip-proxy]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Import bootstrap: the repo directory itself is the ``mrna_editflow`` package,
# so its PARENT must be importable.
# ---------------------------------------------------------------------------

def _bootstrap(repo_root: Path) -> None:
    parent = str(repo_root.resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)


# ---------------------------------------------------------------------------
# Four-level dataset registry (Task 1)
# ---------------------------------------------------------------------------

# Static level assignment + rationale (the intellectual content of Task 1).
# Fields are merged with the on-disk raw manifests (license/citation/hashes).
LEVEL_MAP: Dict[str, Dict[str, Any]] = {
    "level_a_observational": {
        "definition": "Natural-sequence absolute labels (observational). "
                      "Pretraining / background analysis ONLY; never used as "
                      "local-delta ground truth.",
        "entries": {
            "cao2021_5utr": {
                "role": "endogenous 5'UTR TE (Ribo-seq) in 3 cell types",
                "why_not_local_delta": "cross-gene absolute TE; no source-matched edits",
            },
            "saluki_halflife": {
                "role": "mRNA half-life labels",
                "why_not_local_delta": "cross-transcript absolute half-life",
            },
            "codonbert_stability": {
                "role": "mRNA stability labels",
                "why_not_local_delta": "cross-transcript absolute stability",
            },
            "refseq_gencode_catalog": {
                "role": "sequence assets feeding p0_data_reconstruction_v1",
                "why_not_local_delta": "no functional labels at all",
            },
        },
    },
    "level_b_cross_construct": {
        "definition": "Cross-construct design data (many DIFFERENT constructs, "
                      "absolute labels). Ranking / analysis ONLY; not "
                      "source-matched perturbation.",
        "entries": {
            "sample2019_mpra_random": {
                "role": "random 50-mer / varying-length MPRA libraries "
                       "(GSM3130441/442 mCherry, GSM4084997 25-100nt)",
                "why_not_local_delta": "random constructs share no anchored wild-type source",
            },
            "lepplek2022_persistseq": {
                "role": "233 full-length mRNA designs with protein output over time",
                "why_not_local_delta": "cross-design comparison, not local edits of one source",
            },
            "khoroshkin2024_parade": {
                "role": "cross-construct mRNA design dataset",
                "why_not_local_delta": "cross-design absolute labels",
            },
        },
    },
    "level_c_source_matched": {
        "definition": "Source-matched local perturbation data (CORE). Only this "
                      "level may anchor local-delta claims.",
        "entries": {
            "sample2019_mpra_snv": {
                "role": "MEASURED tier: snv library (GSM3130443), wild-type-anchored "
                       "single/double/multi 5'UTR variants, MPRA ribosome load",
                "why_not_local_delta": "n/a — this IS local-delta ground truth "
                                       "(wet-lab measured, single cargo/cell context)",
            },
            "p3_01_proxy_neighborhood": {
                "role": "PROXY tier: reconstructed 5'UTR controlled neighborhoods "
                       "scored by frozen CNN-50mer cross-fitted ensemble (P1-04)",
                "why_not_local_delta": "predicted/internal proxy — NOT wet-lab; "
                                       "value_qualifier marks it on every record",
            },
        },
    },
    "level_d_prospective_intervention": {
        "definition": "Prospective internal intervention data (held-out wet-lab "
                      "assay of model-proposed edits). NONE exists yet; planned "
                      "in P3-03. Registered empty by design.",
        "entries": {},
    },
}


def build_dataset_registry(repo_root: Path) -> Dict[str, Any]:
    """Assemble the machine-readable four-level registry from on-disk manifests."""
    raw = repo_root / "data" / "raw"

    def _load_manifest(ds: str) -> Dict[str, Any]:
        p = raw / ds / "manifest.json"
        if not p.exists():
            return {}
        with open(p) as fh:
            return json.load(fh)

    def _file_entries(ds: str) -> List[Dict[str, Any]]:
        man = _load_manifest(ds)
        out = []
        for f in man.get("files", []):
            out.append({
                "local_path": f.get("local_path"),
                "sha256": f.get("sha256"),
                "record_count": f.get("record_count"),
                "source_url": f.get("source_url"),
                "license": f.get("license"),
                "citation": f.get("citation"),
            })
        return out

    registry: Dict[str, Any] = {
        "registry_id": "p3_01_dataset_registry",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "levels": {},
    }

    ds_manifest_names = {
        "cao2021_5utr": "cao2021_5utr",
        "saluki_halflife": "saluki_halflife",
        "codonbert_stability": "codonbert_stability",
        "lepplek2022_persistseq": "lepplek2022_persistseq",
        "khoroshkin2024_parade": "khoroshkin2024_parade",
    }
    for level_key, level in LEVEL_MAP.items():
        entries_out = []
        for ds_id, meta in level["entries"].items():
            if ds_id == "refseq_gencode_catalog":
                entries_out.append({
                    "dataset_id": ds_id,
                    **meta,
                    "files": [
                        {"local_path": "data/raw/human.1.rna.gbff.gz"},
                        {"local_path": "data/raw/gencode.v45.pc_transcripts.fa.gz"},
                    ],
                    "reconstruction_manifest":
                        "data/reconstructed/p0_data_reconstruction_v1/combined/"
                        "combined_reconstruction_manifest.json",
                })
                continue
            if ds_id == "sample2019_mpra_random":
                files = [f for f in _file_entries("sample2019_mpra")
                         if f["local_path"] and "designed_library" not in f["local_path"]]
                entries_out.append({"dataset_id": ds_id, **meta, "files": files})
                continue
            if ds_id == "sample2019_mpra_snv":
                files = [f for f in _file_entries("sample2019_mpra")
                         if f["local_path"] and "designed_library" in f["local_path"]]
                entries_out.append({"dataset_id": ds_id, **meta, "files": files})
                continue
            if ds_id == "p3_01_proxy_neighborhood":
                entries_out.append({
                    "dataset_id": ds_id, **meta,
                    "files": [{"local_path": "data/p3/benchmark/proxy_tier.jsonl",
                               "sha256": "<filled by benchmark manifest>"}],
                })
                continue
            manifest_key = ds_manifest_names.get(ds_id)
            if manifest_key:
                man = _load_manifest(manifest_key)
                entries_out.append({
                    "dataset_id": ds_id,
                    **meta,
                    "license": man.get("license"),
                    "citation": man.get("citation"),
                    "description": man.get("description"),
                    "files": _file_entries(manifest_key),
                })
            else:
                entries_out.append({"dataset_id": ds_id, **meta})
        registry["levels"][level_key] = {
            "definition": level["definition"],
            "n_entries": len(entries_out),
            "entries": entries_out,
        }
    return registry


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _gc(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in ("G", "C")) / len(seq)


def build_global_groups(
    tier_records: Mapping[str, Sequence[Any]],
    source_meta: Mapping[str, Dict[str, Any]],
) -> List[Any]:
    """One SourceGroup per source_id, merged ACROSS tiers (leakage-atomic)."""
    from mrna_editflow.data.p3_split import SourceGroup

    counts: Dict[str, int] = {}
    families: Dict[str, str] = {}
    for tier, records in tier_records.items():
        for rec in records:
            counts[rec.source_id] = counts.get(rec.source_id, 0) + 1
            if rec.family_cluster_id is not None:
                families[rec.source_id] = str(rec.family_cluster_id)
    groups: List[SourceGroup] = []
    for source_id, n in counts.items():
        meta = source_meta[source_id]
        groups.append(SourceGroup(
            group_id=source_id,
            family_id=families.get(source_id, f"singleton:{source_id}"),
            n_records=n,
            gc=meta["gc"],
            length=meta["length"],
            cohort=meta["cohort"],
        ))
    return groups


def apply_roles(
    tier_records: Mapping[str, Sequence[Any]],
    roles: Mapping[str, str],
) -> Dict[str, int]:
    """Set split_role on every record from the global source->role map."""
    per_tier: Dict[str, int] = {}
    for tier, records in tier_records.items():
        n = 0
        for rec in records:
            role = roles.get(rec.source_id)
            if role is None:
                raise KeyError(f"source {rec.source_id} missing split role")
            rec.split_role = role
            n += 1
        per_tier[tier] = n
    return per_tier


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-dir", default="data/p3")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny deterministic build (16 proxy / 8 cds sources)")
    ap.add_argument("--skip-proxy", action="store_true",
                    help="measured + unlabeled only (no CNN ensemble needed)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--audit-out", default="docs/p3_01_split_audit.json",
                    help="split audit output path (absolute, or repo-relative)")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    _bootstrap(repo_root)
    os.chdir(repo_root)

    from mrna_editflow.data.p3_benchmark_builder import (
        BuildConfig,
        build_measured_tier,
        build_proxy_tier,
        build_unlabeled_cds_tier,
        checkpoint_hashes,
        filter_eligible_sources,
        load_ensemble,
        load_reconstructed_sources,
        load_snv_rows,
        select_proxy_sources,
        RECORDS_JSONL,
        FAMILY_JSON,
        RECON_MANIFEST,
        SNV_CSV,
    )
    from mrna_editflow.data.p3_local_edit_schema import (
        SCHEMA_VERSION,
        read_benchmark_jsonl,
        sha256_file,
        write_benchmark_jsonl,
    )
    from mrna_editflow.data.p3_split import (
        SplitConfig,
        assign_split_roles,
        audit_split,
        resolve_cross_role_sequence_collisions,
        split_assignment_sha256,
        canonical_assignment_json,
    )
    import random

    t0 = time.time()
    cfg = BuildConfig(seed=args.seed, device=args.device,
                      torch_threads=args.torch_threads, smoke=args.smoke).effective()
    out_dir = repo_root / args.out_dir
    bench_dir = out_dir / "benchmark"
    man_dir = out_dir / "manifests"
    bench_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "docs").mkdir(exist_ok=True)

    summary: Dict[str, Any] = {
        "phase": "P3-01",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "seed": cfg.seed, "smoke": cfg.smoke, "skip_proxy": args.skip_proxy,
            "proxy_sources": cfg.proxy_sources, "cds_sources": cfg.cds_sources,
            "device": cfg.device, "torch_threads": cfg.torch_threads,
        },
        "tiers": {},
    }

    # ------------------------------------------------------------------
    # 1. measured tier
    # ------------------------------------------------------------------
    print("[1/6] measured tier (Sample2019 snv) ...", flush=True)
    snv_rows = load_snv_rows(SNV_CSV)
    measured_records, measured_stats = build_measured_tier(snv_rows)
    summary["tiers"]["measured"] = {"stats": measured_stats}
    print(f"      rows={measured_stats['n_rows']} mothers={measured_stats['n_mothers']} "
          f"anchored={measured_stats['n_anchored_mothers']} "
          f"records={len(measured_records)}", flush=True)

    # ------------------------------------------------------------------
    # 2. reconstructed sources
    # ------------------------------------------------------------------
    print("[2/6] reconstructed sources + eligibility ...", flush=True)
    sources = load_reconstructed_sources(RECORDS_JSONL, FAMILY_JSON)
    eligible, reject_reasons = filter_eligible_sources(sources)
    summary["eligibility"] = {
        "n_input_sources": len(sources),
        "n_eligible": len(eligible),
        "reject_reasons": reject_reasons,
    }
    print(f"      input={len(sources)} eligible={len(eligible)} "
          f"rejects={reject_reasons}", flush=True)

    # ------------------------------------------------------------------
    # 3. proxy tier (+ unlabeled 5'UTR byproduct)
    # ------------------------------------------------------------------
    proxy_records: List[Any] = []
    unlabeled_five_utr: List[Any] = []
    proxy_stats: Dict[str, Any] = {"skipped": bool(args.skip_proxy)}
    if not args.skip_proxy:
        print("[3/6] proxy tier (CNN-50mer ensemble, CPU) ...", flush=True)
        proxy_sources = select_proxy_sources(eligible, cfg)
        models = load_ensemble(str(repo_root), cfg.device, cfg.torch_threads)
        print(f"      sources={len(proxy_sources)} models={len(models)}", flush=True)
        proxy_records, unlabeled_five_utr, proxy_stats = build_proxy_tier(
            proxy_sources, models, cfg)
        print(f"      proxy_records={len(proxy_records)} "
              f"unlabeled_5utr={len(unlabeled_five_utr)} "
              f"scored_seqs={proxy_stats.get('n_scored_sequences')}", flush=True)
    else:
        proxy_sources = []
        print("[3/6] proxy tier SKIPPED (--skip-proxy)", flush=True)
    summary["tiers"]["proxy"] = {"stats": proxy_stats}

    # ------------------------------------------------------------------
    # 4. unlabeled CDS + joint tier
    # ------------------------------------------------------------------
    print("[4/6] unlabeled CDS/joint tier ...", flush=True)
    rng = random.Random(f"{cfg.seed}:cds_source_pick")
    cds_pool = [s for s in eligible if len(s["cds"]) // 3 >= 52]
    cds_pool = list(cds_pool)
    rng.shuffle(cds_pool)
    cds_sources = cds_pool[: cfg.cds_sources]
    cds_records, cds_stats = build_unlabeled_cds_tier(cds_sources, cfg)
    unlabeled_records = list(unlabeled_five_utr) + list(cds_records)
    summary["tiers"]["unlabeled"] = {"stats": {
        "n_five_utr_records": len(unlabeled_five_utr),
        "cds_tier": cds_stats,
    }}
    print(f"      sources={len(cds_sources)} records={len(cds_records)} "
          f"protein_identity_failures={cds_stats['protein_identity_failures']}", flush=True)

    tier_records: Dict[str, Sequence[Any]] = {
        "measured": measured_records,
        "proxy": proxy_records,
        "unlabeled": unlabeled_records,
    }

    # ------------------------------------------------------------------
    # 5. GLOBAL leakage-safe split
    # ------------------------------------------------------------------
    print("[5/6] global split + audit ...", flush=True)
    # source metadata for grouping (gc/length/cohort of the SOURCE sequence)
    source_meta: Dict[str, Dict[str, Any]] = {}
    for rec in measured_records:
        if rec.edit_count == 0:  # wild-type anchor carries the mother sequence
            source_meta[rec.source_id] = {
                "gc": _gc(rec.source_sequence),
                "length": len(rec.source_sequence),
                "cohort": "sample2019_snv",
            }
    seen_src: Dict[str, Dict[str, Any]] = {}
    for s in list(proxy_sources) + list(cds_sources):
        seen_src[s["transcript_id"]] = s
    for s in seen_src.values():
        source_meta[s["transcript_id"]] = {
            "gc": _gc(s["five_utr"]),
            "length": len(s["five_utr"]),
            "cohort": "reconstructed",
        }

    groups = build_global_groups(tier_records, source_meta)
    split_cfg = SplitConfig(seed=cfg.seed)
    roles, split_meta = assign_split_roles(groups, split_cfg)
    assignment_sha = split_assignment_sha256(roles)
    per_tier_counts = apply_roles(tier_records, roles)
    print(f"      groups={len(groups)} assignment_sha256={assignment_sha[:16]}... "
          f"role_records={split_meta['role_record_counts']}", flush=True)

    # Cross-role exact-sequence collision resolution: near-homologous sources
    # in different families can emit byte-identical candidates; keep the
    # highest-priority role (test > ood > val > train), drop the rest.
    dropped_ids, collision_stats = resolve_cross_role_sequence_collisions(tier_records)
    if dropped_ids:
        for tier in list(tier_records.keys()):
            tier_records[tier] = [r for r in tier_records[tier]
                                  if r.record_id not in dropped_ids]
    split_meta["role_record_counts"] = {
        role: sum(1 for recs in tier_records.values() for r in recs
                  if r.split_role == role)
        for role in ("train", "val", "test", "ood")
    }
    split_meta["collision_resolution"] = collision_stats
    collision_art = {
        "resolution_id": "p3_01_collision_resolution",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stats": collision_stats,
        "dropped_record_ids": sorted(dropped_ids),
    }
    with open(man_dir / "p3_01_collision_resolution.json", "w") as fh:
        json.dump(collision_art, fh, indent=1, sort_keys=True)
    print(f"      collisions={collision_stats['n_collision_sequences']} "
          f"dropped={collision_stats['n_dropped_records']} "
          f"by_role={collision_stats['drops_by_role']}", flush=True)

    # Write tier JSONLs (split_role baked in)
    def _display_path(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root))
        except ValueError:
            return str(p)

    tier_paths: Dict[str, str] = {}
    tier_hashes: Dict[str, str] = {}
    for tier, records in tier_records.items():
        path = bench_dir / f"{tier}_tier.jsonl"
        n = write_benchmark_jsonl(records, str(path))
        tier_paths[tier] = _display_path(path)
        tier_hashes[tier] = sha256_file(str(path))
        summary["tiers"][tier]["n_records"] = n
        summary["tiers"][tier]["path"] = tier_paths[tier]
        summary["tiers"][tier]["sha256"] = tier_hashes[tier]
        print(f"      wrote {tier_paths[tier]} n={n} sha256={tier_hashes[tier][:16]}...",
              flush=True)

    # Audit per tier (streaming) + global
    audits: Dict[str, Any] = {}
    for tier in ("measured", "proxy", "unlabeled"):
        audits[tier] = audit_split(
            (r.to_dict() for r in read_benchmark_jsonl(tier_paths[tier])), tier)
    global_records_iter = (rec.to_dict()
                           for tier in ("measured", "proxy", "unlabeled")
                           for rec in read_benchmark_jsonl(tier_paths[tier]))
    audits["global"] = audit_split(global_records_iter, "global")
    audit_doc = {
        "audit_id": "p3_01_split_audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_config": {
            "train_frac": split_cfg.train_frac, "val_frac": split_cfg.val_frac,
            "test_frac": split_cfg.test_frac, "seed": split_cfg.seed,
            "ood_gc_quantile": split_cfg.ood_gc_quantile,
            "ood_length_quantile": split_cfg.ood_length_quantile,
            "per_cohort_ood": True,
        },
        "assignment_sha256": assignment_sha,
        "split_metadata": split_meta,
        "audits": audits,
        "passed": all(a["passed"] for a in audits.values()),
    }
    audit_out = Path(args.audit_out)
    audit_path = audit_out if audit_out.is_absolute() else (repo_root / audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as fh:
        json.dump(audit_doc, fh, indent=2, sort_keys=True)
    print(f"      audit passed={audit_doc['passed']} -> {audit_path}", flush=True)

    # ------------------------------------------------------------------
    # 6. Freeze manifests
    # ------------------------------------------------------------------
    print("[6/6] freeze manifests ...", flush=True)
    # split assignment (committed; small)
    assignment_path = man_dir / "p3_01_split_assignment.json"
    with open(assignment_path, "w") as fh:
        json.dump({
            "assignment_sha256": assignment_sha,
            "seed": split_cfg.seed,
            "roles": dict(sorted(roles.items())),
        }, fh, indent=1, sort_keys=True)

    # dataset registry (Task 1)
    registry = build_dataset_registry(repo_root)
    registry_path = man_dir / "p3_01_dataset_registry.json"
    with open(registry_path, "w") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)

    # benchmark manifest (Task 5)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
    except Exception:
        git_commit = "unknown"
    manifest = {
        "manifest_id": "p3_01_local_edit_benchmark",
        "created_utc": summary["created_utc"],
        "schema_version": SCHEMA_VERSION,
        "dataset_card": {
            "name": "mRNA-EditFlow P3-01 Local-Edit Benchmark",
            "task_contract": "configs/p3_primary_task.yaml (p3_task_v2, frozen)",
            "primary_task": "task_a_active (5'UTR substitution, edit_budget <= 10)",
            "gated_tasks": ["task_b_frozen_fallback (synonymous CDS)",
                             "task_c_locked_extension (joint 5'UTR+CDS)"],
            "endpoint_semantics": {
                "measured": "wet-lab measured MPRA ribosome load (Sample 2019); "
                            "delta = variant - wild-type of the SAME mother",
                "proxy": "predicted/internal proxy MRL (CNN-50mer cross-fitted "
                         "ensemble, P1-04); first-50nt 5'UTR window only",
                "unlabeled": "no values by contract (null); NOT local-delta "
                             "ground truth",
            },
        },
        "tiers": {
            tier: {
                "path": tier_paths[tier],
                "sha256": tier_hashes[tier],
                "n_records": summary["tiers"][tier]["n_records"],
            } for tier in tier_paths
        },
        "inputs": {
            "sample2019_snv_csv": {"path": SNV_CSV, "sha256": sha256_file(SNV_CSV)},
            "reconstructed_records": {"path": RECORDS_JSONL,
                                       "sha256": sha256_file(RECORDS_JSONL)},
            "family_assignments": {"path": FAMILY_JSON,
                                    "sha256": sha256_file(FAMILY_JSON)},
            "reconstruction_manifest": RECON_MANIFEST,
            "cnn50mer_checkpoint_sha256": (
                {} if args.skip_proxy else checkpoint_hashes(str(repo_root))),
        },
        "filter_rules": {
            "source_eligibility": [
                "five_utr length >= 50 nt",
                "ACGU alphabet (5'UTR and CDS)",
                "valid CDS (AUG start, stop end, no internal stop, len%3==0)",
                "no upstream in-frame start codon in source 5'UTR",
                "no homopolymer run >= 6 nt (5'UTR, first 150 nt of CDS)",
                "no cryptic splice consensus in source 5'UTR",
            ],
            "edit_legality": [
                "substitutions only (no indels; length preserved)",
                "CDS edits synonymous by construction; protein identity "
                "verified against full CDS on every CDS/joint record",
                "start/stop codon never edited",
                "motif_policy_v1 hard_forbidden tier excluded at action space",
                "edit_count <= 10 (p3_task_v2 budget)",
            ],
            "measured_tier": [
                "variants anchored to wild-type of the SAME mother only",
                "hamming distance 1..10 retained; others excluded",
                "wild-type anchor value = median rl of WT replicates",
            ],
        },
        "split": {
            "assignment_sha256": assignment_sha,
            "assignment_path": str(assignment_path),
            "audit_path": str(audit_path),
            "audit_passed": audit_doc["passed"],
            "role_record_counts": split_meta["role_record_counts"],
            "collision_resolution": collision_stats,
            "collision_resolution_path":
                str(man_dir / "p3_01_collision_resolution.json"),
            "grouping": "one group per source_id across ALL tiers; test holds "
                        "whole protein families; train/val source-disjoint; "
                        "OOD flagged per-cohort on GC/length tails",
        },
        "known_confounders": [
            "measured tier: single cargo (EGFP), single cell context (HEK293T), "
            "50 nt 5'UTR window, ribosome-load endpoint (not protein output)",
            "proxy tier: CNN-50mer ensemble was trained on the SAME MPRA assay "
            "family as the measured tier (assay-family correlation; NOT an "
            "independent oracle)",
            "measured mothers are human SNP-context 50-mers, not therapeutic UTRs",
            "GC/length distributions differ between measured and reconstructed "
            "cohorts (handled by per-cohort OOD flagging)",
            "unlabeled tier carries NO values; any downstream use as local-delta "
            "ground truth is a contract violation",
            "5'UTR proxy coverage limited to first 50 nt; longer-UTR edits are "
            "unlabeled by design",
        ],
        "build": {
            "seed": cfg.seed,
            "smoke": cfg.smoke,
            "code_git_commit": git_commit,
            "command": " ".join(sys.argv),
            "runtime_sec": round(time.time() - t0, 2),
        },
    }
    manifest_path = man_dir / "p3_01_benchmark_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"      manifests -> {man_dir}", flush=True)

    summary["split"] = {
        "assignment_sha256": assignment_sha,
        "audit_passed": audit_doc["passed"],
        "role_record_counts": split_meta["role_record_counts"],
        "role_group_counts": split_meta["role_group_counts"],
    }
    summary["runtime_sec"] = round(time.time() - t0, 2)
    summary_path = out_dir / "p3_01_build_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print("\n==== P3-01 BUILD SUMMARY ====", flush=True)
    print(json.dumps({
        "tiers": {t: summary["tiers"][t].get("n_records") for t in summary["tiers"]},
        "split": summary["split"],
        "runtime_sec": summary["runtime_sec"],
    }, indent=2), flush=True)
    return 0 if audit_doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
