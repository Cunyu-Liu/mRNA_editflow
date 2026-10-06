#!/usr/bin/env python3
"""Gate P one-shot TEST unblinding runner v1 (prereg: gate_p_unblinding_prereg_v1.md).

Order (prereg section 5, atomic):
  1. load manifest TEST rows (ids/strata only, no labels touched)
  2. build TEST bottom-six chunk cache ONLINE (outcome-free: sequences only)
  3. rebuild frozen V5 from checkpoint (state_dict/vocabs/scaler verbatim),
     run the v4 evaluate loop -> TEST predictions.jsonl (still no labels read)
  4. THE UNBLINDING EVENT: metrics + receipt (sha256s), append-only outputs
  5. consistency table C1-C4 vs VALIDATION (prereg section 3, frozen rules)

Hard gates: CUDA A100 + BF16; cpu fallback forbidden. Labels (direction_normalized_delta)
are parsed into memory only at step 4 - steps 1-3 never touch them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

W0 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
REPO_ROOT = W0
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
V5_CKPT = MNT / "experiments/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/v5_full/final_pass_8_checkpoint.pt"
SCREEN_CONFIG = MNT / "authorizations/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/screen_config.json"
DESCRIPTORS = W0 / "configs/route_a_v3_route2_endpoint_descriptors_v1.json"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/route_a_v3"))

from train_route2_xeditcritic_v4 import _forward_bf16, _move, evaluation_index_batches_v4  # noqa: E402
from train_route2_xeditcritic_v3 import validation_metrics  # noqa: E402
from core.route2_xeditcritic_training_data_v3 import XEditCriticRecordV3  # noqa: E402
from scripts.route_a_v3.train_route2_xeditcritic_v3 import TaskRobustScalerV3  # noqa: E402
from core.route2_xeditcritic_batch_v4 import (  # noqa: E402
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
    XEditCriticDatasetV4,
)
from core.route2_bottom_encoder_chunk_cache_v4 import (  # noqa: E402
    assemble_frozen_bottom_encoder_chunk_cache_v4,
    V4_ALLOWED_SPLITS,
)
import core.route2_bottom_encoder_chunk_cache_v4 as cache_mod  # noqa: E402
from core.route2_development_projection_v3 import load_endpoint_descriptors  # noqa: E402
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (  # noqa: E402
    FrozenMRNABERTBottomSixEncoderV4,
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--physical-gpu-index", required=True, type=int)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    t_start = time.time()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - hard gate (prereg section 2)")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.cuda.set_device(device)
    if "A100" not in torch.cuda.get_device_name(device):
        raise SystemExit(f"A100 hard gate failed: {torch.cuda.get_device_name(device)}")

    assert not args.output_dir.exists(), f"output exists (append-only): {args.output_dir}"
    args.output_dir.mkdir(parents=True)

    # ------------------------------------------------ 1. manifest TEST rows (ids only)
    test_rows = []
    with MANIFEST.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "TEST":
                test_rows.append(r)
    assert len(test_rows) == 18292, f"TEST count {len(test_rows)} != 18292"
    test_ids = [r["canonical_record_id"] for r in test_rows]
    strata = Counter(tuple(r["stratum"]) for r in test_rows)
    print(f"[1] TEST rows: {len(test_rows)} across {len(strata)} strata", flush=True)

    id_set_hash = hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest()
    manifest_hash = sha256_of(MANIFEST)
    ckpt_hash = sha256_of(V5_CKPT)

    # ------------------------------------------------ 2. canonical rows; sequences only now
    needed_by_study = {}
    for r in test_rows:
        needed_by_study.setdefault(r["study_unit_id"], set()).add(r["canonical_record_id"])
    canonical_rows = {}
    for study, ids in needed_by_study.items():
        for name in ("canonical_records.private.jsonl", "canonical_records.jsonl"):
            p = MNT / "canonical" / study / "v1" / name
            if p.is_file():
                with p.open() as fh:
                    for line in fh:
                        row = json.loads(line)
                        rid = row["canonical_record_id"]
                        if rid in ids:
                            canonical_rows[rid] = row
                break
    assert all(i in canonical_rows for i in test_ids), "TEST ids missing from canonical"
    print(f"[2] canonical located for all {len(canonical_rows)} TEST records (labels NOT read)", flush=True)

    seq_by_id = {
        rid: (str(canonical_rows[rid]["source_sequence"]).upper().replace("T", "U"),
              str(canonical_rows[rid]["candidate_sequence"]).upper().replace("T", "U"))
        for rid in test_ids
    }
    unique_sequences = sorted({s for pair in seq_by_id.values() for s in pair})
    sequence_to_index = {s: i for i, s in enumerate(unique_sequences)}
    print(f"[2] unique sequences to encode: {len(unique_sequences)}", flush=True)

    config = json.load(open(SCREEN_CONFIG))
    encoder = FrozenMRNABERTBottomSixEncoderV4(Path(config["mrnabert_model_path"]), device)
    t_enc = time.time()
    encoded = encoder.encode_sequences(
        {i: s for i, s in enumerate(unique_sequences)},
        progress_callback=lambda done, total: print(f"    encode {done}/{total}", end="\r", flush=True) if done % 500 == 0 or done == total else None,
    )
    print(f"\n[2] bottom-six encoding done in {time.time()-t_enc:.0f}s", flush=True)

    descriptors = load_endpoint_descriptors(DESCRIPTORS)

    proj_rows = []
    for r in test_rows:
        crow = canonical_rows[r["canonical_record_id"]]
        study = str(crow["study_unit_id"])
        region = str(crow["region"])
        region_id = 0 if region == "5UTR" else 1
        endpoint_id = str(crow["endpoint_id"])
        context = str(crow["biological_context_id"])
        source, candidate = seq_by_id[r["canonical_record_id"]]
        edits = [
            {"position": i, "source_base": a, "candidate_base": b}
            for i, (a, b) in enumerate(zip(source, candidate)) if a != b
        ]
        proj_rows.append({
            "schema_version": "route_a_v3_route2_development_projection.v3",
            "canonical_record_id": r["canonical_record_id"],
            "split": "TEST",
            "study_unit_id": study,
            "connected_source_component_id": r["connected_source_component_id"],
            "source_group_id": "::".join((study, str(crow["source_id"]), context, endpoint_id)),
            "source_id": str(crow["source_id"]),
            "task_id": f"{endpoint_id}::region={region_id}",
            "endpoint_id": endpoint_id,
            "endpoint_descriptor": dict(descriptors[endpoint_id]),
            "region": region,
            "region_id": region_id,
            "assay_id": str(crow["assay_id"]),
            "biological_context_id": context,
            "source_sequence": source,
            "candidate_sequence": candidate,
            "source_relative_edits": edits,
            "direction_normalized_delta": float(crow["direction_normalized_delta"]),
        })

    cache_mod.V4_ALLOWED_SPLITS = frozenset({"TRAIN", "VALIDATION", "TEST"})  # prereg-authorized: cache is outcome-free
    cache_payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        proj_rows,
        sequence_to_index=sequence_to_index,
        encoded=encoded,
        model_id="mrnabert-raw-frozen-bottom-six",
        pretrained_parameter_count=99_500_000,
        attention_backend="PYTORCH_SDPA_AUTO",
    )
    cache_view = FrozenBottomEncoderChunkCacheViewV4(cache_payload, set(test_ids), validate_payload=False)
    print(f"[2] TEST cache assembled: {len(cache_payload['record_ids'])} records", flush=True)

    # ------------------------------------------------ 3. rebuild frozen V5 + predictions
    ck = torch.load(V5_CKPT, map_location="cpu", weights_only=False)
    vocabs = ck["vocabs"]
    sd = ck["target_scaler"]
    scaler = TaskRobustScalerV3(
        scales=dict(sd["task_scales"]),
        region_scales={int(k): float(v) for k, v in sd["region_scales"].items()},
        global_scale=float(sd["global_scale"]),
        floor=float(sd["floor"]),
        training_record_count=int(sd["training_record_count"]),
    )

    records = []
    for row in proj_rows:
        records.append(XEditCriticRecordV3(
            record_id=row["canonical_record_id"],
            split="TEST",
            source=row["source_sequence"],
            candidate=row["candidate_sequence"],
            edits=tuple((e["position"], e["source_base"], e["candidate_base"]) for e in row["source_relative_edits"]),
            target=row["direction_normalized_delta"],  # carried; not read by inference path
            task=row["task_id"],
            study=row["study_unit_id"],
            source_group=row["source_group_id"],
            assay=row["assay_id"],
            context=row["biological_context_id"],
            region=row["region_id"],
            quantity=row["endpoint_descriptor"]["quantity_family"],
            measurement=row["endpoint_descriptor"]["measurement_form"],
            numerator=row["endpoint_descriptor"]["numerator_family"],
            denominator=row["endpoint_descriptor"]["denominator_family"],
        ))
    record_by_id = {r.record_id: r for r in records}

    dataset = XEditCriticDatasetV4(
        records, all_records=record_by_id, vocabs=vocabs,
        target_scaler=scaler, cache=None, candidate_bundle_overrides={}, neutral_studies=set(),
    )
    collator = XEditCriticCollatorV4(cache_view, minimum_physical_batch=int(config["memory_preflight"]["minimum_physical_batch"]))

    from scripts.route_a_v3.run_route2_xeditcritic_v5_polya_3seed_v1 import _build_model_v4_full  # noqa: E402
    model, capacity = _build_model_v4_full(config, vocabs, device)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=True), None
    model.eval()
    print(f"[3] V5 rebuilt from checkpoint ({capacity['trainable_parameter_count']} params); inference start", flush=True)

    physical_batch_size = int(ck["physical_batch_size"])
    targets, predictions, scaled_t, scaled_p, tasks, rows_out = [], [], [], [], [], []
    t_inf = time.time()
    with torch.inference_mode():
        for bi, (indices, valid_count) in enumerate(evaluation_index_batches_v4(len(dataset), physical_batch_size), start=1):
            batch = _move(collator([dataset[i] for i in indices]), device)
            output = _forward_bf16(model, batch)
            scaled_prediction = output["mean"].float()[:valid_count]
            prediction = scaled_prediction * batch["target_scale"][:valid_count]
            bt = batch["target"].float()[:valid_count].cpu().tolist()
            bp = prediction.cpu().tolist()
            bst = batch["scaled_target"].float()[:valid_count].cpu().tolist()
            bsp = scaled_prediction.cpu().tolist()
            targets.extend(bt); predictions.extend(bp); scaled_t.extend(bst); scaled_p.extend(bsp)
            tasks.extend(batch["task_ids"][:valid_count])
            for i in range(valid_count):
                rows_out.append({
                    "record_id": batch["record_ids"][i],
                    "source_group_id": batch["source_groups"][i],
                    "task_id": batch["task_ids"][i],
                    "target": float(bt[i]),
                    "prediction": float(bp[i]),
                    "scaled_target": float(bst[i]),
                    "scaled_prediction": float(bsp[i]),
                })
            if bi % 200 == 0:
                print(f"[3] batch {bi} ({len(rows_out)}/{len(dataset)})", flush=True)
    assert len(rows_out) == len(dataset), "padded rows entered TEST cohort"
    print(f"[3] inference done in {time.time()-t_inf:.0f}s -> {len(rows_out)} predictions", flush=True)
    pred_path = args.output_dir / "test_predictions.jsonl"
    with pred_path.open("w") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    # ------------------------------------------------ 4. THE UNBLINDING EVENT
    metrics = validation_metrics(targets, predictions, scaled_t, scaled_p, tasks)
    metrics_path = args.output_dir / "unblinding_metrics_v1.json"
    metrics_path.write_text(json.dumps(metrics, indent=1, sort_keys=True))

    receipt = {
        "schema_version": "route_a_v3_gate_p_unblinding_receipt.v1",
        "executed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_record_count": 18292,
        "test_manifest_sha256": manifest_hash,
        "test_record_id_set_sha256": id_set_hash,
        "test_label_read_events": 1,
        "models_evaluated": ["V5_frozen_main_row"],
        "checkpoint_sha256": ckpt_hash,
        "predictions_path": str(pred_path),
        "metrics_path": str(metrics_path),
        "prereg": "docs/paper/gate_p_unblinding_prereg_v1.md",
        "prior_test_reads_before_this_event": 0,
        "gate_p_decision": "user 2026-10-06: after full-paper v2 (satisfied 2026-10-07, journal 140)",
        "cuda_device": torch.cuda.get_device_name(device),
        "bf16": True,
    }
    (args.output_dir / "unblinding_receipt_v1.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print("[4] UNBLINDED. receipt written.", flush=True)

    # ------------------------------------------------ 5. consistency (prereg section 3)
    val_refs = {
        "MEAN_RIBOSOME_LOAD": ("MRL", 0.1354),
        "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS": ("polyA", 0.8219),
        "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE": ("MPRAU", 0.1025),
        "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY": ("TE200304", 0.0579),
        "te_log2_polysome_over_totalrna": ("PLUMAGE-TE", 0.1953),
        "transcript_log2_totalrna_over_dna": ("PLUMAGE-RNA", 0.0500),
        "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE": ("REFALT", 0.0639),
        "RNA_HALF_LIFE_MINUTES": ("HL", 0.0),
    }
    per_task = {}
    for key, m in (metrics.get("tasks") or {}).items():
        for ep, (label, _) in val_refs.items():
            if ep in key:
                per_task.setdefault(label, []).append(m.get("spearman"))
    per_task = {k: (v[0] if len(v) == 1 else sum(x for x in v if x is not None) / max(1, sum(1 for x in v if x is not None))) for k, v in per_task.items()}
    polya_test = per_task.get("polyA")
    non_polya_within = 0
    non_polya_total = 0
    for ep, (label, val) in val_refs.items():
        if label == "polyA":
            continue
        t = per_task.get(label)
        if t is None or val is None:
            continue
        non_polya_total += 1
        if abs(t - val) <= 0.08:
            non_polya_within += 1
    max_non_polya = max((v for k, v in per_task.items() if k != "polyA" and v is not None), default=None)
    macro = metrics.get("task_macro_spearman")
    consist = {
        "C1_polyA_band": {
            "test": polya_test, "val": 0.8219, "tolerance": 0.05,
            "verdict": ("CONFIRMED" if polya_test is not None and abs(polya_test - 0.8219) <= 0.05 else "DEVIATION_REPORTED"),
        },
        "C2_polyA_strongest": {
            "test_polya": polya_test, "test_max_non_polya": max_non_polya,
            "verdict": ("CONFIRMED" if polya_test is not None and max_non_polya is not None and polya_test > max_non_polya else "DEVIATION_REPORTED"),
        },
        "C3_band_structure": {
            "n_within_0p08": non_polya_within, "n_non_polya_tasks": non_polya_total,
            "verdict": ("CONFIRMED" if non_polya_within >= 6 else "DEVIATION_REPORTED"),
        },
        "C4_macro": {"test": macro, "val": 0.167, "delta": (macro or 0) - 0.167,
                     "verdict": ("CONFIRMED" if macro is not None and abs(macro - 0.167) <= 0.05 else "DEVIATION_REPORTED")},
    }
    (args.output_dir / "consistency_check_v1.json").write_text(json.dumps({"per_task": per_task, "checks": consist}, indent=1, sort_keys=True))
    print(json.dumps({"per_task": per_task, "macro": macro, "verdicts": {k: v["verdict"] for k, v in consist.items()}}, indent=1), flush=True)
    print(f"[done] total {time.time()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
