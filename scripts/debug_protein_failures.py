"""Debug: find protein-identity failures in build_unlabeled_cds_tier."""
import sys
from pathlib import Path

repo = Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow")
sys.path.insert(0, str(repo.parent))

from mrna_editflow.data.p3_benchmark_builder import (
    BuildConfig, build_unlabeled_cds_tier, filter_eligible_sources,
    load_reconstructed_sources, RECORDS_JSONL, FAMILY_JSON,
)
from mrna_editflow.data.p3_legality import protein_identical, translate
import random

sources = load_reconstructed_sources(RECORDS_JSONL, FAMILY_JSON)
eligible, _ = filter_eligible_sources(sources)
cfg = BuildConfig(smoke=True).effective()
rng = random.Random(f"{cfg.seed}:cds_source_pick")
pool = [s for s in eligible if len(s["cds"]) // 3 >= 52]
rng.shuffle(pool)
cds_sources = pool[: cfg.cds_sources]

records, stats = build_unlabeled_cds_tier(cds_sources, cfg)
print("stats:", stats)
src_by_id = {s["transcript_id"]: s for s in cds_sources}
fails = [r for r in records if not r.protein_identity]
print(f"n_failures={len(fails)}")
for r in fails[:10]:
    s = src_by_id[r.source_id]
    print("=" * 60)
    print("region:", r.edited_region, "edit_type:", r.edit_type,
          "edit_count:", r.edit_count)
    print("edits:", r.edit_list)
    print("scope_len:", len(r.source_sequence), "cds_len:", len(s["cds"]))
    # re-derive what the protein change is
    if r.edited_region == "joint_5utr_cds":
        continue
    # scope application
    scope = r.source_sequence
    cand = r.candidate_sequence
    off = {"cds_first30": 3, "cds_first50": 3, "cds_remaining": 150}[r.edited_region]
    full_new = s["cds"][:off] + cand + s["cds"][off + len(cand):]
    pa, pb = translate(s["cds"]), translate(full_new)
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(pa, pb)) if a != b]
    print("protein diffs (idx, old, new):", diffs)
