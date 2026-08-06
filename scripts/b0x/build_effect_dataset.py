"""B0-X effect dataset builder.

Loads the rebuilt D1 staging pairs + sequence entities + functional
observations and joins them with the read-only main-repo canonical records
(which carry the actual sequence text, edit scripts and candidate labels), then
derives the source-relative effect delta per asset per config.DELTA_SPECS.

No data is fabricated.  When a source anchor is genuinely not derivable, the
record is emitted with delta_source_status="source_anchor_unavailable" and a
null delta (excluded from delta-based ranking but present in the manifest).

Usage:
    python -m scripts.b0x.build_effect_dataset --out-dir artifacts/b0x
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

CAND_SUFFIX = "__cand"
SRC_SUFFIX = "__src"


def _study_of(seq_id: str) -> str:
    return seq_id.split("_")[0]


def load_pairs(pairs_path: Path) -> List[dict]:
    pairs = []
    active = set(config.ACTIVE_STUDIES)
    for line in pairs_path.open(encoding="utf-8"):
        r = json.loads(line)
        st = _study_of(r["candidate_sequence_id"])
        if st in active and r.get("scientific_track") == "E":
            pairs.append(r)
    return pairs


def load_canonical(records_path: Path, needed_record_ids: set) -> Dict[str, dict]:
    """Stream canonical records; keep only needed record_ids."""
    out: Dict[str, dict] = {}
    for line in records_path.open(encoding="utf-8"):
        r = json.loads(line)
        rid = r.get("record_id")
        if rid in needed_record_ids:
            out[rid] = r
    # Also load reconstructed pairs (rebuilt assets not in the pre-rebuild
    # main-repo canonical records file).
    for study, rp_path in config.RECONSTRUCTED_PAIRS.items():
        if not Path(rp_path).exists():
            continue
        for line in Path(rp_path).open(encoding="utf-8"):
            r = json.loads(line)
            rid = r.get("record_id")
            if rid in needed_record_ids:
                out[rid] = r
    return out


def load_candidate_values(obs_path: Path, cand_seq_ids: Dict[str, set]) -> Dict[str, Dict[str, float]]:
    """Map candidate_sequence_id -> {endpoint_id: value}. Keep only active studies."""
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for line in obs_path.open(encoding="utf-8"):
        r = json.loads(line)
        sid = r.get("sequence_id")
        st = _study_of(sid)
        if st not in cand_seq_ids:
            continue
        if sid not in cand_seq_ids[st]:
            continue
        # average over replicates if multiple obs rows
        out[sid][r["endpoint_id"]] = r["value"]
    return dict(out)


def load_gse114002_source_anchors() -> Dict[str, float]:
    """WT (source) rl anchor per sequence text from the designed library."""
    import pandas as pd
    df = pd.read_csv(config.GSE114002_RAW_LIB, low_memory=False)
    seq_rl: Dict[str, List[float]] = defaultdict(list)
    for seq, rl in zip(df["utr"].astype(str), pd.to_numeric(df["rl"], errors="coerce")):
        if pd.notna(rl) and len(seq) > 10:
            seq_rl[seq].append(float(rl))
    return {k: float(sum(v) / len(v)) for k, v in seq_rl.items()}


def load_gse200304_source_anchors() -> Dict[str, float]:
    """WT Freq anchor per merged_id (the _WT barcode) from the count file."""
    import pandas as pd
    cdf = pd.read_csv(config.GSE200304_COUNT, sep="\t", compression="infer")
    bc_freq: Dict[str, float] = {}
    for bc, fq in zip(cdf["Barcode"].astype(str), pd.to_numeric(cdf["Freq"], errors="coerce")):
        if pd.notna(fq):
            bc_freq[bc] = float(fq)
    # Map WT merged_id -> Freq. merged_id format "chr2:69461620_G-C_WT".
    tdf = pd.read_csv(config.GSE200304_TWIST, sep="\t", compression="infer")
    wt_freq: Dict[str, float] = {}
    for mid in tdf["merged_id"].astype(str):
        if mid.endswith("_WT"):
            wt_freq[mid] = bc_freq.get(mid)
    return wt_freq


def derive_delta(study: str, spec: dict, record: dict,
                 cand_values: Dict[str, Dict[str, float]],
                 src_anchor_gse114002: Dict[str, float],
                 src_anchor_gse200304: Dict[str, float],
                 cand_seq_id: str) -> List[dict]:
    """Return one or more delta records for a pair (GSE217518 yields two)."""
    cand_seq = record.get("candidate_sequence")
    src_seq = record.get("source_sequence")
    edit_list = record.get("edit_script", [])
    metadata = record.get("metadata", {}) or {}

    def base():
        return {
            "pair_id": None,
            "record_id": record.get("record_id"),
            "study": study,
            "benchmark": spec["benchmark"],
            "source_sequence": src_seq,
            "candidate_sequence": cand_seq,
            "edit_list": edit_list,
            "edit_count": len(edit_list),
            "source_id": cand_seq_id[:-len(CAND_SUFFIX)] + SRC_SUFFIX,
            "candidate_id": cand_seq_id,
        }

    t = spec["type"]

    if t == "diff_cand_src":
        cand_val = cand_values.get(cand_seq_id, {}).get(spec["cand_endpoint"])
        src_val = None
        status = "source_anchor_unavailable"
        if spec["src_source"] == "raw_library":
            src_val = src_anchor_gse114002.get(src_seq)
        elif spec["src_source"] == "raw_count":
            # GSE200304: source anchor is the _WT barcode of this pair.
            mid = metadata.get("merged_id")
            if mid:
                wt_id = mid[:-len("_Mutant")] + "_WT" if str(mid).endswith("_Mutant") else (str(mid) + "_WT")
                src_val = src_anchor_gse200304.get(wt_id)
        if cand_val is not None and src_val is not None:
            status = "derived"
        rec = base()
        rec["endpoint"] = spec["cand_endpoint"]
        rec["candidate_value"] = cand_val
        rec["source_value"] = src_val
        rec["delta"] = (cand_val - src_val) if (cand_val is not None and src_val is not None) else None
        rec["delta_source_status"] = status
        return [rec]

    if t == "diff_ref_alt":
        rec = base()
        rec["endpoint"] = f"{spec['ref_endpoint']}/{spec['alt_endpoint']}"
        ref = cand_values.get(cand_seq_id, {}).get(spec["ref_endpoint"])
        alt = cand_values.get(cand_seq_id, {}).get(spec["alt_endpoint"])
        rec["source_value"] = ref
        rec["candidate_value"] = alt
        rec["delta"] = (alt - ref) if (alt is not None and ref is not None) else None
        rec["delta_source_status"] = "derived" if rec["delta"] is not None else "source_anchor_unavailable"
        return [rec]

    if t == "log2fc":
        rec = base()
        rec["endpoint"] = spec["endpoint"]
        val = cand_values.get(cand_seq_id, {}).get(spec["endpoint"])
        rec["source_value"] = None
        rec["candidate_value"] = val
        rec["delta"] = val  # log2FC endpoint is the effect directly
        rec["delta_source_status"] = "log2fc_direct" if val is not None else "source_anchor_unavailable"
        return [rec]

    if t == "diff_wt_meta":
        out = []
        for ep_spec in spec["endpoints"]:
            cand_val = cand_values.get(cand_seq_id, {}).get(ep_spec["cand_endpoint"])
            wt = metadata.get(ep_spec["wt_meta"])
            wt = float(wt) if wt is not None else None
            rec = base()
            rec["endpoint"] = ep_spec["cand_endpoint"]
            rec["source_value"] = wt
            rec["candidate_value"] = cand_val
            rec["delta"] = (cand_val - wt) if (cand_val is not None and wt is not None) else None
            rec["delta_source_status"] = "derived" if rec["delta"] is not None else "source_anchor_unavailable"
            out.append(rec)
        return out

    raise ValueError(f"unknown delta type {t}")


def build(out_dir: Path) -> dict:
    staging = Path(config.STAGING_DIR)
    pairs = load_pairs(staging / "utr_edit_pairs.jsonl")

    # Map pair -> candidate_seq_id / source_seq_id / study
    needed_record_ids = set()
    cand_seq_ids_by_study: Dict[str, set] = defaultdict(set)
    for p in pairs:
        cid = p["candidate_sequence_id"]
        rid = cid[:-len(CAND_SUFFIX)]
        needed_record_ids.add(rid)
        cand_seq_ids_by_study[_study_of(cid)].add(cid)

    canonical = load_canonical(Path(config.CANONICAL_RECORDS), needed_record_ids)

    cand_values = load_candidate_values(staging / "functional_observations.jsonl", cand_seq_ids_by_study)

    src_114 = load_gse114002_source_anchors() if "GSE114002" in config.ACTIVE_STUDIES else {}
    src_200 = load_gse200304_source_anchors() if "GSE200304" in config.ACTIVE_STUDIES else {}

    records = []
    per_asset = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        cid = p["candidate_sequence_id"]
        rid = cid[:-len(CAND_SUFFIX)]
        study = _study_of(cid)
        rec0 = canonical.get(rid)
        if rec0 is None:
            per_asset[study]["missing_canonical"] += 1
            continue
        spec = config.DELTA_SPECS[study]
        deltas = derive_delta(study, spec, rec0, cand_values,
                              src_114, src_200, cid)
        for d in deltas:
            d["pair_id"] = p["pair_id"]
            records.append(d)
            per_asset[study]["total"] += 1
            per_asset[study][d["delta_source_status"]] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "effect_dataset.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "n_records": len(records),
        "n_pairs_scanned": len(pairs),
        "per_asset": {k: dict(v) for k, v in per_asset.items()},
        "sha256": sha256_file(out_path),
        "config": {
            "staging": str(staging),
            "canonical": config.CANONICAL_RECORDS,
            "split": config.PRIMARY_SPLIT,
            "seed": config.SEED,
        },
    }
    man_path = out_dir / "effect_dataset_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/b0x"))
    args = ap.parse_args()
    man = build(args.out_dir)
    print(json.dumps({k: (v if k != "per_asset" else v) for k, v in man.items() if k != "sha256"}, indent=2))
    print("sha256:", man["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())