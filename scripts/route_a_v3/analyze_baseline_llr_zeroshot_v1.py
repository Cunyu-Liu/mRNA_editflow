#!/usr/bin/env python3
"""Baseline P0-1: zero-shot log-LLR variant-effect baseline family.

SPECS_BASELINE_LEADERBOARD spec 2026-09-08 增补 §V.4 P0-1 (H2 third-mode evidence +
MPRAU-constraint four-mode closure). Models (downloaded via hf-mirror):

- caduceus: kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
  single-nucleotide vocab (A=7 C=8 G=9 T=10, [SEP] appendix, mask id=3).
- ntv2: InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
  non-overlapping 6-mer tokens + trailing single-nucleotide tokens
  (locate the token covering the edit position; mask the whole token;
  alt token = the windowed k-mer with the edited base substituted).

Scoring protocol (per record): DNA-converted source sequence; for each edit
(position, ref, alt) of source_relative_edits, masked-LM log-likelihood ratio
    LLR = log P(alt | context) - log P(ref | context)
record-level prediction = sum over edits. U->T conversion; no truncation
needed (UTR lengths << model context). Position semantics verified against
source_sequence (0-based first, 1-based fallback; mismatch -> flagged skip).

Tasks (VALIDATION split, frozen manifest; protected reads = 0):
- mprau (ENCSR854RUF): unique-variant scoring (LLR has no cell conditioning,
  same calibre note as the Saluki weak control) -> variant pair-mean rho
- gse186455 / gse149487 (te + rna) / gse217518 (5UTR + 3UTR): record-level rho
Dual calibres: signed rho (primary) and |y| rho (secondary, Tang & Koo style).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RUNNER = REPO_ROOT / "scripts/route_a_v3/run_route2_v8_stage2_adapt_v1.py"
_spec = importlib.util.spec_from_file_location("r2", _RUNNER)
r2 = importlib.util.module_from_spec(_spec)
sys.modules["r2"] = r2
_spec.loader.exec_module(r2)

MNT = r2.MNT
ASSETS = MNT / "external_model_assets"
CADUCEUS_DIR = ASSETS / "caduceus/caduceus-ph_131k_d256_n16"
NTV2_DIR = ASSETS / "nucleotide_transformer/nt-v2-500m-multi-species"
HYENADNA_DIR = ASSETS / "hyenadna/hyenadna-small-32k-seqlen-hf"
OUT_ROOT = MNT / "experiments/analysis_baseline_llr_zeroshot_20260908"

CANONICAL_ROOT = MNT / "canonical"
STUDY_REL = {
    "ENCSR854RUF": "ENCSR854RUF/v1/canonical_records.private.jsonl",
    "GSE186455": "GSE186455/v1/canonical_records.private.jsonl",
    "GSE149487": "GSE149487/v1/canonical_records.private.jsonl",
    "GSE217518": "GSE217518/v1/canonical_records.jsonl",
}
# task -> (study, region, endpoint) strata; mprau handled separately
TASKS = {
    "gse186455": ("GSE186455", None, None),
    "gse149487_te": ("GSE149487", "5UTR", "te_log2_polysome_over_totalrna"),
    "gse149487_rna": ("GSE149487", "5UTR", "transcript_log2_totalrna_over_dna"),
    "gse217518_5utr": ("GSE217518", "5UTR", "RNA_HALF_LIFE_MINUTES"),
    "gse217518_3utr": ("GSE217518", "3UTR", "RNA_HALF_LIFE_MINUTES"),
}
BATCH = 32
SEED = 20260816


def manifest_validation_ids() -> dict[str, set[str]]:
    by_study: dict[str, set[str]] = {}
    with (MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "VALIDATION":
                by_study.setdefault(str(row["study_unit_id"]), set()).add(str(row["canonical_record_id"]))
    return by_study


def load_canonical(rel: str) -> dict[str, dict]:
    records = {}
    with (CANONICAL_ROOT / rel).open() as handle:
        for line in handle:
            row = json.loads(line)
            records[str(row["canonical_record_id"])] = row
    return records


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    v = spearmanr(x, y).statistic
    return float(v) if np.isfinite(v) else float("nan")


# ------------------------------------------------------------------ models --
class CaduceusScorer:
    key = "caduceus"
    def __init__(self, device):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.tok = AutoTokenizer.from_pretrained(CADUCEUS_DIR, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(CADUCEUS_DIR, trust_remote_code=True).to(device).eval()
        self.device = device
        self.mask_id = int(self.tok.mask_token_id)
        self.nt_ids = {c: self.tok.convert_tokens_to_ids(c) for c in "ACGT"}

    def build_jobs(self, dna: str, edits: list[tuple[int, str, str]]):
        """Each job = (input_ids, mask_pos_in_ids, ref_id, alt_id)."""
        enc = self.tok(dna, add_special_tokens=True)  # single-nt ids + [SEP]
        ids = list(enc["input_ids"])
        # verify token alignment: token i (i < len(ids)-1) should be dna[i]
        jobs = []
        for pos, ref, alt in edits:
            if pos < 0 or pos >= len(dna) or ids[pos] != self.nt_ids.get(dna[pos]):
                return None  # alignment surprise -> caller flags the record
            m = list(ids)
            m[pos] = self.mask_id
            jobs.append((m, pos, self.nt_ids[ref], self.nt_ids[alt]))
        return jobs


class NTV2Scorer:
    key = "ntv2"
    def __init__(self, device):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.tok = AutoTokenizer.from_pretrained(NTV2_DIR, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(NTV2_DIR, trust_remote_code=True).to(device).eval()
        self.device = device
        self.mask_id = int(self.tok.mask_token_id)

    def build_jobs(self, dna: str, edits: list[tuple[int, str, str]]):
        enc = self.tok(dna, add_special_tokens=True)
        ids = list(enc["input_ids"])
        tokens = self.tok.convert_ids_to_tokens(ids)
        # char-coverage map: walk tokens, accumulate covered characters
        cover: list[tuple[int, int, int]] = []  # (token_idx, char_start, char_len)
        pos_cursor = 0
        for ti, t in enumerate(tokens):
            if t in ("<cls>", "<pad>", "<mask>", "<sep>", "<unk>"):
                continue
            L = len(t)
            cover.append((ti, pos_cursor, L))
            pos_cursor += L
        if pos_cursor != len(dna):
            return None  # tokenisation does not tile the sequence -> flag
        jobs = []
        for pos, ref, alt in edits:
            hit = None
            for ti, cs, cl in cover:
                if cs <= pos < cs + cl:
                    hit = (ti, cs, cl); break
            if hit is None:
                return None
            ti, cs, cl = hit
            if dna[pos] != ref:
                return None  # caller verifies separately; belt-and-braces
            token_str = tokens[ti]
            offset = pos - cs
            alt_token = token_str[:offset] + alt + token_str[offset + 1:]
            alt_id = self.tok.convert_tokens_to_ids(alt_token)
            ref_id = ids[ti]
            if alt_id is None or alt_id == self.tok.unk_token_id:
                return None
            m = list(ids)
            m[ti] = self.mask_id
            jobs.append((m, ti, ref_id, alt_id))
        return jobs


@torch.no_grad()
def score_jobs(scorer, jobs: list) -> list[float]:
    """Batch-forward masked inputs; return log P(alt)-log P(ref) per job."""
    out = []
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        maxlen = max(len(m) for m, *_ in chunk)
        pad_id = scorer.tok.pad_token_id if scorer.tok.pad_token_id is not None else 0
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, (m, _, _, _) in enumerate(chunk):
            input_ids[i, :len(m)] = torch.tensor(m, dtype=torch.long)
            attn[i, :len(m)] = 1
        logits = scorer.model(input_ids=input_ids.to(scorer.device),
                              attention_mask=attn.to(scorer.device)).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1).cpu()
        for i, (_, mpos, ref_id, alt_id) in enumerate(chunk):
            out.append(float(logprobs[i, mpos, alt_id] - logprobs[i, mpos, ref_id]))
    return out


class HyenaDNAScorer:
    """Causal-LM pseudo-LL scorer (LongSafari/hyenadna-*-hf, character vocab).

    One forward per record on the SOURCE sequence: for each edit at position p,
    LLR = log P(alt | prefix x_<p) - log P(ref | prefix x_<p), both read off the
    same logits row p-1 (the model's prediction of token p from the prefix).
    """
    key = "hyenadna"
    def __init__(self, device):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(HYENADNA_DIR, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(HYENADNA_DIR, trust_remote_code=True).to(device).eval()
        self.device = device

    def score_records(self, records: dict[str, dict]):
        jobs, flagged = [], 0
        for rid, rec in sorted(records.items()):
            dna, edits = normalise_edits(rec)
            if edits is None:
                flagged += 1
                continue
            if any(pos == 0 for pos, _, _ in edits):  # no prefix for position 0
                flagged += 1
                continue
            jobs.append((rid, dna, edits))
        preds: dict[str, float] = {}
        with torch.no_grad():
            pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            for start in range(0, len(jobs), BATCH):
                chunk = jobs[start:start + BATCH]
                encoded = [self.tok(dna, add_special_tokens=True)["input_ids"] for _, dna, _ in chunk]
                maxlen = max(len(e) for e in encoded)
                input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
                for i, e in enumerate(encoded):
                    input_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
                    attn[i, :len(e)] = 1
                # right-padding + causal LM: pads cannot influence earlier positions,
                # and the custom forward() does not accept attention_mask
                logits = self.model(input_ids=input_ids.to(self.device)).logits
                lsm = torch.log_softmax(logits.float(), dim=-1).cpu()
                for i, (rid, dna, edits) in enumerate(chunk):
                    ids = encoded[i]
                    tokens = self.tok.convert_ids_to_tokens(ids)
                    # char-offset map (skip specials); logits row ti predicts token ti+1
                    offset_of: dict[int, int] = {}
                    cursor = 0
                    for ti, t in enumerate(tokens):
                        if t in ("<cls>", "<bos>", "<eos>", "<sep>", "<pad>", "<unk>", "<mask>"):
                            continue
                        for k in range(len(t)):
                            offset_of[cursor + k] = ti + 1
                        cursor += len(t)
                    total = 0.0
                    ok = True
                    for pos, ref, alt in edits:
                        ti = offset_of.get(pos)
                        if ti is None:
                            ok = False
                            break
                        ref_id = self.tok.convert_tokens_to_ids(ref)
                        alt_id = self.tok.convert_tokens_to_ids(alt)
                        total += float(lsm[i, ti - 1, alt_id] - lsm[i, ti - 1, ref_id])
                    if ok:
                        preds[rid] = preds.get(rid, 0.0) + total
                    else:
                        flagged += 1
        return preds, len(preds), flagged


def get_scorer(model_key: str, device):
    if model_key == "caduceus":
        return CaduceusScorer(device)
    if model_key == "ntv2":
        return NTV2Scorer(device)
    if model_key == "hyenadna":
        return HyenaDNAScorer(device)
    raise ValueError(model_key)


def normalise_edits(record: dict) -> tuple[str, list[tuple[int, str, str]] | None]:
    """Return (dna, edits) with 0-based positions verified, or (dna, None) if unresolvable.

    Canonical schema: `edit_operations` entries {type: SUB, position_zero_based, ref, alt}.
    Legacy `source_relative_edits` ({position, ref, alt}) kept as fallback.
    """
    src = str(record["source_sequence"]).upper().replace("U", "T")
    raw = record.get("edit_operations") or record.get("source_relative_edits") or []
    edits = []
    ok = True
    for e in raw:
        pos = int(e.get("position_zero_based", e.get("position")))
        ref = str(e["ref"]).upper().replace("U", "T")
        alt = str(e["alt"]).upper().replace("U", "T")
        if 0 <= pos < len(src) and src[pos] == ref:
            edits.append((pos, ref, alt))
        elif 1 <= pos <= len(src) and src[pos - 1] == ref:  # 1-based fallback
            edits.append((pos - 1, ref, alt))
        else:
            ok = False
    if not ok or not edits:
        return src, None
    return src, edits


def score_records(scorer, records: dict[str, dict]) -> tuple[dict[str, float], int, int]:
    """Score every record; returns (rid -> LLR sum, n_scored, n_flagged).

    Scorers with their own score_records method (causal-PLL) bypass the
    per-edit masked-LM job path.
    """
    if hasattr(scorer, "score_records") and callable(getattr(scorer, "score_records")) \
            and not isinstance(scorer, (CaduceusScorer, NTV2Scorer)):
        return scorer.score_records(records)
    all_jobs, meta = [], []
    flagged = 0
    for rid, rec in sorted(records.items()):
        dna, edits = normalise_edits(rec)
        if edits is None:
            flagged += 1
            continue
        jobs = scorer.build_jobs(dna, edits)
        if jobs is None:
            flagged += 1
            continue
        for j in jobs:
            all_jobs.append(j)
            meta.append(rid)
    values = score_jobs(scorer, all_jobs)
    per_record: dict[str, float] = {}
    for rid, v in zip(meta, values):
        per_record[rid] = per_record.get(rid, 0.0) + v
    return per_record, len(per_record), flagged


def mprau_pair_mean(records: dict[str, dict], preds: dict[str, float]) -> dict:
    by_variant: dict[str, list[str]] = {}
    for rid in preds:
        by_variant.setdefault(rid.split(":context:")[0], []).append(rid)
    t_list, p_list = [], []
    for variant, rids in by_variant.items():
        if len(rids) >= 2:
            t_list.append(float(np.mean([float(records[r]["direction_normalized_delta"]) for r in rids])))
            p_list.append(float(np.mean([preds[r] for r in rids])))
    t = np.asarray(t_list); p = np.asarray(p_list)
    return {
        "n_variants": len(t_list),
        "pair_mean_spearman_signed": spearman(p, t),
        "pair_mean_spearman_abs_target": spearman(np.abs(p), np.abs(t)),
        "calibre_note": "LLR has no cell conditioning (per-variant constant across 6 cell rows); "
                        "same no-context-scorer note as the Saluki weak control",
    }


def record_task_metrics(records: dict[str, dict], preds: dict[str, float]) -> dict:
    ids = sorted(preds)
    t = np.asarray([float(records[r]["direction_normalized_delta"]) for r in ids])
    p = np.asarray([preds[r] for r in ids])
    return {
        "n_records": len(ids),
        "spearman_signed": spearman(p, t),
        "spearman_abs_target": spearman(np.abs(p), np.abs(t)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("caduceus", "ntv2", "hyenadna"))
    parser.add_argument("--physical-gpu-index", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="cap to 40 records/task")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(SEED)

    scorer = get_scorer(args.model, device)
    out_dir = Path(str(OUT_ROOT / args.model) + ("_smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    val_ids = manifest_validation_ids()
    results = {"schema_version": "route_a_v3_baseline_llr_zeroshot.v1",
               "model": args.model, "split": "VALIDATION", "seed": SEED,
               "protocol": "masked-LM per-edit log-likelihood ratio, U->T, no truncation; "
                           "signed (primary) + |y| (secondary) calibres",
               "tasks": {}}

    # MPRAU: score unique variants only (LLR constant per variant across cells)
    mprau_records = {rid: rec for rid, rec in load_canonical(STUDY_REL["ENCSR854RUF"]).items()
                     if rid in val_ids.get("ENCSR854RUF", set())}
    # deduplicate to one representative row per variant (same source/edits across 6 cells)
    variant_rep: dict[str, dict] = {}
    for rid in sorted(mprau_records):
        variant_rep.setdefault(rid.split(":context:")[0], mprau_records[rid])
    if args.smoke:
        variant_rep = dict(list(sorted(variant_rep.items()))[:40])
    var_preds, n_scored, n_flagged = score_records(scorer, variant_rep)
    # broadcast variant-level LLR back to all cell rows
    mprau_preds = {rid: var_preds[rid.split(":context:")[0]]
                   for rid in mprau_records if rid.split(":context:")[0] in var_preds}
    results["tasks"]["mprau"] = mprau_pair_mean(mprau_records, mprau_preds) | {
        "n_unique_variants_scored": n_scored, "n_records_broadcast": len(mprau_preds),
        "n_variants_flagged": n_flagged}
    print(f"[{args.model}] mprau: {json.dumps(results['tasks']['mprau'])}", flush=True)
    with (out_dir / "mprau_predictions.jsonl").open("w") as fh:
        for rid, v in sorted(mprau_preds.items()):
            fh.write(json.dumps({"canonical_record_id": rid, "llr": v}) + "\n")

    # record-level tasks
    for task_key, (study, region, endpoint) in TASKS.items():
        records = {rid: rec for rid, rec in load_canonical(STUDY_REL[study]).items()
                   if rid in val_ids.get(study, set())}
        if region is not None:
            records = {rid: rec for rid, rec in records.items()
                       if str(rec.get("region")) == region and str(rec.get("endpoint_id")) == endpoint}
        if args.smoke:
            records = dict(list(sorted(records.items()))[:40])
        if not records:
            results["tasks"][task_key] = {"n_records": 0, "skipped": "no VALIDATION records"}
            continue
        preds, n_scored, n_flagged = score_records(scorer, records)
        results["tasks"][task_key] = record_task_metrics(records, preds) | {
            "n_records_flagged": n_flagged}
        print(f"[{args.model}] {task_key}: {json.dumps(results['tasks'][task_key])}", flush=True)
        with (out_dir / f"{task_key}_predictions.jsonl").open("w") as fh:
            for rid, v in sorted(preds.items()):
                fh.write(json.dumps({"canonical_record_id": rid, "llr": v}) + "\n")

    results["cpu_fallback_used"] = False
    (out_dir / "llr_results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out_dir / 'llr_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
