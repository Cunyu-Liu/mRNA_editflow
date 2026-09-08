#!/usr/bin/env python3
"""V9 Stage 0 zero-training diagnostics (SPECS_CRITIC_V6 spec N.4, approved 2026-09-08).

Parts (select with --part):
- m2  : parameter-interference localisation (CPI-FT arXiv 2508.21741 method):
        task vectors tau(h_bench9) vs tau(h_mprau_lora, LoRA-merged) on the
        H family (same Stage 1 H init); per-task top-k% magnitude core-region
        masks (k in {5,10,20}%), Jaccard overlap per layer + overall;
        per-layer cosine between the two task vectors.
- m1  : merge feasibility (preregistered grid): TIES (keep in {0.2,0.8,0.9,0.95}
        -- 0.2 = paper-typical density, {0.8,0.9,0.95} = spec grid) / DARE
        (p in {0.9,0.95}) / arithmetic-mean on the H family pair
        {h_bench9, h_mprau_lora}; S family inter-seed soup {s_mprau_in x5}
        (mechanism check vs prediction-ensemble 0.1351).
        KNOWLEDGE-RETENTION GATE (preregistered, spec N.4.3):
        merged per-task >= max(single models) - 0.01.
        Singles re-evaluated through the identical pipeline as cross-check
        (h_bench9 must reproduce MRL 0.2879 / polyA 0.8067 / MPRAU 0.0556).
- m2b : task-pair gradient cosine on the V8-S Stage 1 backbone (extended to
        9 domains / 6 cells exactly as the Stage 2 init = the V9-1a init),
        >=100 batches/task, batch 32 pairs, eval mode (no dropout), BF16.
        Mechanistic adjudication of the optimisation-layer switch (spec N.3.5).

Discipline: CUDA BF16 only (m1/m2b inference+grad, cpu_fallback false);
protected reads = 0 (VALIDATION split only, no TEST/Evaluation outcome);
FINAL-EPOCH-FIXED checkpoints only; zero training; products /mnt.
Small-key merge policy (documented, <0.01% of params): cell_embeddings and
domain rows 3-8 have no Stage 1 base value -> element-wise mean of the two
models; domain rows 0-2 enter the task-vector merge via sliced base.
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
EXP = MNT / "experiments/xeditcritic_route_a"
STAGE1_H = EXP / "v8_stage1_joint_prefinetune_20260904/h_mrl-polya/stage1_h_epoch2.pt"
STAGE1_S = EXP / "v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt"
S2 = EXP / "v8_stage2_adapt_20260907"
H_BENCH9 = S2 / "h_bench9/stage2_h_benchmark_full_epoch6.pt"
H_MPRAU_LORA = S2 / "h_mprau_lora/stage2_h_benchmark_lora_epoch6.pt"
S_SEEDS = [
    S2 / "s_mprau_in/stage2_s_benchmark_full_epoch6.pt",
    S2 / "s_mprau_in_s2/stage2_s_benchmark_full_epoch6.pt",
    S2 / "s_mprau_in_s3/stage2_s_benchmark_full_epoch6.pt",
    S2 / "s_mprau_in_s4/stage2_s_benchmark_full_epoch6.pt",
    S2 / "s_mprau_in_s5/stage2_s_benchmark_full_epoch6.pt",
]
OUT_ROOT = MNT / "experiments/analysis_v9_stage0_20260908"

V5_MPRAU_REFERENCE = 0.1025
PRED_ENSEMBLE_MPRAU = 0.1351  # z-mean prediction ensemble of the same 5 seeds
# preregistered single-model references (FINAL-EPOCH-FIXED, epoch_eval_metrics.jsonl)
SINGLE_REFS = {
    "h_bench9": {"mrl": 0.2879, "polya": 0.8067, "mprau": 0.0556, "macro": 0.1516},
    "h_mprau_lora": {"mprau": 0.1084},
    "s_mprau_in_seed1": {"mprau": 0.1527},
}
RETENTION_EPS = 0.01  # preregistered tolerance


def load_sd(path: Path) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return raw["model_state_dict"]


def unwrap_lora(sd: dict, scaling: float = 2.0) -> dict:
    """LoRALinearV3-wrapped state dict -> plain full state dict (W_eff = W + scaling*B@A)."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".lora_a") or k.endswith(".lora_b"):
            continue
        out[k.replace(".base.", ".")] = v
    for k in list(out.keys()):
        if not k.endswith(".weight"):
            continue
        parent = k[: -len(".weight")]
        a_key, b_key = parent + ".lora_a", parent + ".lora_b"
        if a_key in sd and b_key in sd:
            delta = (sd[b_key].float() @ sd[a_key].float()) * scaling
            out[k] = (out[k].float() + delta).to(out[k].dtype)
    return out


def layer_group(key: str) -> str:
    if key.startswith("base.embeddings"):
        return "base.embeddings"
    if key.startswith("base.encoder.layer."):
        idx = key.split(".")[3]
        return f"base.encoder.layer.{idx}"
    if key.startswith("base."):
        return "base.other"
    if key.startswith("stem."):
        return "stem"
    if key.startswith("head."):
        return "head"
    if key.startswith("domain_embeddings"):
        return "domain_embeddings"
    if key.startswith("cell_embeddings"):
        return "cell_embeddings"
    return "other"


def shared_entries(base_sd: dict, sd1: dict, sd2: dict) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Ordered (name, base, t1, t2) entries on which task vectors are defined.

    domain_embeddings handled by slicing: rows 0-2 (Stage 1 ids) join the task
    vectors; rows 3-8 are Stage-2-only (merged as element-wise mean elsewhere).
    """
    entries = []
    for k in sd1:
        if k not in sd2 or k not in base_sd:
            continue
        if k == "domain_embeddings.weight":
            entries.append((k, base_sd[k][:3].clone(), sd1[k][:3].clone(), sd2[k][:3].clone()))
            continue
        if sd1[k].shape != base_sd[k].shape or sd2[k].shape != base_sd[k].shape:
            continue
        entries.append((k, base_sd[k].clone(), sd1[k].clone(), sd2[k].clone()))
    return entries


def flatten_taus(entries) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, int, int]]]:
    """Task vectors tau_i = sd_i - base, flattened over the shared entry order."""
    t1_parts, t2_parts, index = [], [], []
    pos = 0
    for name, b, a, c in entries:
        n = b.numel()
        index.append((name, pos, pos + n))
        t1_parts.append((a - b).flatten().float())
        t2_parts.append((c - b).flatten().float())
        pos += n
    return torch.cat(t1_parts), torch.cat(t2_parts), index


# ---------------------------------------------------------------- part m2 ----
def part_m2() -> dict:
    base_sd = load_sd(STAGE1_H)
    sd1 = load_sd(H_BENCH9)
    sd2 = unwrap_lora(load_sd(H_MPRAU_LORA))
    entries = shared_entries(base_sd, sd1, sd2)
    tau1, tau2, index = flatten_taus(entries)
    t1_64, t2_64 = tau1.double(), tau2.double()
    n = tau1.numel()
    cos = float((t1_64 * t2_64).sum() / (t1_64.norm() * t2_64.norm()))

    report = {
        "schema_version": "route_a_v3_route2_v9_stage0_m2.v1",
        "pair": ["h_bench9 (9-task balanced, full-FT)", "h_mprau_lora (MPRAU-only, LoRA, merged)"],
        "base": str(STAGE1_H),
        "n_task_vector_elements": int(n),
        "tau1_norm": float(t1_64.norm()),
        "tau2_norm": float(t2_64.norm()),
        "overall_cosine": cos,
        "overall_cosine_interpretation": "cos(tau_bench9, tau_mprau_lora) on the shared H-family parameter space (float64; task vectors = sd - Stage1 base)",
        "per_layer": {},
        "topk_overlap": {},
    }
    # per-layer cosine (full task vectors, per layer; float64; zero-norm marked)
    for name, i0, i1 in index:
        g = layer_group(name)
        a, b = t1_64[i0:i1], t2_64[i0:i1]
        na, nb = float(a.norm()), float(b.norm())
        if na == 0.0 or nb == 0.0:
            c = None  # tau2 is exactly zero on non-LoRA keys (frozen base)
        else:
            c = float((a * b).sum() / (na * nb))
        rep = report["per_layer"].setdefault(g, {"cosine_sum": 0.0, "n_tensors": 0, "n_zero": 0})
        if c is None:
            rep["n_zero"] += 1
        else:
            rep["cosine_sum"] += c
            rep["n_tensors"] += 1
    for g, rep in report["per_layer"].items():
        rep["cosine_mean"] = rep.pop("cosine_sum") / rep["n_tensors"] if rep["n_tensors"] else None

    # per-task top-k% global masks -> Jaccard overall + per layer
    for k_pct in (5, 10, 20):
        keep = max(1, int(n * k_pct / 100))
        def top_mask(t: torch.Tensor) -> torch.Tensor:
            thresh = torch.sort(t.abs(), descending=True).values[keep - 1]
            return t.abs() >= thresh
        m1, m2 = top_mask(tau1), top_mask(tau2)
        inter = int((m1 & m2).sum())
        union = int((m1 | m2).sum())
        entry = {
            "k_pct": k_pct,
            "jaccard_overall": inter / union if union else None,
            "mask1_size": int(m1.sum()), "mask2_size": int(m2.sum()),
            "intersection": inter, "union": union,
            "per_layer_jaccard": {},
        }
        for name, i0, i1 in index:
            g = layer_group(name)
            a, b = m1[i0:i1], m2[i0:i1]
            it, un = int((a & b).sum()), int((a | b).sum())
            rep = entry["per_layer_jaccard"].setdefault(g, {"inter": 0, "union": 0})
            rep["inter"] += it
            rep["union"] += un
        for g, rep in entry["per_layer_jaccard"].items():
            rep["jaccard"] = rep["inter"] / rep["union"] if rep["union"] else None
        report["topk_overlap"][f"top{k_pct}pct"] = entry
    return report


# ---------------------------------------------------------------- part m1 ----
def ties_merge(taus: list[torch.Tensor], keep_frac: float) -> torch.Tensor:
    stacked = torch.stack(taus)  # [T, N]
    T, N = stacked.shape
    keep = max(1, int(N * keep_frac))
    trimmed = torch.zeros_like(stacked)
    for i in range(T):
        vals = stacked[i].abs()
        thresh = torch.sort(vals, descending=True).values[keep - 1]
        trimmed[i] = torch.where(vals >= thresh, stacked[i], torch.zeros_like(stacked[i]))
    elected = torch.sign(trimmed.sum(dim=0))
    agree = (torch.sign(trimmed) == elected.unsqueeze(0)) & (trimmed != 0)
    cnt = agree.sum(dim=0).clamp(min=1)
    merged = (trimmed * agree).sum(dim=0) / cnt
    return torch.where(elected != 0, merged, torch.zeros_like(merged))


def dare_merge(taus: list[torch.Tensor], drop_p: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    stacked = torch.stack(taus)
    N = stacked.shape[1]
    keep_mask = (torch.rand(N, generator=g) >= drop_p).unsqueeze(0)
    rescaled = stacked * keep_mask.to(stacked.dtype) / (1.0 - drop_p)
    return rescaled.mean(dim=0)


def assemble_merged(base_sd, sd1, sd2, merged_tau, entries, index) -> dict:
    out: dict[str, torch.Tensor] = {}
    for (name, b, _a, _c), (_nm, i0, i1) in zip(entries, index):
        delta = merged_tau[i0:i1].reshape(b.shape)
        if name == "domain_embeddings.weight":
            # base rows 0-2 merged; rows 3-8 = element-wise mean of the two models
            rest = (sd1[name][3:] + sd2[name][3:]) / 2
            out[name] = torch.cat([(b + delta), rest.to(b.dtype)])
        else:
            out[name] = b + delta.to(b.dtype)
    # Stage-2-only keys (cell_embeddings etc.) + anything not covered
    for k in set(sd1) | set(sd2):
        if k in out:
            continue
        if k in sd1 and k in sd2 and sd1[k].shape == sd2[k].shape:
            out[k] = ((sd1[k] + sd2[k]) / 2).to(sd1[k].dtype if k in sd1 else sd2[k].dtype)
        elif k in sd1:
            out[k] = sd1[k]
        else:
            out[k] = sd2[k]
    return out


def eval_full(model, tokenizer, device, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    r2._OUT_DIR = str(out_dir)
    results = []
    for study, (domain, _rel) in r2.BENCHMARK_DOMAINS.items():
        if study == "ENCSR854RUF":
            continue
        results.append(r2.eval_study_frozen_delta(model, tokenizer, device, study, domain))
    mprau = r2.eval_mprau_validation(model, tokenizer, device, r2.DOMAIN_IDS["mprau"])
    results.append({"domain": "mprau", **mprau})
    return {"task_macro_spearman": r2._task_macro(results), "per_task": results, "mprau": mprau}


def per_task_table(eval_report: dict) -> dict:
    table = {}
    for rec in eval_report["per_task"]:
        d = rec.get("domain")
        table[d] = {"spearman": rec.get("task_macro_spearman"),
                    "pair_mean": rec.get("pair_mean_spearman")}
    table["macro"] = eval_report["task_macro_spearman"]
    return table


def part_m1(gpu_index: int) -> dict:
    device = torch.device(f"cuda:{gpu_index}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(r2.MRNABERT_PATH, local_files_only=True)

    base_sd = load_sd(STAGE1_H)
    sd1 = load_sd(H_BENCH9)
    sd2 = unwrap_lora(load_sd(H_MPRAU_LORA))
    entries = shared_entries(base_sd, sd1, sd2)
    tau1, tau2, index = flatten_taus(entries)
    taus = [tau1, tau2]

    model = r2.build_v8_regressor(r2.MRNABERT_PATH, "h",
                                  num_domains=len(r2.DOMAIN_IDS), num_cells=r2.NUM_CELLS).to(device)

    # variant factories (lazy: build state dict -> eval -> free)
    def arith_tau():
        return torch.stack(taus).mean(dim=0)

    factories: list[tuple[str, object]] = [
        ("single_h_bench9", lambda: sd1),
        ("single_h_mprau_lora", lambda: sd2),
        ("arith_mean", lambda: assemble_merged(base_sd, sd1, sd2, arith_tau(), entries, index)),
    ]
    for keep in (0.2, 0.8, 0.9, 0.95):
        factories.append((f"ties_keep{int(keep * 100)}pct",
                          (lambda k: lambda: assemble_merged(base_sd, sd1, sd2, ties_merge(taus, k), entries, index))(keep)))
    for p in (0.9, 0.95):
        factories.append((f"dare_drop{int(p * 100)}pct",
                          (lambda pp: lambda: assemble_merged(base_sd, sd1, sd2, dare_merge(taus, pp, 20260908), entries, index))(p)))

    tables: dict[str, dict] = {}
    m1_out = OUT_ROOT / "m1_merge"
    for name, factory in factories:
        sd = factory()
        model.load_state_dict(sd, strict=True)
        rep = eval_full(model, tokenizer, device, m1_out / name)
        tables[name] = per_task_table(rep)
        (m1_out / name / "eval.json").write_text(json.dumps(rep, indent=1, default=str))
        print(f"[m1] {name}: {json.dumps(tables[name])}", flush=True)
        del sd

    # S family inter-seed soup (mechanism check): weight-space mean of 5 seeds
    s_sds = [load_sd(p) for p in S_SEEDS]
    soup = {}
    for k in s_sds[0]:
        soup[k] = torch.stack([sd[k].float() for sd in s_sds]).mean(dim=0).to(s_sds[0][k].dtype)
    s_model = r2.build_v8_regressor(r2.MRNABERT_PATH, "s",
                                    num_domains=len(r2.DOMAIN_IDS), num_cells=r2.NUM_CELLS).to(device)
    s_model.load_state_dict(load_sd(S_SEEDS[0]), strict=True)
    rep = eval_full(s_model, tokenizer, device, m1_out / "single_s_mprau_in_seed1")
    tables["single_s_mprau_in_seed1"] = per_task_table(rep)
    print(f"[m1] single_s_mprau_in_seed1: {json.dumps(tables['single_s_mprau_in_seed1'])}", flush=True)
    s_model.load_state_dict(soup, strict=True)
    rep = eval_full(s_model, tokenizer, device, m1_out / "s_soup_5seed")
    tables["s_soup_5seed"] = per_task_table(rep)
    print(f"[m1] s_soup_5seed: {json.dumps(tables['s_soup_5seed'])}", flush=True)

    # knowledge-retention gate on the H family pair
    def val(t, key):
        v = t.get(key, {}) if isinstance(t.get(key), dict) else {}
        return v.get("spearman"), v.get("pair_mean")

    gates = {}
    for name, t in tables.items():
        if not name.startswith(("ties", "dare", "arith")):
            continue
        mrl = t.get("mrl", {}).get("spearman")
        pol = t.get("polya", {}).get("spearman")
        mp = t.get("mprau", {}).get("pair_mean")
        gates[name] = {
            "mrl": {"value": mrl, "bar": SINGLE_REFS["h_bench9"]["mrl"] - RETENTION_EPS,
                    "pass": mrl is not None and mrl >= SINGLE_REFS["h_bench9"]["mrl"] - RETENTION_EPS},
            "polya": {"value": pol, "bar": SINGLE_REFS["h_bench9"]["polya"] - RETENTION_EPS,
                      "pass": pol is not None and pol >= SINGLE_REFS["h_bench9"]["polya"] - RETENTION_EPS},
            "mprau": {"value": mp, "bar": SINGLE_REFS["h_mprau_lora"]["mprau"] - RETENTION_EPS,
                      "pass": mp is not None and mp >= SINGLE_REFS["h_mprau_lora"]["mprau"] - RETENTION_EPS},
        }
        gates[name]["gate_pass_all"] = all(g["pass"] for g in gates[name].values())

    adjudication = {
        "schema_version": "route_a_v3_route2_v9_stage0_m1.v1",
        "preregistration": "SPECS_CRITIC_V6 spec N.4.3 M1 合并门: merged per-task >= max(single) - 0.01",
        "single_refs_preregistered": SINGLE_REFS,
        "single_refs_reproduced": {k: tables[k] for k in
                                   ("single_h_bench9", "single_h_mprau_lora", "single_s_mprau_in_seed1")},
        "h_family_variants": {k: v for k, v in tables.items()
                              if k.startswith(("ties", "dare", "arith"))},
        "gates": gates,
        "s_family": {"soup_5seed_weight_space": tables.get("s_soup_5seed"),
                     "prediction_ensemble_reference": PRED_ENSEMBLE_MPRAU},
        "cpu_fallback_used": False,
    }
    return adjudication


# --------------------------------------------------------------- part m2b ----
M2B_STUDIES = {
    "GSE114002": "mrl",
    "GSE269595": "polya",
    "ENCSR854RUF": "mprau",
    "GSE200304": "gse200304",
    "GSE217518": "gse217518",
}


def part_m2b(gpu_index: int, batches: int = 100, batch_pairs: int = 32) -> dict:
    device = torch.device(f"cuda:{gpu_index}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(r2.MRNABERT_PATH, local_files_only=True)

    model = r2.build_v8_regressor(r2.MRNABERT_PATH, "s",
                                  num_domains=len(r2.DOMAIN_IDS), num_cells=r2.NUM_CELLS).to(device)
    init_summary = r2.load_init(model, Path(STAGE1_S))
    model.eval()  # clean measurement: no dropout noise

    libraries = r2.load_benchmark_domains(tokenizer)
    rng = np.random.default_rng(20260908)
    named = [(n, p) for n, p in model.named_parameters()]
    backbone_elems = sum(p.numel() for n, p in named if n.startswith("base."))

    task_grads: dict[str, torch.Tensor] = {}
    task_grads_backbone: dict[str, torch.Tensor] = {}
    for study in M2B_STUDIES:
        lib, cell_ids = libraries[study]
        n_pairs = lib.targets.shape[0]
        acc = torch.zeros(sum(p.numel() for _, p in named), dtype=torch.float64)
        for _ in range(batches):
            idx = torch.as_tensor(rng.choice(n_pairs, size=batch_pairs, replace=False))
            src_ids = lib.input_ids[idx * 2]
            cnd_ids = lib.input_ids[idx * 2 + 1]
            src_mask = lib.attention_mask[idx * 2]
            cnd_mask = lib.attention_mask[idx * 2 + 1]
            targets = lib.targets[idx].to(device)
            dom = torch.full((idx.shape[0],), lib.domain_id, dtype=torch.long, device=device)
            cells = cell_ids[idx].to(device) if cell_ids is not None else None
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                src_pred = model(src_ids.to(device), src_mask.to(device), dom, cells)
                cnd_pred = model(cnd_ids.to(device), cnd_mask.to(device), dom, cells)
                loss = torch.nn.functional.mse_loss((cnd_pred - src_pred).float(), targets)
            loss.backward()
            flat = torch.cat([ (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten().double()
                               for _, p in named ])
            acc += flat.cpu()
        task_grads[study] = (acc / batches).float()
        task_grads_backbone[study] = task_grads[study][:backbone_elems].clone()
        print(f"[m2b] {study}: grad norm {float(task_grads[study].norm()):.4f} "
              f"({batches} batches x {batch_pairs} pairs)", flush=True)

    def cos_matrix(grads: dict[str, torch.Tensor]) -> dict:
        keys = sorted(grads)
        out = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                out[f"{a}|{b}"] = float(torch.nn.functional.cosine_similarity(grads[a], grads[b], dim=0))
        return out

    report = {
        "schema_version": "route_a_v3_route2_v9_stage0_m2b.v1",
        "substrate": str(STAGE1_S) + " extended to 9 domains / 6 cells (V9-1a init)",
        "init": {"loaded_keys": init_summary.get("loaded_keys")},
        "batches_per_task": batches, "batch_pairs": batch_pairs, "seed": 20260908,
        "mode": "eval (no dropout), BF16 autocast, pair-delta MSE on z-scored targets",
        "grad_norms": {k: float(v.norm()) for k, v in task_grads.items()},
        "cosine_all_params": cos_matrix(task_grads),
        "cosine_backbone_only": cos_matrix(task_grads_backbone),
        "adjudication_rule": "MPRAU-related pairs with cosine < 0 => PCGrad arm registrable; "
                             "cosines >= 0 everywhere => optimisation layer closed (representation layer confirmed)",
        "cpu_fallback_used": False,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", required=True, choices=("m2", "m1", "m2b"))
    parser.add_argument("--physical-gpu-index", type=int, default=3)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.part == "m2":
        report = part_m2()
        out = OUT_ROOT / "m2_param_overlap.json"
    elif args.part == "m1":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable - GPU required")
        report = part_m1(args.physical_gpu_index)
        out = OUT_ROOT / "m1_merge/m1_adjudication.json"
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable - GPU required")
        report = part_m2b(args.physical_gpu_index, batches=args.batches)
        out = OUT_ROOT / "m2b_grad_cosine.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
