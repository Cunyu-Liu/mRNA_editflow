#!/usr/bin/env python
"""P0-02: result-artifact consistency audit.

Checks every paper-candidate result artifact for internal and cross-artifact
consistency so that every headline number is traceable through:

    raw data -> preprocessing -> split -> checkpoint -> inference output
    -> statistical analysis -> final table cell

Checks implemented
------------------
1. empty_json               zero-byte / unparseable JSON artifacts
2. missing_references       files referenced by docs/JSON but absent from repo
3. checkpoint_correspondence result seeds <-> checkpoint files on disk
4. gate_status_conflict     Gate A/B verdicts and criteria disagree across files
5. hash_mismatch            .sha256 sidecar files that do not match their payload
6. record_count_mismatch    declared counts (n_seeds, n_pairs, ...) vs actual
7. summary_vs_records       aggregate statistics recomputed from per-record data
8. frozen_artifact_overwrite files whose sha256 differs from the freeze manifest

Usage:
    python scripts/audit_result_artifacts.py [--repo-root .] [--write-report] [--strict]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SEVERITY_FAIL = "FAIL"
SEVERITY_WARN = "WARN"

# Result artifacts that feed paper-candidate numbers (per P0-02 spec).
SPECIAL_JSON_GLOBS = [
    "docs/p3_07_search_results.json",
    "docs/p3_08_grpo_results*.json",
    "docs/p3_09_oracle_transfer.json",
    "docs/p3_10_synergy*.json",
    "docs/p3_10_run_*/p3_10_synergy*.json",
    "docs/p3_11_*.json",
    "docs/p3_11_run_*/p3_11_*.json",
]

# Markdown / JSON reference extensions we treat as repo artifacts.
_REF_EXTENSIONS = (".json", ".pt", ".pth", ".py", ".sh", ".md", ".yaml", ".yml",
                   ".tsv", ".csv", ".npy", ".npz", ".sha256")

# Directory names that are never treated as audit-relevant JSON artifacts.
_JSON_SCAN_DIRS = ("docs", "artifacts", "configs")


@dataclass
class Finding:
    check: str
    severity: str
    artifact: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    generated_at: str
    repo_root: str
    findings: List[Finding] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)
    traceability: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, check: str, severity: str, artifact: str, message: str,
            **details: Any) -> None:
        self.findings.append(Finding(check, severity, artifact, message, details))

    @property
    def n_fail(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_FAIL)

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_WARN)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "repo_root": self.repo_root,
            "n_fail": self.n_fail,
            "n_warn": self.n_warn,
            "checks_run": self.checks_run,
            "findings": [vars(f) for f in self.findings],
            "traceability": self.traceability,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """Return (data, error). error is None on success."""
    try:
        text = path.read_text()
    except OSError as exc:
        return None, f"unreadable: {exc}"
    if not text.strip():
        return None, "empty file"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"


def _collect_special_jsons(repo_root: Path) -> List[Path]:
    out: List[Path] = []
    for pattern in SPECIAL_JSON_GLOBS:
        out.extend(repo_root.glob(pattern))
    return sorted(set(p for p in out if p.is_file()))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------------
# check 1: empty / unparseable JSON
# ---------------------------------------------------------------------------

def check_empty_json(repo_root: Path, report: AuditReport) -> None:
    for dirname in _JSON_SCAN_DIRS:
        base = repo_root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            rel = str(path.relative_to(repo_root))
            if path.stat().st_size == 0:
                report.add("empty_json", SEVERITY_FAIL, rel, "zero-byte JSON artifact")
                continue
            _, err = _load_json(path)
            if err:
                report.add("empty_json", SEVERITY_FAIL, rel, err)


# ---------------------------------------------------------------------------
# check 2: references to files that do not exist
# ---------------------------------------------------------------------------

_MD_REF_RE = re.compile(r"[\`\(]\s*([A-Za-z0-9_./-]+\.(?:json|pt|pth|py|sh|md|yaml|yml|tsv|csv|npy|npz|sha256))[\`\)]")
_JSON_PATH_KEYS = re.compile(r"(path|file|checkpoint|ckpt|manifest)", re.IGNORECASE)


def _iter_json_strings(obj: Any, key_hint: str = ""):
    """Yield (key_hint, string_value) for strings whose key looks path-like."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and _JSON_PATH_KEYS.search(str(k)):
                yield str(k), v
            else:
                yield from _iter_json_strings(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_json_strings(v, key_hint)


# Top-level directories that hold repo artifacts; a missing reference whose
# first component is one of these is a FAIL, otherwise (external repo, prose)
# it is a WARN.
_REPO_TOP_LEVELS = {"artifacts", "baselines", "benchmark", "benchmark_v21",
                    "checkpoints", "ckpts", "configs", "core", "data", "docs",
                    "eval", "external_tools", "logs", "models", "rl", "scripts",
                    "tests", "mrna_editflow"}


# Result-asset extensions: a missing reference to one of these under a repo
# top-level directory is a FAIL (a result/checkpoint/data asset is gone).
# Missing .md/.py/.sh/.yaml references are documentation drift or planned
# files mentioned in design docs — WARN.
_RESULT_ASSET_EXTENSIONS = (".json", ".pt", ".pth", ".npy", ".npz", ".sha256",
                            ".tsv", ".csv")


def _resolve_ref(repo_root: Path, md: Path, ref: str) -> bool:
    """True if the referenced file exists (several resolution conventions)."""
    candidates = [repo_root / ref, md.parent / ref]
    # historical docs use the package-prefixed path (mrna_editflow/...)
    if ref.startswith("mrna_editflow/"):
        candidates.append(repo_root / ref[len("mrna_editflow/"):])
    return any(c.exists() for c in candidates)


def check_missing_references(repo_root: Path, report: AuditReport) -> None:
    docs = repo_root / "docs"
    if docs.is_dir():
        for md in sorted(docs.rglob("*.md")):
            rel_md = str(md.relative_to(repo_root))
            try:
                text = md.read_text(errors="replace")
            except OSError:
                continue
            seen: set = set()
            for m in _MD_REF_RE.finditer(text):
                ref = m.group(1)
                if ref.startswith(("http", "../", "/")) or ref in seen:
                    continue
                if "..." in ref:  # literal placeholder in prose
                    continue
                seen.add(ref)
                if _resolve_ref(repo_root, md, ref):
                    continue
                first = ref.split("/", 1)[0]
                is_result_asset = ref.endswith(_RESULT_ASSET_EXTENSIONS)
                severity = (SEVERITY_FAIL
                            if first in _REPO_TOP_LEVELS and is_result_asset
                            else SEVERITY_WARN)
                report.add("missing_references", severity, rel_md,
                           f"referenced file not found: {ref}", reference=ref)

    for path in _collect_special_jsons(repo_root):
        rel = str(path.relative_to(repo_root))
        data, err = _load_json(path)
        if err:
            continue
        for key, value in _iter_json_strings(data):
            for token in re.findall(r"[A-Za-z0-9_./-]+\.(?:json|pt|pth)", value):
                if token.startswith("http"):
                    continue
                if not (repo_root / token).exists() and not (path.parent / token).exists():
                    report.add("missing_references", SEVERITY_FAIL, rel,
                               f"JSON field {key!r} references missing file: {token}",
                               field=key, reference=token)


# ---------------------------------------------------------------------------
# check 3: results <-> checkpoint correspondence (p3_08)
# ---------------------------------------------------------------------------

def check_checkpoint_correspondence(repo_root: Path, report: AuditReport) -> None:
    specs = [
        # (results glob, checkpoint dir, step values, seed json field)
        ("docs/p3_08_grpo_results_gateA.json", "checkpoints/p3_08_gateA",
         [200, 400, 600, 800, 1000]),
        ("docs/p3_08_grpo_results_gateB_gpu1.json", "checkpoints/p3_08_gateB_gpu1",
         [1000, 2000, 3000, 4000, 5000]),
        ("docs/p3_08_grpo_results_gateB_gpu6.json", "checkpoints/p3_08_gateB_gpu6",
         [1000, 2000, 3000, 4000, 5000]),
    ]
    for results_rel, ckpt_dir_rel, steps in specs:
        results_path = repo_root / results_rel
        if not results_path.exists():
            report.add("checkpoint_correspondence", SEVERITY_FAIL, results_rel,
                       "results file missing")
            continue
        data, err = _load_json(results_path)
        if err:
            continue
        seeds = sorted({r.get("seed") for r in data.get("results", []) if "seed" in r})
        ckpt_dir = repo_root / ckpt_dir_rel
        for seed in seeds:
            for step in steps:
                ckpt = ckpt_dir / f"grpo_seed{seed}_step{step}.pt"
                if not ckpt.exists():
                    report.add("checkpoint_correspondence", SEVERITY_FAIL, results_rel,
                               f"seed {seed} in results but checkpoint missing",
                               checkpoint=str(ckpt.relative_to(repo_root)))
        # orphan checkpoints: on disk but seed not in results
        if ckpt_dir.is_dir():
            on_disk = {int(m.group(1)) for p in ckpt_dir.glob("grpo_seed*_step*.pt")
                       if (m := re.search(r"grpo_seed(\d+)_step", p.name))}
            for orphan in sorted(on_disk - set(seeds)):
                report.add("checkpoint_correspondence", SEVERITY_WARN, ckpt_dir_rel,
                           f"checkpoints for seed {orphan} exist but seed absent from results",
                           seed=orphan)


# ---------------------------------------------------------------------------
# check 4: Gate A/B status conflicts
# ---------------------------------------------------------------------------

def _criteria_arrays_equal(a: Dict[str, Any], b: Dict[str, Any],
                           keys: Sequence[str]) -> List[str]:
    mismatched = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb) or any(abs((x or 0) - (y or 0)) > 1e-9
                                         if isinstance(x, (int, float)) and isinstance(y, (int, float))
                                         else x != y for x, y in zip(va, vb)):
                mismatched.append(k)
        elif va != vb:
            mismatched.append(k)
    return mismatched


def check_gate_status_conflict(repo_root: Path, report: AuditReport) -> None:
    gate_a_path = repo_root / "docs/p3_08_grpo_results_gateA.json"
    gate_b_path = repo_root / "docs/p3_08_grpo_results_gateB.json"
    shard_paths = [repo_root / "docs/p3_08_grpo_results_gateB_gpu1.json",
                   repo_root / "docs/p3_08_grpo_results_gateB_gpu6.json"]
    top_path = repo_root / "docs/p3_08_grpo_results.json"

    loaded: Dict[str, Any] = {}
    for p in [gate_a_path, gate_b_path, top_path, *shard_paths]:
        if p.exists():
            d, err = _load_json(p)
            if not err:
                loaded[p.name] = d

    # 4a: merged Gate B verdict vs shard verdicts
    shards = [loaded[p.name] for p in shard_paths if p.name in loaded]
    if "p3_08_grpo_results_gateB.json" in loaded and len(shards) == 2:
        merged = loaded["p3_08_grpo_results_gateB.json"]
        shard_verdicts = [s.get("verdict") for s in shards]
        expected = "PASS" if all(v == "PASS" for v in shard_verdicts) else "FAIL"
        if merged.get("verdict") != expected:
            report.add("gate_status_conflict", SEVERITY_FAIL,
                       "docs/p3_08_grpo_results_gateB.json",
                       f"merged verdict {merged.get('verdict')} != shard verdicts "
                       f"{shard_verdicts} (expected {expected})",
                       shard_verdicts=shard_verdicts)
        # criteria arrays of merged must equal concatenation of shard arrays
        array_keys = ["finite_fractions", "constraint_validities", "beat_warm_start",
                      "positive_rates", "stop_rates"]
        concat: Dict[str, List[Any]] = {k: [] for k in array_keys}
        for s in shards:
            crit = s.get("criteria", {})
            for k in array_keys:
                v = crit.get(k)
                if isinstance(v, list):
                    concat[k].extend(v)
        mcrit = merged.get("criteria", {})
        for k in array_keys:
            mv = mcrit.get(k)
            if isinstance(mv, list) and concat[k]:
                # compare as multisets: shard concatenation order in the merged
                # file depends on merge argument order, which is not canonical
                def _canon(vals):
                    return sorted(str(round(v, 9)) if isinstance(v, float) else str(v)
                                  for v in vals)
                if _canon(mv) != _canon(concat[k]):
                    report.add("gate_status_conflict", SEVERITY_FAIL,
                               "docs/p3_08_grpo_results_gateB.json",
                               f"merged criteria.{k} inconsistent with shard criteria",
                               merged=mv, shards_concat=concat[k])
        # boolean criteria must agree
        for k in ["no_collapse", "hard_constraints_100", "two_thirds_beat_warm",
                  "no_reward_hacking", "stop_not_collapsed"]:
            shard_vals = [s.get("criteria", {}).get(k) for s in shards]
            expected_bool = all(v is True for v in shard_vals)
            if k in mcrit and bool(mcrit[k]) != expected_bool:
                report.add("gate_status_conflict", SEVERITY_FAIL,
                           "docs/p3_08_grpo_results_gateB.json",
                           f"merged criteria.{k}={mcrit[k]} but shards give {shard_vals}",
                           shard_values=shard_vals)

    # 4b: top-level gate_b_verdict vs gateB.json verdict
    if "p3_08_grpo_results.json" in loaded and "p3_08_grpo_results_gateB.json" in loaded:
        top_v = loaded["p3_08_grpo_results.json"].get("gate_b_verdict")
        b_v = loaded["p3_08_grpo_results_gateB.json"].get("verdict")
        if top_v is not None and b_v is not None and top_v != b_v:
            report.add("gate_status_conflict", SEVERITY_FAIL,
                       "docs/p3_08_grpo_results.json",
                       f"gate_b_verdict={top_v} conflicts with "
                       f"p3_08_grpo_results_gateB.json verdict={b_v}",
                       gate_b_verdict=top_v, gate_b_file_verdict=b_v)

    # 4c: gate A verdict vs its own criteria all_pass (if present)
    if "p3_08_grpo_results_gateA.json" in loaded:
        ga = loaded["p3_08_grpo_results_gateA.json"]
        crit = ga.get("criteria", {})
        if "all_pass" in crit:
            expected = "PASS" if crit["all_pass"] else "FAIL"
            if ga.get("verdict") != expected:
                report.add("gate_status_conflict", SEVERITY_FAIL,
                           "docs/p3_08_grpo_results_gateA.json",
                           f"verdict={ga.get('verdict')} but criteria.all_pass={crit['all_pass']}")


# ---------------------------------------------------------------------------
# check 5: .sha256 sidecars
# ---------------------------------------------------------------------------

def check_hash_sidecars(repo_root: Path, report: AuditReport) -> None:
    for sidecar in sorted(repo_root.rglob("*.sha256")):
        if any(part.startswith(".") for part in sidecar.parts):
            continue
        rel = str(sidecar.relative_to(repo_root))
        try:
            content = sidecar.read_text().strip()
        except OSError:
            continue
        # formats: "<hash>  <filename>" (sha256sum) or bare "<hash>"
        parts = content.split()
        if not parts:
            report.add("hash_mismatch", SEVERITY_WARN, rel, "empty sha256 sidecar")
            continue
        expected_hash = parts[0]
        if len(parts) >= 2:
            payload = sidecar.parent / parts[-1].lstrip("*")
        else:
            payload = sidecar.with_suffix("")  # strip .sha256
        if not payload.exists():
            report.add("hash_mismatch", SEVERITY_WARN, rel,
                       f"sidecar payload missing: {payload.name}")
            continue
        actual = _sha256_file(payload)
        if actual != expected_hash:
            report.add("hash_mismatch", SEVERITY_FAIL, rel,
                       f"sha256 mismatch for {payload.relative_to(repo_root)}",
                       expected=expected_hash, actual=actual)


# ---------------------------------------------------------------------------
# check 6: record counts
# ---------------------------------------------------------------------------

def check_record_counts(repo_root: Path, report: AuditReport) -> None:
    # p3_08-style files: n_seeds / n_seeds_completed / n_seeds_failed vs results
    for path in _collect_special_jsons(repo_root):
        rel = str(path.relative_to(repo_root))
        data, err = _load_json(path)
        if err or not isinstance(data, dict):
            continue
        results = data.get("results") or data.get("seeds")
        if isinstance(results, list) and "n_seeds" in data:
            if data["n_seeds"] != len(results):
                report.add("record_count_mismatch", SEVERITY_FAIL, rel,
                           f"n_seeds={data['n_seeds']} but {len(results)} per-seed records",
                           declared=data["n_seeds"], actual=len(results))
            completed = data.get("n_seeds_completed")
            failed = data.get("n_seeds_failed")
            if completed is not None and failed is not None:
                if completed + failed != data["n_seeds"]:
                    report.add("record_count_mismatch", SEVERITY_FAIL, rel,
                               f"n_seeds_completed({completed})+n_seeds_failed({failed})"
                               f" != n_seeds({data['n_seeds']})")
                if completed != len(results):
                    report.add("record_count_mismatch", SEVERITY_FAIL, rel,
                               f"n_seeds_completed={completed} but {len(results)} records")

    # p3_09: n_pairs vs per_pair_records
    p9 = repo_root / "docs/p3_09_oracle_transfer.json"
    if p9.exists():
        data, err = _load_json(p9)
        if not err:
            records = data.get("per_pair_records", [])
            declared = data.get("n_pairs")
            if declared is not None and isinstance(records, list) and declared != len(records):
                report.add("record_count_mismatch", SEVERITY_FAIL,
                           "docs/p3_09_oracle_transfer.json",
                           f"n_pairs={declared} but {len(records)} per_pair_records",
                           declared=declared, actual=len(records))

    # p3_10: per_sequence count vs config.n_sequences if declared
    for p10 in sorted(repo_root.glob("docs/p3_10_run_*/p3_10_synergy_results.json")):
        data, err = _load_json(p10)
        if err:
            continue
        per_seq = data.get("per_sequence", [])
        n_declared = data.get("config", {}).get("n_sequences")
        rel = str(p10.relative_to(repo_root))
        if n_declared is not None and isinstance(per_seq, list) and n_declared != len(per_seq):
            report.add("record_count_mismatch", SEVERITY_FAIL, rel,
                       f"config.n_sequences={n_declared} but {len(per_seq)} per_sequence records",
                       declared=n_declared, actual=len(per_seq))


# ---------------------------------------------------------------------------
# check 7: summary vs per-record recomputation
# ---------------------------------------------------------------------------

def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def check_summary_vs_records(repo_root: Path, report: AuditReport) -> None:
    # 7a: Gate B merged summary_stats recomputed from results
    gb = repo_root / "docs/p3_08_grpo_results_gateB.json"
    if gb.exists():
        data, err = _load_json(gb)
        if not err:
            rel = "docs/p3_08_grpo_results_gateB.json"
            results = data.get("results", [])
            stats = data.get("summary_stats", {})
            if results and stats:
                fv = [r.get("final_validation", {}) for r in results]
                pos = [v.get("positive_improvement_rate", 0.0) for v in fv]
                declared = stats.get("pos_rate", {}).get("mean")
                if declared is not None and pos and not _close(declared, _mean(pos), 1e-4):
                    report.add("summary_vs_records", SEVERITY_FAIL, rel,
                               f"summary_stats.pos_rate.mean={declared} but recomputed "
                               f"{_mean(pos)} from {len(pos)} per-seed records",
                               declared=declared, recomputed=_mean(pos))
                values = stats.get("pos_rate", {}).get("values")
                if isinstance(values, list) and values and pos:
                    if len(values) != len(pos) or any(
                            not _close(a, b, 1e-6) for a, b in zip(values, pos)):
                        report.add("summary_vs_records", SEVERITY_FAIL, rel,
                                   "summary_stats.pos_rate.values != per-seed "
                                   "final_validation.positive_improvement_rate")

    # 7b: top-level p3_08 results: criteria arrays vs per-seed fields
    top = repo_root / "docs/p3_08_grpo_results.json"
    if top.exists():
        data, err = _load_json(top)
        if not err:
            rel = "docs/p3_08_grpo_results.json"
            seeds = data.get("seeds", [])
            crit = data.get("gate_b_criteria", {})
            if seeds:
                if not _close(_mean([s.get("final_constraint", 0) for s in seeds]), 1.0, 1e-9) \
                        and crit.get("constraint_100pct"):
                    report.add("summary_vs_records", SEVERITY_FAIL, rel,
                               "gate_b_criteria.constraint_100pct=true but per-seed "
                               "final_constraint mean < 1.0")
                n_beat = sum(1 for s in seeds if s.get("beat_warm_start"))
                if n_beat < (2 * len(seeds)) // 3 + 1 and data.get("gate_b_verdict") == "PASS":
                    report.add("summary_vs_records", SEVERITY_FAIL, rel,
                               f"gate_b_verdict=PASS but only {n_beat}/{len(seeds)} seeds "
                               "beat warm start")

    # 7c: p3_09 per_oracle_summary vs per_pair_records
    p9 = repo_root / "docs/p3_09_oracle_transfer.json"
    if p9.exists():
        data, err = _load_json(p9)
        if not err:
            records = data.get("per_pair_records", [])
            summary = data.get("per_oracle_summary", {})
            if records and isinstance(summary, dict) and summary:
                # group records by oracle pair if the field exists
                key_field = next((k for k in ("pair", "oracle_pair", "name")
                                  if records and k in records[0]), None)
                if key_field:
                    from collections import defaultdict
                    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
                    for r in records:
                        groups[r.get(key_field)].append(r)
                    for gname, grows in groups.items():
                        s = summary.get(str(gname)) or summary.get(gname)
                        if not isinstance(s, dict):
                            continue
                        for metric, skey in (("mean_transfer", "mean_transfer"),
                                             ("mean_delta", "mean_delta")):
                            vals = [r.get(metric) for r in grows
                                    if isinstance(r.get(metric), (int, float))]
                            declared = s.get(skey)
                            if vals and declared is not None and not _close(
                                    declared, _mean(vals), 1e-4):
                                report.add("summary_vs_records", SEVERITY_FAIL,
                                           "docs/p3_09_oracle_transfer.json",
                                           f"per_oracle_summary[{gname}].{skey}={declared} "
                                           f"but recomputed {_mean(vals)}",
                                           group=str(gname), declared=declared,
                                           recomputed=_mean(vals))


# ---------------------------------------------------------------------------
# check 8: frozen artifact overwrite (vs freeze manifest)
# ---------------------------------------------------------------------------

def check_frozen_overwrite(repo_root: Path, report: AuditReport,
                           manifest_path: Optional[Path] = None) -> None:
    manifest_path = manifest_path or repo_root / "artifacts/nmi_phase0_freeze_manifest.json"
    if not manifest_path.exists():
        report.add("frozen_artifact_overwrite", SEVERITY_WARN,
                   "artifacts/nmi_phase0_freeze_manifest.json",
                   "freeze manifest not found; run scripts/build_freeze_manifest.py first")
        return
    data, err = _load_json(manifest_path)
    if err:
        return
    artifacts = data.get("artifacts", {})
    n_checked = 0
    for group_name, group in artifacts.items():
        files = group.get("files", []) if isinstance(group, dict) else []
        for meta in files:
            if not isinstance(meta, dict):
                continue
            rel = meta.get("path")
            recorded = meta.get("sha256")
            if not rel or not recorded:
                continue
            path = repo_root / rel
            if not path.exists():
                report.add("frozen_artifact_overwrite", SEVERITY_FAIL, rel,
                           f"frozen artifact (group {group_name}) was deleted after freeze",
                           group=group_name)
                continue
            n_checked += 1
            actual = _sha256_file(path)
            if actual != recorded:
                report.add("frozen_artifact_overwrite", SEVERITY_FAIL, rel,
                           f"frozen artifact (group {group_name}) modified after freeze",
                           group=group_name, frozen_sha256=recorded, current_sha256=actual)
    if n_checked == 0:
        report.add("frozen_artifact_overwrite", SEVERITY_WARN,
                   str(manifest_path.relative_to(repo_root)),
                   "freeze manifest contains no file hashes to verify")


# ---------------------------------------------------------------------------
# traceability chain (acceptance criterion)
# ---------------------------------------------------------------------------

def build_traceability(repo_root: Path, report: AuditReport) -> None:
    """For every special result artifact, verify it appears in the freeze
    manifest with a sha256 and that its declared inputs exist and are hashed."""
    manifest_path = repo_root / "artifacts/nmi_phase0_freeze_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        data, err = _load_json(manifest_path)
        if not err:
            manifest = data
    artifacts = manifest.get("artifacts", {})
    # index: relative path -> (group, sha256)
    index: Dict[str, Tuple[str, str]] = {}
    for gname, group in artifacts.items():
        if not isinstance(group, dict):
            continue
        for meta in group.get("files", []):
            if isinstance(meta, dict) and meta.get("path") and meta.get("sha256"):
                index[meta["path"]] = (gname, meta["sha256"])

    for path in _collect_special_jsons(repo_root):
        rel = str(path.relative_to(repo_root))
        entry: Dict[str, Any] = {"artifact": rel, "in_manifest": rel in index}
        if rel in index:
            gname, sha = index[rel]
            entry["manifest_group"] = gname
            entry["sha256"] = sha
            entry["current_sha256"] = _sha256_file(path)
            entry["hash_matches"] = entry["current_sha256"] == sha
            group = artifacts.get(gname, {})
            prov = group.get("provenance", {}) if isinstance(group, dict) else {}
            inputs = prov.get("inputs", []) if isinstance(prov, dict) else []
            entry["provenance_command"] = prov.get("command") if isinstance(prov, dict) else None
            entry["declared_inputs"] = inputs
            entry["inputs_present"] = all((repo_root / i).exists() for i in inputs)
        else:
            report.add("traceability", SEVERITY_WARN, rel,
                       "paper-candidate artifact not recorded in freeze manifest")
        report.traceability.append(entry)


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------

def render_markdown(report: AuditReport) -> str:
    lines = [
        "# NMI Artifact Discrepancy Report (P0-02)",
        "",
        f"Generated: {report.generated_at}",
        f"Repo root: `{report.repo_root}`",
        "",
        f"**FAIL findings: {report.n_fail}** | WARN findings: {report.n_warn}",
        "",
        "## Findings",
        "",
    ]
    if not report.findings:
        lines.append("No discrepancies found.")
    else:
        lines.append("| Check | Severity | Artifact | Message |")
        lines.append("|---|---|---|---|")
        for f in report.findings:
            msg = f.message.replace("|", "\\|")
            lines.append(f"| {f.check} | {f.severity} | `{f.artifact}` | {msg} |")
    lines += ["", "## Traceability (paper-candidate numbers)", "",
              "| Artifact | In freeze manifest | Hash matches | Inputs present |",
              "|---|---|---|---|"]
    for t in report.traceability:
        lines.append(
            f"| `{t['artifact']}` | {t.get('in_manifest')} | "
            f"{t.get('hash_matches', 'n/a')} | {t.get('inputs_present', 'n/a')} |")
    lines += [
        "",
        "Traceability chain per artifact: raw data → preprocessing → split → "
        "checkpoint → inference output → statistical analysis → final table cell. "
        "Each artifact's declared inputs and provenance command are recorded in "
        "`configs/nmi_execution.yaml` and hashed in "
        "`artifacts/nmi_phase0_freeze_manifest.json`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# documented dispositions
# ---------------------------------------------------------------------------

def load_dispositions(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    """Load documented dispositions (docs/nmi_artifact_dispositions.json).

    Returns {(check, reference): disposition_entry}. A matching FAIL finding
    is downgraded to WARN and annotated with the disposition rationale.
    """
    path = repo_root / "docs/nmi_artifact_dispositions.json"
    if not path.exists():
        return {}
    data, err = _load_json(path)
    if err or not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("dispositions", []):
        key = (entry.get("check", ""), entry.get("reference", ""))
        out[key] = entry
    return out


def apply_dispositions(report: AuditReport,
                       dispositions: Dict[str, Dict[str, Any]]) -> None:
    if not dispositions:
        return
    for f in report.findings:
        ref = f.details.get("reference", "")
        entry = dispositions.get((f.check, ref))
        if entry and f.severity == SEVERITY_FAIL:
            f.severity = SEVERITY_WARN
            f.message += (f" [disposition: {entry.get('disposition')} — "
                          f"{entry.get('impact')}]")
            f.details["disposition"] = entry.get("disposition")
            f.details["disposition_rationale"] = entry.get("rationale")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_audit(repo_root: Path,
              manifest_path: Optional[Path] = None) -> AuditReport:
    report = AuditReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        repo_root=str(repo_root),
    )
    checks = [
        ("empty_json", check_empty_json),
        ("missing_references", check_missing_references),
        ("checkpoint_correspondence", check_checkpoint_correspondence),
        ("gate_status_conflict", check_gate_status_conflict),
        ("hash_mismatch", check_hash_sidecars),
        ("record_count_mismatch", check_record_counts),
        ("summary_vs_records", check_summary_vs_records),
        ("frozen_artifact_overwrite",
         lambda root, rep: check_frozen_overwrite(root, rep, manifest_path)),
    ]
    for name, fn in checks:
        fn(repo_root, report)
        report.checks_run.append(name)
    apply_dispositions(report, load_dispositions(repo_root))
    build_traceability(repo_root, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--manifest", default=None,
                        help="Path to freeze manifest (default: artifacts/nmi_phase0_freeze_manifest.json)")
    parser.add_argument("--write-report", action="store_true",
                        help="Write artifacts/nmi_artifact_audit.json and "
                             "docs/nmi_artifact_discrepancy_report.md")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any FAIL-severity finding")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    manifest = Path(args.manifest) if args.manifest else None
    report = run_audit(repo_root, manifest)

    print(f"[audit] checks run: {', '.join(report.checks_run)}")
    print(f"[audit] FAIL={report.n_fail} WARN={report.n_warn}")
    for f in report.findings:
        if f.severity == SEVERITY_FAIL:
            print(f"  FAIL [{f.check}] {f.artifact}: {f.message}")

    if args.write_report:
        out_json = repo_root / "artifacts/nmi_artifact_audit.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report.to_dict(), indent=2))
        out_md = repo_root / "docs/nmi_artifact_discrepancy_report.md"
        out_md.write_text(render_markdown(report))
        print(f"[audit] wrote {out_json.relative_to(repo_root)} and "
              f"{out_md.relative_to(repo_root)}")

    if args.strict and report.n_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
