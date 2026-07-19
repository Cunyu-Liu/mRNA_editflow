"""Audit region-adapter Stage B checkpoints.

This utility turns checkpoint/profile metadata into a durable JSON/Markdown
artifact. It is intentionally read-only: it does not instantiate sampling or
evaluation, and it never mutates checkpoints. The main invariant checked here is
that each checkpoint is a ``B_region`` adapter checkpoint whose trainable names
are all under ``adapters.*``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tail_jsonl(path: str) -> Optional[dict[str, object]]:
    if not os.path.exists(path):
        return None
    last = ""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    if not last:
        return None
    return json.loads(last)


def _load_checkpoint(path: str) -> Mapping[str, object]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    return payload


def _mode_paths(project_root: str, task_id: str, slice_name: str, mode: str) -> tuple[str, str]:
    task_lc = task_id.lower()
    ckpt = os.path.join(
        project_root,
        "ckpts",
        f"region_adapter_{task_lc}_{mode}_{slice_name}",
        f"stage_b_region_{task_lc}_best.pt",
    )
    profile = os.path.join(
        project_root,
        "logs",
        f"region_adapter_{task_lc}_{mode}_{slice_name}.profile.jsonl",
    )
    return ckpt, profile


def audit_mode(
    *,
    project_root: str,
    task_id: str,
    slice_name: str,
    mode: str,
    expected_profile_step: Optional[int] = None,
) -> dict[str, object]:
    """Audit one region-adapter mode. Complexity is dominated by checkpoint IO."""
    ckpt_path, profile_path = _mode_paths(project_root, task_id, slice_name, mode)
    row: dict[str, object] = {
        "mode": mode,
        "checkpoint_path": ckpt_path,
        "profile_path": profile_path,
        "checkpoint_exists": os.path.exists(ckpt_path),
        "profile_exists": os.path.exists(profile_path),
    }
    if not row["checkpoint_exists"]:
        row["ok"] = False
        row["error"] = "missing checkpoint"
        return row

    payload = _load_checkpoint(ckpt_path)
    trainable = [str(x) for x in payload.get("trainable_names", [])]
    changed_frozen = [str(x) for x in payload.get("changed_frozen_names", [])]
    row.update(
        {
            "checkpoint_sha256": _sha256_file(ckpt_path),
            "checkpoint_bytes": int(os.path.getsize(ckpt_path)),
            "stage": payload.get("stage"),
            "task_id": payload.get("task_id"),
            "checkpoint_step": payload.get("step"),
            "best_loss": payload.get("best_loss"),
            "region_ids": payload.get("region_ids"),
            "adapter_bottleneck": payload.get("adapter_bottleneck"),
            "n_trainable_names": len(trainable),
            "all_trainable_adapters": bool(trainable)
            and all(name.startswith("adapters.") for name in trainable),
            "changed_frozen_names": changed_frozen,
            "changed_frozen_count": len(changed_frozen),
        }
    )

    profile = _tail_jsonl(profile_path)
    if profile is not None:
        row.update(
            {
                "profile_sha256": _sha256_file(profile_path),
                "profile_bytes": int(os.path.getsize(profile_path)),
                "profile_step": profile.get("step"),
                "profile_stage": profile.get("stage"),
                "profile_task_id": profile.get("task_id"),
                "profile_region_ids": profile.get("region_ids"),
                "profile_finite_loss": profile.get("finite_loss"),
                "profile_finite_grad": profile.get("finite_grad"),
                "profile_loss": profile.get("loss"),
                "profile_grad_norm": profile.get("grad_norm"),
            }
        )
    if expected_profile_step is not None:
        row["profile_reached_expected_step"] = bool(
            profile is not None and int(profile.get("step", -1)) >= int(expected_profile_step)
        )

    row["ok"] = bool(
        row.get("stage") == "B_region"
        and row.get("task_id") == task_id
        and row.get("all_trainable_adapters") is True
        and row.get("changed_frozen_count") == 0
        and (profile is None or row.get("profile_finite_loss") is True)
        and (profile is None or row.get("profile_finite_grad") is True)
        and (
            expected_profile_step is None
            or row.get("profile_reached_expected_step") is True
        )
    )
    return row


def summarize_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize region-adapter checkpoint audit rows."""
    return {
        "n_modes": len(rows),
        "all_ok": bool(rows) and all(bool(row.get("ok")) for row in rows),
        "all_checkpoints_exist": bool(rows) and all(bool(row.get("checkpoint_exists")) for row in rows),
        "all_profiles_exist": bool(rows) and all(bool(row.get("profile_exists")) for row in rows),
        "all_b_region": bool(rows) and all(row.get("stage") == "B_region" for row in rows),
        "all_trainable_adapters": bool(rows)
        and all(row.get("all_trainable_adapters") is True for row in rows),
        "changed_frozen_total": int(sum(int(row.get("changed_frozen_count", 0)) for row in rows)),
    }


def write_markdown(payload: Mapping[str, object], out_md: str) -> None:
    rows = list(payload.get("rows", []))
    lines = [
        "# Region Adapter Checkpoint Audit",
        "",
        f"- Slice: {payload.get('slice')}",
        f"- Task: {payload.get('task_id')}",
        f"- All OK: {payload.get('summary', {}).get('all_ok') if isinstance(payload.get('summary'), Mapping) else None}",
        "",
        "| mode | ok | stage | region_ids | ckpt step | profile step | adapter-only | ckpt sha256 |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        sha = str(row.get("checkpoint_sha256", ""))
        lines.append(
            "| {mode} | {ok} | {stage} | {region_ids} | {ckpt_step} | {profile_step} | {adapter_only} | `{sha}` |".format(
                mode=row.get("mode"),
                ok=row.get("ok"),
                stage=row.get("stage"),
                region_ids=row.get("region_ids"),
                ckpt_step=row.get("checkpoint_step", ""),
                profile_step=row.get("profile_step", ""),
                adapter_only=row.get("all_trainable_adapters"),
                sha=sha[:12] + ("..." if len(sha) > 12 else ""),
            )
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_audit(
    *,
    project_root: str,
    task_id: str = "T5",
    slice_name: str = "head256",
    modes: Sequence[str] = ("utr5", "cds", "utr3", "all"),
    expected_profile_step: Optional[int] = 500,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
) -> dict[str, object]:
    rows = [
        audit_mode(
            project_root=project_root,
            task_id=task_id,
            slice_name=slice_name,
            mode=mode,
            expected_profile_step=expected_profile_step,
        )
        for mode in modes
    ]
    payload = {
        "artifact_kind": "region_adapter_checkpoint_audit",
        "project_root": project_root,
        "task_id": task_id,
        "slice": slice_name,
        "modes": list(modes),
        "expected_profile_step": expected_profile_step,
        "summary": summarize_audit(rows),
        "rows": rows,
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    if out_md:
        write_markdown(payload, out_md)
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--task-id", default="T5")
    parser.add_argument("--slice", dest="slice_name", default="head256")
    parser.add_argument("--modes", nargs="+", default=["utr5", "cds", "utr3", "all"])
    parser.add_argument("--expected-profile-step", type=int, default=500)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = run_audit(
        project_root=args.project_root,
        task_id=args.task_id,
        slice_name=args.slice_name,
        modes=args.modes,
        expected_profile_step=args.expected_profile_step,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(json.dumps({"summary": payload["summary"], "out_json": args.out_json, "out_md": args.out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
