#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from validate_dec028_static_bundle import validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    issues = validate(args.repo_root)
    args.run_root.mkdir(parents=True, exist_ok=False)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {"record_type": "DEC028_SS0_STATIC_PREFLIGHT", "generated_at": generated_at, "issue_count": len(issues), "issues": issues, "scientific_claim_status": "NOT_ESTABLISHED", "data_rows_accessed": False, "cuda_probed": False, "model_or_optimizer_constructed": False, "checkpoint_accessed": False, "parameter_updates": 0, "g1_launched": False}
    acceptance = {"record_type": "DEC028_SS0_ACCEPTANCE", "status": "PASS_STATIC_AUTHORITY_ONLY" if not issues else "FAIL_STATIC_AUTHORITY_ONLY", "scientific_claim_status": "NOT_ESTABLISHED", "next_phase": "FRESH_RUNTIME_AUTHORITY_SYNC_REQUIRED", "g1_launched": False}
    manifest = {"record_type": "DEC028_SS0_RUN_MANIFEST", "run_id": args.run_root.name, "scope": "STATIC_AUTHORITY_ONLY", "outputs": ["SS0_STATIC_PREFLIGHT.json", "RUN_MANIFEST.json", "ACCEPTANCE.json", "RUN_LOG.md"], "data_rows_accessed": False, "cuda_probed": False, "parameter_updates": 0}
    (args.run_root / "SS0_STATIC_PREFLIGHT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.run_root / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.run_root / "ACCEPTANCE.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    (args.run_root / "RUN_LOG.md").write_text("generated_at: " + generated_at + "\nscope: STATIC_AUTHORITY_ONLY\nissue_count: " + str(len(issues)) + "\n")
    return 0 if not issues else 1

if __name__ == "__main__":
    raise SystemExit(main())
