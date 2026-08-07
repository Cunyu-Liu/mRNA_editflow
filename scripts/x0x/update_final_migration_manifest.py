"""Update FINAL_MIGRATION_MANIFEST.json terminal_state + report hash, and
regenerate FINAL_MIGRATION_SHA256SUMS. Pure dev-finalization; no data touched."""
import hashlib
import json
from pathlib import Path

WT = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


man_path = WT / "artifacts/migration/FINAL_MIGRATION_MANIFEST.json"
report = WT / "reports/migration/FINAL_MIGRATION_REPORT.md"

man = json.loads(man_path.read_text())
old_state = man["terminal_state"]
man["terminal_state"] = "BLOCKED_WITH_EVIDENCE"
man["date"] = "2026-08-07"
man["report_note"] = (
    "Terminal state updated from MIGRATION_READY_FOR_DATA_REBUILD to "
    "BLOCKED_WITH_EVIDENCE after full migration execution (B0-X->X0-X incl. "
    "CDS-B1 rebuild audit). Sealed-final (GSE246381) decision and CDS-B1 "
    "sequence recovery remain the blockers for a GO/NO-GO declaration.")
man["artifacts"]["reports/migration/FINAL_MIGRATION_REPORT.md"] = sha256_file(report)

man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")

# regenerate FINAL_MIGRATION_SHA256SUMS for all artifacts listed in manifest
sums = {k: sha256_file(WT / k) for k in man["artifacts"]}
# add the manifest itself + report (not self-hashed in manifest)
extra = {
    "artifacts/migration/FINAL_MIGRATION_MANIFEST.json": sha256_file(man_path),
    "reports/migration/FINAL_MIGRATION_REPORT.md": sha256_file(report),
}
lines = sorted(
    f"{v}  {k}\n" for k, v in {**sums, **extra}.items()
)
(WT / "artifacts/migration/FINAL_MIGRATION_SHA256SUMS").write_text(
    "".join(lines), encoding="utf-8")

print("old_state:", old_state)
print("new_terminal_state:", man["terminal_state"])
print("n_artifacts:", len(man["artifacts"]))
print("manifest_sha:", sha256_file(man_path))
print("report_sha:", sha256_file(report))
