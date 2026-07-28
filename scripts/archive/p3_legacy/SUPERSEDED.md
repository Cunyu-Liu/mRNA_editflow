# SUPERSEDED - LEGACY RESEARCH CONTRACT (P3 / NMI era)

**Status:** SUPERSEDED_LEGACY
**Superseded by:** public_intervention_contract_v1

**New contract files:**

- configs/public_intervention_contract.yaml
- docs/public_intervention_scientific_question.md
- docs/public_intervention_claim_matrix.md

## Why this was archived

The legacy contract evidence chain required prospective wet-lab protein-output validation, which is no longer executable. All legacy goals tied to prospective wet-lab validation (protein-output AUC primary endpoint, multi-cargo wet-lab, wet-lab unlocking of CDS/3UTR/joint editing, calling model-designed candidates real beneficial mRNA) are withdrawn.

The new project is a machine-learning methods + data-benchmark study on publicly measured mRNA intervention data (mRNA-EditBench + SparseEditFormer), with no new wet-lab claims.

## Rules for this archive

1. These files are retained for historical traceability only.
2. New training code, paper mode, and result-generation code MUST NOT read anything in this archive as a constraint source.
3. Old results in this archive MUST NOT be modified to appear consistent with the new scientific question.
4. Git history is preserved; files were moved with git mv.
5. Enforcement: python scripts/contracts/audit_legacy_references.py --strict must report active paper code references to legacy contract = 0.

## Archival record

- archived_on: 2026-07-28
- pre_archive_commit: 20d33f104d164c783cd2182fcfa8824bfb2f6b28
