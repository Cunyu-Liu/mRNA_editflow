# mRNA-EditFlow

**Source-conditioned, region-aware, grammar-constrained continuous-time Edit Flow for 5′UTR / 3′UTR minimal editing.**

> **Research status — 2026-08-01**
>
> Active contract: `utr_editflow_contract_v2` (FROZEN). Supersedes `public_intervention_contract_v1` (archived under `archive/legacy_predictor_first_v1/contracts_v1/`).
>
> All scientific hypotheses (H1–H8) are **PENDING** until they pass their pre-registered gates. Current computational results are E0 (engineering) or E1 (internal computational proxy) unless explicitly labeled otherwise. **No wet-lab evidence is in scope** (E6 not in scope).

---

## 1. Scope and Method

This project studies whether a **source-conditioned, region-aware, grammar-constrained continuous-time Edit Flow** can learn transferable legal edit-trajectory distributions over public measured UTR data and generate diverse, sparse, variable-length, controllable 5′UTR / 3′UTR candidates.

**In scope** (`configs/utr_editflow_contract_v2.yaml` §1):

- 5′UTR and 3′UTR (UTR-only first phase)
- Edit Flow as the **primary method** (not optional, not a fallback)
- Continuous-time edit rates λ_ins, λ_sub, λ_del, λ_stop + token/action distributions
- INS/DEL/SUB/STOP as first-class actions; variable length and multi-step trajectories
- Hard action masks (region boundary, anchors, alphabet, length, edit budget, forbidden identity)
- Source conditioning `p(candidate, trajectory | source, region, context, target, constraints)`
- Region-aware rate fields (shared trunk + 5′UTR adapter + 3′UTR adapter)
- Conditional generation (endpoint, assay, context, target direction/quantile, budget, motif preservation)
- Matched-budget Pareto comparison vs AR / masked-diffusion / generic Edit Flow / scorer+search
- Foundation-model reuse-first (frozen → adapter → LoRA → partial unfreeze → full FT only if justified)

**Out of scope** (`configs/utr_editflow_contract_v2.yaml` §4.2):

- CDS synonymous generation
- Protein-conditioned codon flow
- Full-length mRNA joint optimization
- Full-length therapeutic mRNA generation
- Cross-region full-transcript synergy
- New wet-lab experiments (E6)
- Reinforcement learning as a primary method (RL is not the central methodological story)

## 2. Authority Hierarchy

The project is controlled by a **single active contract**. If a lower-level file conflicts with a higher-level one, fail closed.

1. Goal document — `提示词/mrna 最新构建合同-先做.md` (scientific thought, immutable boundaries)
2. `configs/utr_editflow_contract_v2.yaml` — structured executable contract
3. `docs/utr_editflow_scientific_question_v2.md` — scientific question + H1–H8 hypotheses
4. `docs/utr_editflow_claim_matrix_v2.md` — allowed / conditional / forbidden claims
5. `docs/execution/task_registry_v2.yaml` — phase task registry
6. Per-experiment frozen config / manifest / run artifacts
7. `docs/decision_log.md` — amendment history
8. `docs/contracts/v2_contract_conflict_matrix.md` — v1→v2 conflict resolution record

Older contracts are archived:

- `archive/legacy_predictor_first_v1/contracts_v1/` — `public_intervention_contract_v1` (predictor-first, SparseEditFormer, Flow-optional, CDS, GSE246381 sealed). **SUPERSEDED. Historical reference only.**
- `configs/archive/p3_legacy/`, `docs/archive/p3_legacy/` — P3/NMI legacy. **SUPERSEDED.**

## 3. Phase Structure

Forward-only state machine. Upstream Gate not passed → may continue non-conflicting parallel prep, but no downstream formal scientific conclusions.

```text
C0  合同与现实对齐                          [DONE]
D0  科学问题驱动的数据发现                  [D0-01..04 DONE, D0-05 PENDING]
D1  数据资格、重建与暴露审计                [PENDING]
B0  生成式 UTR benchmark 与 splits          [PENDING]
FM0 Foundation model 接入                   [PENDING]  (FM0 → MK0 → EF0 mandatory gate)
MK0 UTR Edit Flow 数学内核冻结              [PENDING]  (gate)
EF0 True UTR Edit Flow 工程实现             [PENDING]
GP0 Generative prior GPU 训练               [PENDING]  (GPU-only)
FC0 Functional conditioning / critic 系统   [PENDING]
ME0 Measured-support 与 candidate freeze    [PENDING]  (ME0 → MB0-Freeze → MB0-Run mandatory)
MB0 Matched-budget 正式比较                 [PENDING]  (gate)
TR0 5′UTR → 3′UTR 迁移                       [PENDING]
ER0 Robustness、failure 与机制分析          [PENDING]
PP0 论文、复现与发布                        [PENDING]
FL0 未来 full-length 决策 (not in current scope)
```

Full task list: `docs/execution/task_registry_v2.yaml`.

## 4. Data Status

| Dataset | Provider | Files | Status |
|---|---|---:|---|
| GSE114002 / GSE145046 / GSE149487 / GSE173083 / GSE200304 / GSE207584 / GSE217518 / GSE246381 | GEO | 75 | COMPLETE (sha256 verified) |
| ENCSR854RUF raw reads | ENCODE | 62 | COMPLETE (provider_md5 + file_size verified, ~357 GB at `data/p0/ENCSR854RUF/reconstructed/`) |
| ENCSR854RUF processed (MPRAu Supp Table1) | PMC | 1 | COMPLETE (sha256 verified) |

`GSE246381` data integrity is verified. The current committed exposure ledger contains
1,184 paired 5′UTR records classified as `D_C` / `E2` with
`exposure_status=unexposed`; the D1 dataset-coverage and edit-script audits pass.
The earlier description that treated GSE246381 as a data-quality or data-validity
problem is withdrawn.

The active contract's historical-exposure/E4 field is a separate provenance and
admission-policy field; it must not be conflated with whether the dataset itself is
valid. Until an authorized contract amendment changes that policy field, downstream
code must continue to obey the contract's explicit label rules and record both fields
separately.

`ENCSR854RUF` raw reads provide **unlabeled/observational** pretraining only (D_A). They cannot provide wt–mutant causal labels, multi-step real trajectories, prospective improvement, or final independent oracle evidence.

Download verification: `docs/data/download_verification.md` (verdict: COMPLETE).

## 5. Hard Constraints

- Edit Flow is the primary method. **Do not rename a GPT generator as Edit Flow.**
- UTR-only first phase. CDS / full-length / cross-region full-transcript synergy are out of scope.
- No new wet-lab experiments (E6 not in scope).
- GPU-only training. CPU fallback for formal neural training is forbidden.
- Foundation and effect predictor are **support systems**, not the Edit Flow. Teacher / guidance, selection, and final evaluator roles must be separated.
- 100% hard-constrained validity at every generation step and final sample. Soft penalty / post-repair / illegal-sample deletion cannot substitute constructive legal action space.
- Forward-only phase state machine. No gate lowering post-hoc.
- Failure evidence must not be deleted. Negative results and failed seeds remain visible.
- No "the first …" claims without search date, database, query, exclusion criteria, and per-field difference table vs nearest prior art.

## 6. Forbidden Claims (excerpt)

Full list: `docs/utr_editflow_claim_matrix_v2.md` §4.

- Generated candidates improve real therapeutic mRNA efficacy
- Unmeasured candidates have experimentally validated improvement
- MRL / TE / half-life equals or necessarily improves protein output
- Model trajectory is observed biological trajectory
- GSE246381 is untouched sealed test
- Full-length mRNA optimization is complete; CDS grammar is verified
- Edit Flow naturally superior to GPT/diffusion/search without proof
- "The first Edit Flow for biological sequences" / "source-conditioned" / "constrained" / "variable-length"

## 7. Installation

```bash
git clone https://github.com/Cunyu-Liu/mRNA_editflow.git
cd mRNA_editflow

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Recommended optional tools: MMseqs2 (clustering), ViennaRNA / RNAfold (structural features), CUDA-enabled PyTorch (training).

## 8. Smoke Workflow

```bash
# Verify v2 contract
python -m pytest tests/test_utr_editflow_contract_v2.py -q

# Audit active-contract consistency
python scripts/contracts/audit_active_contracts.py --strict

# Validate task registry
python scripts/execution/validate_registry.py docs/execution/task_registry_v2.yaml

# Run unit tests
pytest -q
```

These commands validate engineering plumbing. They are **not** paper-grade training or biological evidence.

## 9. Reproducibility Contract

Paper-mode runs must record (in `training_manifest_required_fields` of the v2 contract):

- `goal_contract.id` + `sha256`
- `scientific_question_id`
- `phase_id`, `task_id`, `git_commit`
- `data_manifest_sha256`, `split_manifest_sha256`
- `foundation_checkpoint` + `sha256` (or `none`)
- `exposure_ledger_version`

Missing these fields → development smoke only, **not paper evidence**.

## 10. Governance

The active contract is `utr_editflow_contract_v2`. Changes follow the amendment rules of the contract and are recorded in `docs/decision_log.md`.

Forbidden practices:

```text
silently rewriting the primary task
changing thresholds after seeing results
moving failed families out of the test set
dropping failed seeds
weakening strong baselines to preserve a story
claiming measured improvement from an internal predictor
overwriting frozen artifacts without an amendment
renaming a GPT generator as Edit Flow
lowering a gate post-hoc
deleting failure evidence
```

The project route is selected by evidence, not fixed in advance as an RL, full-transcript, or synergy paper.

## 11. Project Summary

mRNA-EditFlow tests the following evidence chain (under `utr_editflow_contract_v2`):

```text
Public UTR data has measurable source→candidate edit structure
        ↓
continuous-time legal edit-rate field is a sound generative model for it
        ↓
source / region / target conditioning yields controllable distributions
        ↓
under matched budget, Edit Flow is not fully dominated by strong search
        ↓
results transfer across sources / studies / regions
        ↓
(no wet-lab claims are made — E6 out of scope)
```

The primary intended contribution is:

> A source-conditioned, region-aware, grammar-constrained continuous-time Edit Flow that generates diverse, biologically legal minimal edits for 5′ and 3′ UTRs, evaluated under matched-budget generative and search baselines on public measured data.
