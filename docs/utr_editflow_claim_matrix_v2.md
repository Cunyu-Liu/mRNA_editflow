# Claim Matrix — utr_editflow_contract_v2

**Contract:** `configs/utr_editflow_contract_v2.yaml` (`utr_editflow_contract_v2`)
**Authority:** Goal document `提示词/mrna 最新构建合同-先做.md` (supreme) → this file (§0.2 #4).

This matrix fixes what the project may claim, the evidence each claim requires, and the claims that are permanently forbidden.

---

## 1. Evidence Grades (Goal §7.1)

| Grade | Description |
|---|---|
| E0 | Engineering — unit tests, shape tests, smoke, synthetic sanity |
| E1 | Internal computational — train/validation, proxy reward, development split |
| E2 | Retrospective measured — held-out measured source/candidate labels |
| E3 | Cross-study / context — study-disjoint or context-disjoint measured evidence |
| E4 | Historically exposed external — independent study but prior label exposure recorded |
| E5 | Untouched external — genuinely unexposed and frozen before access |
| E6 | Prospective experimental — new wet-lab measurement |

**Current contract max expected evidence: E3–E5. GSE246381 fixed at E4. E6 not in scope.**

---

## 2. Allowed Primary Claim (Goal §7.2)

Only after passing the corresponding Gate:

> We introduce a source-conditioned, region-aware and grammar-constrained continuous-time Edit Flow that generates diverse and biologically legal minimal edits for 5′ and 3′ UTRs, and evaluate its controllability and transfer under matched-budget generative and search baselines.

中文边界：
> 我们提出一个以 source 为锚、以合法编辑动作为基本事件、支持变长和多步生成的 UTR Edit Flow，并在公开测量数据和严格计算协议下评估其生成、控制和迁移能力。

---

## 3. Conditional Secondary Claims (Goal §7.3)

Claimable only when evidence supports:

| ID | Claim | Required evidence |
|---|---|---|
| S1 | Continuous-time edit-process modeling outperforms candidate-independent modeling | Held-out generative likelihood, candidate recovery, calibration, multi-seed |
| S2 | Explicit legal action geometry achieves 100% constraint validity | Hard-constraint audit at every step + final sample |
| S3 | Variable-length flow improves infilling/refinement capability | Variable-length infilling/refinement benchmark results |
| S4 | Edit Flow improves candidate/query efficiency vs strong search | Matched-budget Pareto frontier comparison |
| S5 | Foundation-model initialization improves sample efficiency or transfer | From-scratch vs frozen vs adapter vs full-FT comparison |
| S6 | Region-conditioned adapters outperform fully-shared or fully-independent | Region adapter ablation + wrong-region negative control |
| S7 | Uncertainty/abstention reduces false-beneficial selection | ECE, coverage-risk curves, selective prediction |
| S8 | Public UTR intervention data supports generative edit evaluation | Benchmark construction + provenance audit |

---

## 4. Permanently Forbidden Claims (Goal §7.4)

1. Generated candidates improve real therapeutic mRNA efficacy
2. Unmeasured candidates have experimentally validated functional improvement
3. MRL equals protein output
4. TE equals protein output
5. Half-life improvement necessarily improves protein output
6. Predictor high score equals real biological optimality
7. Model trajectory is observed biological or real biochemical trajectory
8. GSE246381 is untouched sealed test
9. Full-length mRNA optimization is complete
10. CDS grammar is verified
11. Mechanism conclusion from attention heatmap alone
12. Same predictor self-guides, self-selects, and self-proves
13. Edit Flow naturally superior to GPT, diffusion, or search without proof
14. "The first Edit Flow for biological sequences"
15. "The first source/template-conditioned Edit Flow"
16. "The first constrained or function-guided Edit Flow"
17. "The first variable-length biological sequence editor"

**Paper wording preference:** "we formulate / we develop / we evaluate" — NOT "the first" unless with search date, database, query, exclusion criteria, and per-field difference table vs nearest prior art.

---

## 5. UTR-only Publishability Judgment (Goal §7.5)

5′UTR and 3′UTR are sufficient for a complete paper, provided:

- True generative tasks (not two single-point prediction tasks)
- Variable-length and multi-step
- 5′/3′ region-aware comparison
- Foundation and task-native baselines
- Matched-budget generator/search
- Cross study/context
- Measured-support and open-support separate tracks
- Uncertainty, failure, and reproducibility
- Complete benchmark/provenance

No new wet-lab does NOT mean no publication value, but limits: unmeasured candidate biological function claims, therapeutic/translational claims, highest-level biological validation strength.

---

## 6. Submission Readiness (Goal §39.5)

At minimum:

- H1–H8 results complete or negative-result boundary clear
- True Flow semantics tested
- 5′/3′ at least independently reported
- Matched-budget baselines complete
- Generation metrics not solely predictor-dependent
- Foundation exposure disclosed
- GSE246381 historical exposure disclosed
- No-wet-lab claim boundary
- 5 seeds/CI
- Artifacts, container, data/model card
- Claim–evidence matrix
- Failure card
- All main figures reconstructable from frozen artifacts

---

## 7. Evidence Status Ledger

| Claim | Status | Evidence pointer |
|---|---|---|
| Primary | PENDING | Populated as Phases C0→PP0 complete |
| S1–S8 | PENDING | Populated as corresponding Gates pass |

*No wet-lab claim will ever be added to this matrix.*
