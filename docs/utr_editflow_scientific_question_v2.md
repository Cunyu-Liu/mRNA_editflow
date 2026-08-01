# Scientific Question — utr_editflow_contract_v2

**Contract:** `configs/utr_editflow_contract_v2.yaml` (`utr_editflow_contract_v2`)
**Status:** FROZEN. Supersedes `public_intervention_contract_v1` (archived under `archive/legacy_predictor_first_v1/contracts_v1/`).
**Authority:** Goal document `提示词/mrna 最新构建合同-先做.md` (supreme) → this file (§0.2 #3).

---

## 1. Primary Research Question (Goal §2.1)

> **给定一个既有 UTR source、region、assay/context、功能 endpoint 和目标条件，source-conditioned、region-aware、grammar-constrained continuous-time mRNA-EditFlow 能否学习可迁移的合法编辑轨迹分布，并生成多样、稀疏、变长且可控的 5′UTR/3′UTR 候选？**

This question studies not "can we score a finished candidate" but:

```text
A high-function UTR candidate
should be modeled as
a generative result evolving from a source along legal edit actions
```

## 2. Comparative Method Question (Goal §2.2)

> **在匹配训练数据、backbone、可训练参数、GPU 预算、candidate budget、oracle-query budget 和约束条件下，mRNA-EditFlow 能否相对 autoregressive generation、masked/discrete diffusion、generic Edit Flow、direct scorer + search 获得更好的"功能控制—合法性—多样性—编辑成本—生成效率"Pareto frontier？**

## 3. Why This Question Has Value (Goal §2.3)

- vs absolute-property predictor: studies source→candidate edit process, not just endpoint score
- vs GPT/autoregressive: preserves source identity and local auditable edits
- vs search: learns reusable amortized proposal distribution
- vs masked/diffusion: INS/DEL/SUB/STOP are explicit continuous-time events, not fixed-length token repair

## 4. Concept Boundaries (Goal §2.4)

"Edit trajectory" in this project = **latent algorithmic edit trajectory** (model's latent variable and algorithmic generation path). It is NOT:

- biological RNA editing biochemical process
- evolutionary history
- experimentally stepped edit trajectory
- causally measured path

If public data only has source and final candidate, the intermediate path can only be called `latent algorithmic edit trajectory`, not `observed biological trajectory`.

## 5. Maximum Scientific Uncertainty (Goal §2.5)

1. Whether public data contains enough measured multi-edit, indel, variable-length, and same-source multi-candidate landscapes
2. Whether true continuous-time Edit Flow outperforms strong scorer/search/AR/diffusion under matched budget
3. Whether foundation model, library proposal bias, and critic mask the true Flow contribution
4. Whether 5′UTR and 3′UTR share strong enough regularities or need region-specific adapters
5. How strong credible computational evidence can be without new wet-lab experiments

These are uncertain, falsifiable questions. The contract does not pre-write answers.

---

## 6. Falsifiable Hypotheses (Goal §3)

### H1: Edit-process modeling
Explicit continuous-time edit-rate field outperforms candidate-only absolute model, source/candidate subtraction, Siamese difference, autoregressive action model, masked/discrete diffusion, and generic unconstrained Edit Flow. Evidence must include held-out generative likelihood/transition reconstruction, candidate recovery, calibration, and multi-seed statistics.

### H2: Edit Flow architecture irreplacability
Each component must produce measurable contribution: source conditioning, continuous time, insertion rate, deletion rate, substitution rate, STOP, variable-length state, multi-step trajectory, region-conditioned rate field, legal action mask, edit-budget state, target property/context condition. If only iterative greedy/top-k substitution is implemented, it is NOT a complete Edit Flow.

### H3: Hard-constrained validity
At all generation steps and final samples: invalid nucleotide=0, forbidden-position edit=0, anchor violation=0, budget violation=0, length-bound violation=0, identity edit counted as edit=0. Soft penalty, post-generation repair, or deleting illegal samples cannot substitute constructive legal action space.

### H4: Conditional controllability
Changing region/assay/context/endpoint/direction produces interpretable and reproducible distribution changes. Must evaluate: target-direction success, condition consistency, condition sensitivity, condition permutation negative control, target strength monotonicity, identical-condition reproducibility, diversity under fixed condition.

### H5: Generative advantage over search
Under matched candidate count, oracle-query count, wall-clock, GPU-hours, and constraints, Edit Flow must form a Pareto frontier not fully dominated by strong search on: high-effect measured-candidate recovery, independent-critic score, diversity, edit cost, inference latency, oracle-query efficiency. Must compare random legal, greedy, beam, best-of-N, simulated annealing, direct scorer exhaustive ranking.

### H6: Cross-source and cross-study transfer
Transfer under source-disjoint, gene-disjoint, study-disjoint, context-disjoint, and exposure-aware external evaluation. Cannot report only random pair split.

### H7: Foundation-model value
Foundation model should improve representation, sample efficiency, or cross-study generalization, validated through: small from-scratch control vs frozen foundation vs adapter/LoRA vs partial/full fine-tune.

### H8: 5′UTR and 3′UTR unification and difference
Two regions share: source-conditioned edit process, INS/DEL/SUB/STOP semantics, continuous-time parameterization, hard constraint interface, generation and evaluation protocol. Two regions keep independent: endpoint heads, assay/context metadata, motif and anchor rules, length priors, region-specific rate-field adapters, data normalization. Must NOT merge 5′UTR MRL, 3′UTR abundance, half-life, or other endpoints into one unified expression label.

---

## 7. Architecture Advantages That Must Be Fully Exploited (Goal §5)

1. **Continuous-time edit process** — non-negative event rates λ_ins, λ_sub, λ_del, λ_stop; retain t; specify Gillespie/tau-leaping/discrete approximation, error control, multi-event conflict, rate clipping, step-count/quality relationship
2. **Source conditioning** — p(candidate, trajectory | source, region, context, target, constraints); report source preservation rate; current state x_t must be updated after each edit
3. **Variable-length first-class support** — INS/DEL are first-class in training, sampling, evaluation; not data augmentation only, not unit-test only, not post-generation splicing
4. **Multi-step trajectories** — support edit budget k∈{1,3,5}; evaluate order, STOP calibration, cycling, reverse, repeated position, budget utilization, per-step legality, real edit distance
5. **Hard action masks** — region boundary, protected positions, anchor motifs, allowed alphabet, max/min length, edit budget, forbidden identity, optional motif-preservation, source-relative state tracking; act before rate normalization
6. **Region-aware rate fields** — shared trunk + 5′UTR adapter + 3′UTR adapter; compare fully-shared, shared+adapter, independent, wrong-region negative control
7. **Conditional generation** — endpoint, assay, context, target direction, target quantile, max edit budget, length target, must-preserve motif; report train vs sample condition consistency
8. **Diversity and amortization** — multiple candidates per source/condition; report unique rate, pairwise edit distance, motif/structure diversity, mode collapse, duplicate rate, candidates/sec, amortized cost

---

## 8. System Architecture (Goal §6)

```text
Layer A — Foundation representation / sequence prior
    mRNABERT / UTR-LM / 3UTRBERT / Orthrus / alternatives

Layer B — Experimental effect system
    paired-delta model, endpoint-specific heads, uncertainty, independent critic(s)

Layer C — mRNA-EditFlow
    continuous-time legal edit rate field, source/region/target conditioning,
    variable-length multi-step sampling

Layer D — Evaluation and selection
    measured candidate recovery, independent critic, external retrospective,
    matched-budget baselines, calibration, failure analysis
```

Foundation model is NOT an Edit Flow replacement. Effect predictor is NOT Edit Flow. At least separate: Teacher/guidance model, Selection model, Final evaluator. When data is limited, report shared data/weights/features and potential correlation errors; do not use "independent".
