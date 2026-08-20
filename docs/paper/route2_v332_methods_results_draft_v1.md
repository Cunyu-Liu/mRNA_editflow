# Route 2 V3.3.2 methods/results evidence draft

> Internal evidence-bound draft. Not submission-ready. All claim and evidence
> markers must remain until an accountable human verifies the local/remote
> artifacts and approves the scientific wording. No external literature,
> authorship, funding, ethics, availability or AI-disclosure statement is
> supplied here.

## Scope and current answer

Route 2 asks whether a source-relative Delta critic trained on heterogeneous
public mRNA intervention records can transfer across studies and then serve as a
frozen potential for legal, budget-controlled Edit Flow generation. The present
evidence does not yet establish that full claim: the first mRNABERT critic failed
three-seed readiness, Critic V2 is running, guided XEditFlow remains unauthorized,
and a new unexposed, convertible external Evaluation study is still required.
[claim:C-R2-001] [evidence:E-R2-CONTRACT,E-R2-CRITIC-V1,E-R2-CRITIC-V2-LAUNCH]

## Methods draft

### Development data and protected-outcome boundary

The frozen Development collection contains 126,165 source-candidate records:
89,580 TRAIN, 18,293 VALIDATION and 18,292 withheld TEST records. All work
reported in this packet used TRAIN/VALIDATION or the Development measured
neighborhood; Development TEST and the new final external Evaluation remained
closed. [claim:C-R2-002] [evidence:E-R2-CONTRACT,E-R2-FRESH]

### Independent generation evaluator

Generation methods were compared with a frozen Siamese CNN evaluator trained
independently of the guiding checkpoint on Development TRAIN/VALIDATION. The
evaluator contained 509,845 trainable parameters and completed 8 epochs and
22,400 optimizer steps on GPU. Its role was restricted to Development method
selection; it was not an external biological assay or final Evaluation model.
[claim:C-R2-003] [evidence:E-R2-EVAL-TRAIN,E-R2-EVAL-ADJ]

### Matched generation/search comparison

Seven methods were evaluated on the same 891-source cohort under the `SUB +
STOP` action space: random legal search, greedy search, beam search, a genetic
algorithm, local search, generate-then-rerank and unguided learned Base Flow.
The protocol imposed a maximum of 32 candidates per source, 256 critic forwards
per source for critic-using methods and 320 total forward-equivalents per source.
The independent evaluator was frozen before candidate generation, its checkpoint
was distinct from the guiding checkpoint, and all method selection used
Development evidence. [claim:C-R2-004] [evidence:E-R2-GEN-SUITE,E-R2-GEN-INPUT,E-R2-GEN-SELECT]

The primary selection metric was the source-macro maximum independent-evaluator
uplift over the source. A paired bootstrap used the source as the analysis unit
(`n = 891`), seed 20260816 and 10,000 iterations, all of which produced defined
leader-advantage values. Methods whose leader-advantage interval included zero
were treated as uncertainty-equivalent, with lower mean total forward-equivalents
as the prespecified tiebreak. Measured-neighborhood candidate and top-k recovery
were reported as separate Development diagnostics because generated candidates
outside measured support have unknown outcomes. [claim:C-R2-005]
[evidence:E-R2-GEN-SELECT,E-R2-GEN-INPUT]

### Prospective Critic V2 repair

Critic V2 keeps the frozen mRNABERT encoder and 9,342,914-parameter edit-centered
critic but aligns optimization with task-macro selection through fixed
`task -> study -> source-context-endpoint group -> record` draws and equal
per-task Huber aggregation within each batch. Its prospective screen comprises
the full model, within-source/task candidate permutation, a parameter-matched
source-only control and a source-plus-edit-metadata control without candidate
global mRNABERT representation. The four arms share seed 20260825, 100 epochs,
batch size 16 and the same TRAIN/VALIDATION budget. [claim:C-R2-006]
[evidence:E-R2-CRITIC-V2-PROTOCOL,E-R2-CRITIC-V2-LAUNCH]

### Prospective Critic V2 selection and readiness sequence

Critic V2 may advance only through the frozen sequence of a passing four-arm
control screen, three passing confirmation seeds (20260822, 20260823 and
20260824), one Development TEST run at seed 20260823, an all-126,165-record
refit, and three seed-level matched primary-versus-baseline LOSO aggregations.
The single TEST uses 100 epochs and the final epoch because TRAIN and VALIDATION
are folded together at that stage; TEST metrics are report-only and cannot select
the architecture, loss, seed, epoch, threshold or policy. Each LOSO seed must
contain seven aligned nonempty Development studies and have positive model-minus-
baseline macro Spearman. Guided Development generation can be authorized only
after both `CRITIC_READY_FOR_GUIDANCE` and `FLOW_G0_READY`; neither readiness
state is a biological-success claim. [claim:C-R2-014]
[evidence:E-R2-CONTRACT,E-R2-CRITIC-V2-READINESS]
[evidence:E-R2-CRITIC-V2-LOSO-AGG]

## Results draft

### The independent evaluator narrowly crossed its frozen qualification threshold

The evaluator achieved task-macro Spearman 0.1025655 and task-macro standardized
MAE 1.8078551 across nine tasks. Its Spearman exceeded the frozen exclusive
threshold of 0.1012476 by 0.0013180, with five of nine task correlations positive,
yielding `INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED`. The small margin limits
this result to Development method selection and does not constitute biological
validation. [claim:C-R2-007] [evidence:E-R2-EVAL-TRAIN,E-R2-EVAL-ADJ]

### Genetic search led the independent-evaluator criterion, while unguided Flow led measured recovery

All seven methods completed with hard legality 1.0 and zero edit-budget or
candidate-cap violations. The suite generated 28,512 candidates for six methods;
local search returned 21,027 candidates within the same cap. Thus the protocol
matched the candidate ceiling, but it did not force every method to return the
same realized candidate count. [claim:C-R2-008]
[evidence:E-R2-GEN-SUITE,E-R2-GEN-INPUT]

| Method | Candidates | Unique rate | Hamming diversity | Independent-evaluator max uplift | Candidate recovery | Measured top-k recovery | Mean forward-equivalents |
|---|---:|---:|---:|---:|---:|---:|---:|
| Genetic | 28,512 | 1.0000 | 0.06826 | **1.09782** | 0.05443 | 0.00281 | 231.47 |
| Generate then rerank | 28,512 | 1.0000 | 0.07385 | 1.05760 | 0.06828 | 0.00440 | 266.61 |
| Unguided learned Base Flow | 28,512 | 0.8829 | **0.07657** | 0.90768 | **0.20286** | **0.09797** | 124.45 |
| Random legal | 28,512 | 1.0000 | 0.06529 | 0.88960 | 0.07697 | 0.00477 | **64.00** |
| Local search | 21,027 | 1.0000 | 0.05735 | 0.86814 | 0.00365 | 0.00224 | 200.98 |
| Greedy | 28,512 | 1.0000 | 0.03290 | 0.79786 | 0.06584 | 0.00365 | 266.61 |
| Beam | 28,512 | 1.0000 | 0.04278 | 0.75365 | 0.08025 | 0.00440 | 266.61 |

Values are source-macro Development aggregates unless identified as counts. The
full-precision table and engineering columns are retained in
`route2_v332_generation_baseline_table_v1.csv`. [claim:C-R2-009]
[evidence:E-R2-GEN-INPUT,E-R2-GEN-SELECT,E-R2-FLOW-MATCHED]

Genetic search was the point leader and the only method in the 10,000-iteration
bootstrap uncertainty-equivalent set, so it was frozen as the strongest
Development generation baseline for the independent-evaluator criterion. In
contrast, unguided Base Flow had the highest measured candidate recovery
(0.202862) and measured top-k recovery (0.097973). This endpoint-dependent
ranking is a benchmark result: it shows that model-based uplift and recovery of
the sparse measured neighborhood capture different properties, not that either
method produces a verified biological improvement. [claim:C-R2-010]
[evidence:E-R2-GEN-SELECT]

All six paired comparisons supported the point ordering under the frozen
bootstrap: every 95% leader-advantage interval had a lower bound greater than
zero. Generate-then-rerank was the nearest competitor; genetic's source-macro
advantage was 0.04022872 with a 95% interval of [0.01893446, 0.06168808]. This
interval quantifies Development independent-evaluator separation only and does
not provide external or measured biological validation. The exact six-row table
is retained in `route2_v332_generation_bootstrap_table_v1.csv`.
[claim:C-R2-015] [evidence:E-R2-GEN-SELECT]

### Predictor readiness remains unresolved

The first mRNABERT cohort produced three task-macro Spearman values of 0.116129,
0.116908 and 0.137384. Their margins over the frozen strongest same-information
baseline were -0.015586, -0.014806 and +0.005669; only one of three was positive.
The frozen gate therefore stopped before Development TEST and did not authorize
refit, LOSO or guided XEditFlow. [claim:C-R2-011]
[evidence:E-R2-CONTRACT,E-R2-CRITIC-V1]

Critic V2 is currently a running prospective Development control screen, not a
terminal result. The four arms are registered as RUNNING on GPUs 2, 4, 3 and 5;
no Critic V2 Validation metric, control adjudication or three-seed outcome is
reported in this draft. [claim:C-R2-012]
[evidence:E-R2-CRITIC-V2-PROTOCOL,E-R2-CRITIC-V2-LAUNCH]

## Concrete limitations and reporting gaps

1. The independent evaluator cleared its threshold by only 0.0013180 and shares
   the broader Development evidence domain; it is isolated from the guiding
   checkpoint but is not an external biological confirmation.
2. Open generated support contains candidates with unknown outcomes. Candidate
   recovery is observable, whereas closed measured NDCG is undefined under this
   support mode; unknown candidates were not assigned zero gain.
3. The genetic-versus-Flow ranking changes with the reported endpoint. The paper
   must keep independent-evaluator uplift and measured-neighborhood recovery in
   separate columns.
4. Per-method generation wall time was not persisted for the six search methods:
   all 891 source-level `wall_time_seconds` values per method are null. The
   overall suite wall time was 10,959.64 seconds, and independent scoring times
   were recorded, but file timestamps will not be used to reconstruct missing
   generation times. Unguided Flow separately recorded 341.56 seconds and
   556.49 MiB peak VRAM. Future parallel stages now persist per-method wall time,
   but this instrumentation does not retroactively fill the terminal suite.
   [claim:C-R2-013]
   [evidence:E-R2-GEN-SUITE,E-R2-GEN-INPUT,E-R2-FLOW-MATCHED]
5. Development TEST, TEST-preserving predictor/baseline LOSO, guided XEditFlow
   and a new outcome-unexposed convertible external Evaluation study remain
   incomplete. No final biological or cross-study generation claim is unlocked.

## Allowed and prohibited wording at this stage

Allowed:

- “The independent evaluator qualified for frozen Development method selection.”
- “Genetic search was the strongest Development baseline under the frozen
  independent-evaluator criterion.”
- “Unguided Base Flow recovered more of the Development measured neighborhood.”
- “All seven methods produced legal candidates within the declared caps.”
- “The original mRNABERT critic failed three-seed readiness; Critic V2 is running.”

Prohibited until later gates pass:

- “Genetic search or Base Flow improves mRNA biology.”
- “The critic is ready for guidance.”
- “Guided XEditFlow succeeds.”
- “The generation results are externally validated.”
- “Every method produced exactly 32 candidates per source.”
- “GSE232572 is an unbiased final confirmation” or “E-MTAB-10902 was evaluated.”

## Unresolved items before manuscript integration

- Accountable human verification of every evidence locator and numeric claim.
- Critic V2 control and, conditionally, exact three-seed terminal adjudication.
- The contract-ordered single TEST, all-record refit, predictor/baseline LOSO and
  dual critic/Flow readiness stages if and only if every preceding frozen gate
  passes.
- Guided comparison only after critic and Flow readiness both hold.
- Registration and use of a new, convertible, outcome-unexposed external
  Evaluation study.
- Target venue, reporting guideline, author list, contributions, declarations,
  funding, ethics applicability, data/code availability and AI-use disclosure.
