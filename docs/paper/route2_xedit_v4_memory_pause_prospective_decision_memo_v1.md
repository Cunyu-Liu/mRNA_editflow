# XEditCritic V4 memory-pause prospective decision memo

> **Status: FROZEN USER DECISION — CANDIDATE A SELECTED PROSPECTIVELY.**
> The user selected Candidate A on 2026-08-25 before attempt 5, any V4
> parameter update, or any V4 Validation performance read. This amendment
> authorizes attempt 5 and conditionally authorizes the already frozen V4
> screens only after both attempt-5 preflights pass. It does not reclassify
> attempt 4 and does not expose a Validation, Development TEST, or new
> Evaluation performance outcome.

## Focal decision

After the clean attempt-4 preflight measured a 170,481,957-parameter Critic at
6.63995 GiB peak allocated memory for physical batch 32, should Route 2 remove
the non-scientific 20-GiB lower memory floor while retaining the complete V4
method, or instead change the training method to consume more memory?

The user is the decision owner and selected Candidate A. Attempt 4 remains
`XEDITCRITIC_V4_PREFLIGHT_PAUSE`; the amended exact-HEAD attempt 5 must establish
a new Critic PASS and SetFlow PASS before either screen is launched.

## Located evidence

- The formal dataset-bound Critic has 170,481,957 trainable parameters, within
  the frozen 165–175M design target and 120–180M admissible range.
- All declared capacity is present: trainable mRNABERT blocks 6–11, twelve
  edit blocks, local source/candidate cross-attention, semantic top-two experts,
  the raw antisymmetric residual, global residual, and counted readout.
- Batch 4/8/16/32 peak allocations were 2.82297/2.82288/3.87276/6.63995 GiB.
  Every candidate completed BF16 forward, backward, gradient checks and
  optimizer-state materialization without OOM or CPU fallback.
- The low peak is explained by frozen bottom-six caching, local/ragged
  attention, BF16, SDPA and activation checkpointing. It is not evidence that
  the trainable model is small.
- The frozen protocol itself states that peak memory is an audited preflight
  property rather than a success metric. The scientific success criteria are
  ranking, baseline margins, controls, MAE, task breadth and uncertainty.
- SetFlow has independently passed its formal preflight at 100,099,998
  trainable parameters.
- Protected reads remain zero and no V4 Validation performance metric has been
  read. The decision therefore cannot be influenced by V4 performance.

## Candidate A — V4.0.1 resource-only amendment (recommended)

**Idea.** Remove only the 20-GiB lower memory floor. Keep the 35-GiB upper
bound and every scientific/model/training requirement unchanged.

**Prospective specification if selected.**

- Critic architecture and formal trainable range remain unchanged.
- Physical/effective batch remains 32; passes remain 8; updates remain 2,802
  per pass and 22,416 total.
- TRAIN-only capped-sqrt task/source-group sampler, record repeat cap, losses,
  learning rates, seeds, controls, ablations and gates remain unchanged.
- BF16, activation checkpointing, SDPA, no CPU fallback and no artificial
  padding remain unchanged.
- Memory eligibility becomes: physical batch at least 4, complete batch-32
  forward/backward/optimizer-state materialization, finite numerics, and peak
  allocated memory no greater than 35 GiB. There is no lower occupancy gate.
- Attempt 4 remains `PAUSE`; it is not retroactively relabeled. A new exact-HEAD
  `preflight_attempt_5/` is required, with attempt 1–4 read-only.
- The exact-head SetFlow preflight is rerun only as a short execution check in
  the paired package; no SetFlow training or outcome-free generation is rerun.
- Only an attempt-5 dual PASS authorizes the already preregistered V4 screens.
- The preflight launcher has no fixed 37,000/20,000-MiB free-memory floor.
  Selected GPUs must be visible physical indices 0–5; the actual CUDA/BF16
  preflight determines whether execution fits.
- After attempt 5, screen jobs are assigned to any GPU 0–5 with current free
  memory at least the component's measured attempt-5 peak plus 2 GiB. This is a
  workload-derived sufficiency check, not a fixed occupancy target. Arms,
  seeds, budgets and selection rules do not change with GPU assignment.

**Assumption.** Parameter count and verified model paths measure capacity more
directly than minimum allocated memory, while an upper memory bound remains an
important feasibility control.

**Prediction.** Attempt 5 will select physical batch 32 at approximately the
already observed peak, after which the current V4 screen can test the actual
Spearman and control hypotheses without a method change.

**Disconfirming evidence.** Attempt 5 would fail if the exact amended HEAD no
longer reproduces capacity, cache alignment, finite gradients, batch-32 CUDA
execution, protected-read closure, or the 35-GiB ceiling. It does not pass merely
because the lower floor was removed.

**Adversarial review.** This amendment follows a resource observation. It is
therefore recorded explicitly and cannot be presented as originally
preregistered. Its risk is limited because the changed quantity is not an
outcome or selection metric, and the complete architecture/training/scientific
gates remain intact. It supplies no evidence that Spearman will improve.

## Candidate B — high-ranking-batch V4.1 (not recommended without a new protocol)

**Idea.** Raise the physical/effective task batch toward 128 so the soft-rank
and pairwise objectives see more within-task comparisons and use more real
activation memory.

**Potential mechanism.** A batch of 32 exposes at most 496 unordered pairs;
128 exposes at most 8,128. More cross-source-group comparisons could reduce the
variance of the soft-Spearman surrogate. This is a performance hypothesis, not
a finding or guarantee.

**Outcome-free feasibility evidence.** The frozen TRAIN projection contains
89,580 records across seven TRAIN-supported tasks, with task sizes
204, 893, 1,308, 2,443, 3,318, 25,710 and 55,704. At the current four-repeat
cap, total presentation capacity is 358,320 records per pass. Keeping 2,802
updates at batch 128 would require 358,656 presentations, exceeding the absolute
cap by 336 even before enforcing capped-sqrt task balance. Thus the previously
suggested combination “batch 128 + 2,802 updates + repeat cap four” is
mathematically infeasible.

Reducing to at most 2,799 updates would satisfy only the global arithmetic cap;
it would still drive nearly every record to four presentations and largely
erase the intended capped-sqrt balancing behavior. The smallest task has only
204 records and permits six complete batch-128 draws before exceeding its
four-repeat capacity. Consequently a scientifically honest batch-128 plan must
also change at least one of the sampler, update budget, task-homogeneity rule,
or repeat cap.

**Prediction.** Linear extrapolation from batches 16 and 32 suggests that batch
128 might allocate roughly 23 GiB, but ragged geometry can be nonlinear and a
new preflight would be authoritative. Higher occupancy is not evidence of
higher Spearman.

**Disconfirming evidence.** B is uninformative if increased batch size merely
repeats records, reduces task balance, changes effective optimization budget,
or fails to improve the preregistered controls and Spearman margins.

**Adversarial review.** B confounds a resource preference with a material
training-method change. It is compute-heavy across the full model, baseline,
four candidate controls and two mechanism ablations; it also cannot preserve
all current sampler invariants. B therefore requires a separately discussed
V4.1 protocol rather than an implementation patch.

## Candidate C — wider or multiscale context (deferred)

**Idea.** Expand radius-32 cross-attention to multiscale local and sparse global
context. This may add biologically meaningful context and activation memory.

**Uncertainty and adversarial review.** mRNABERT per-token states already carry
chunk context, so wider explicit cross-attention may be redundant or dilute the
edit-local signal. Current V4 has not received one performance screen, and no
evidence locates the bottleneck in context radius. C should be revisited only
after a legitimate V4 screen identifies a mechanism failure; it should not be
used to manufacture a memory target.

## Rejected non-method changes

Artificial tensors, duplicated tokens, unused padding, redundant optimizer
state, CPU fallback, disabling efficient attention solely to raise memory, or
rerunning frozen bottom-six blocks solely for occupancy are rejected. They
would change a resource number without increasing scientific capacity.

## Recommendation and decision log

The technical recommendation is **Candidate A**. It preserves the actual V4
scientific experiment and allows the strict screen to determine whether the
170.48M edit-local semantic model improves Spearman. Candidate B is not a clean
way to preserve the frozen training design and must not be implemented as a
small parameter tweak.

- Decision owner: user
- Decision: **CANDIDATE A SELECTED AT 2026-08-25T20:16:23+08:00**
- Accepted candidate: **V4.0.1 RESOURCE-ONLY AMENDMENT**
- Authorization to edit the frozen protocol: **YES**
- Authorization for attempt 5: **YES**
- Authorization for V4 screen: **CONDITIONAL ON ATTEMPT-5 CRITIC AND SETFLOW PASS**
- Rejected for this execution: **Candidate B** because it changes sampler and
  optimization invariants; **Candidate C** because no screen evidence yet
  locates the bottleneck in explicit context radius
- Development TEST outcome reads: 0
- New final Evaluation outcome reads: 0
