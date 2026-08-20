# Route 2 Critic V2 Guidance Readiness Design

## Boundary

This protocol and implementation are frozen before Critic V2 three-seed, TEST,
refit and LOSO outcomes, without reading control outcomes. They do not build a
real packet now, read protected artifacts, or run guided generation. They define
the only V2 path that may later issue `CRITIC_READY_FOR_GUIDANCE`.

## Evidence chain

The readiness packet binds the exact prospective V2 control, three-seed, single
TEST, all-Development refit, primary LOSO and matched-baseline LOSO protocols. It
contains their terminal adjudications/summaries, the three seed-level LOSO
aggregations, the final refit checkpoint path, frozen guidance reward policy,
online mRNABERT validation, and Base Flow training/validation evidence.

The upstream LOSO aggregator already rejects invalid primary/baseline training
provenance and undefined study metrics. Readiness therefore validates each
terminal aggregate rather than redundantly re-reading 42 training summaries. It
requires exact seeds 20260822/20260823/20260824, seven aligned nonempty studies,
GSE256185 retained as the zero-record Development study, complete GPU provenance,
preserved Development TEST, no Evaluation studies, and positive model-minus-
baseline macro Spearman in every seed.

The single TEST must be complete and free of TEST-driven structure/loss/seed/
epoch/threshold selection, but its metric value is report-only and is not a
readiness threshold. The all-126,165 refit must use the prospectively fixed final
epoch and have a real CUDA parameter update. The final checkpoint must exist.

## Dual readiness

`CRITIC_READY_FOR_GUIDANCE` additionally requires the exact V2 control and
three-seed PASS states, frozen reward/calibration policy, online frozen mRNABERT
alignment, and zero Evaluation use. `FLOW_G0_READY` independently requires a
learned GPU update, checkpoint/device provenance, 100% legality, zero budget and
replay failures, distinguishable terminal causes and the small-graph reference
pass. Guided Development generation is allowed only when both are true. None of
these statuses is a biological-success claim.

## Alternatives rejected

The historical readiness builder/adjudicator uses the old signal-control schema
and accepts any three distinct LOSO seeds. Adding a V2 branch would preserve a
mixed-schema path at the final scientific gate. The V2 packet and adjudicator are
therefore separate.

## Verification

Synthetic tests cover a complete PASS packet, one nonpositive LOSO seed, TEST
metric invariance, upstream/protocol drift, missing checkpoints, reward/encoder
failure, Flow failure, Evaluation contamination and non-overwriting output. No
real outcome is read.
