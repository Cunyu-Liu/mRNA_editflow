# Route 2 Critic V2 TEST-Preserving LOSO Design

## Boundary and order

This design is prospective and does not authorize LOSO now. The V3.3.2 order is
single TEST, all-Development refit, then 7 Development studies × 3 seeds of
TEST-preserving mRNABERT LOSO. The preparer therefore requires an exact terminal
Critic V2 refit config and summary before it can write LOSO configs. It does not
read the real TEST summary, LOSO outcome or Evaluation in this implementation
task and does not schedule training.

## Frozen cohort

The cohort is the Cartesian product of seven nonempty Development studies and
seeds 20260822/20260823/20260824, for 21 runs. The shared assignment schedule
places study-major, seed-minor jobs round-robin on GPU0-5. Every fold uses only
the original Development TRAIN and VALIDATION partitions. The held-out study is
the Validation assessment fold; connected source components crossing from it
into other studies are excluded from training. The 18,292-row Development TEST
partition is withheld from every LOSO run.

Every fold reuses the exact Critic V2 model, Huber loss, task→study→source-group
fixed-draw sampling, task-macro aggregation, robust target scaling and 100-epoch
budget. It uses `FINAL_EPOCH`, not `BEST_VALIDATION`, because the held-out study
is the LOSO assessment fold and must not select its own checkpoint.

## Gate and output

The V2-only preparer validates the prospective refit and LOSO protocols, the
exact refit config, and a terminal refit summary showing all 126,165 Development
records folded into training, CUDA parameter updates, the fixed final epoch and
zero Evaluation reads. It does not inspect refit or TEST metric values.

It emits exactly 21 unique configs under fixed `/mnt` roots. Each config resets
Development TEST access to false for the LOSO run, records that the earlier
single TEST occurred but its metrics were not used for LOSO selection, and keeps
Evaluation and guidance closed. The CLI refuses an existing config root or any
existing run target.

## Alternative rejected

The historical preparer is bound to the old V1 three-seed status and can create
LOSO before TEST/refit. Adding Critic V2 branches to it would create an unused
compatibility path. A separate V2 preparer makes the frozen order explicit.

## Verification

Synthetic tests cover the exact 21-study/seed/GPU mapping, refit-order gate,
policy and record-scope drift, TEST/Evaluation closure and overwrite refusal.
No real protected artifact is read.
