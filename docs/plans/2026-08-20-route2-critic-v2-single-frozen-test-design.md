# Route 2 Critic V2 Single Frozen-TEST Design

## Status and boundary

This design is prospective: the Critic V2 control screen is still running, no
Critic V2 three-seed outcome exists, and Development TEST remains unopened. The
implementation created from this design may prepare one config only after the
exact V2 control and three-seed adjudications pass. It will not schedule, train,
evaluate or inspect the real Development TEST.

## Gate binding

The preparer accepts six inputs: the frozen control protocol, the frozen
three-seed protocol, a separate prospective frozen-TEST protocol, the terminal
control adjudication, the terminal three-seed adjudication and one of the exact
confirmation-seed runtime configs. It requires:

- all three protocols to have their prospectively frozen schema and status;
- both protected-outcome fields in all three protocols to remain false;
- the control adjudication to authorize the exact frozen three seeds;
- the three-seed adjudication to authorize one frozen Development TEST;
- the three-seed results to be exactly seeds 20260822/20260823/20260824 and all
  three margins over the frozen strongest baseline to be positive;
- the selected confirmation config to be the full, non-control Critic V2 model,
  on Development TRAIN/VALIDATION, with one of those exact seeds;
- every frozen training-policy field to match both protocols.

The separate frozen-TEST protocol fixes the single TEST seed at `20260823`,
matching the prior frozen route and avoiding post-outcome seed selection. The
output declares
`development_test_outcomes_accessed=true` because executing that config would
open TEST; it keeps Evaluation closed and targets GPU0-5. Checkpoint selection
before TEST remains `BEST_VALIDATION`. For the executable TEST run, the trainer
folds Development TRAIN and VALIDATION into one training set, so the protocol
prospectively fixes 100 epochs and selects `FINAL_EPOCH`; there is no Validation
loader at this stage. This is a pre-outcome execution rule, not TEST-based epoch
selection. TEST is never used for epoch, checkpoint, model or policy selection.
The CLI refuses an existing config or run directory.

## Alternatives rejected

Extending the historical V1 preparer to accept both schemas would create a
compatibility path that this project does not need and retain its loss-only
validation and its policy-changing `FINAL_EPOCH` override. Copying a seed config and editing stage fields by hand would not
prove the control gate, three-seed gate or policy identity. A dedicated V2
preparer keeps the historical negative cohort immutable and makes the only TEST
entrypoint explicit.

## Verification

Focused tests use synthetic PASS/NO-GO payloads only. They verify the single
seed and stage, exact policy replay, TEST/Evaluation flags, GPU range, rejection
of either failed gate, rejection of seed/policy drift and refusal to overwrite
targets. No real adjudication, TEST record or Evaluation artifact is read.
