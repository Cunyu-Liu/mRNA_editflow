# SetFlow V4 validation-only recovery design

## Decision

Reuse the terminal `v4_full` and `v4_single_mode` pass 4/6/8/10 checkpoints. Do
not retrain either arm. Run the unchanged frozen Validation generation and gate
thresholds into a new output directory, while retaining the original eight
technical failures and original `XEDITSETFLOW_V4_SCREEN_NO_GO` artifact.

## Root cause addressed

Training was launched from Git HEAD `edad89392077a0cf56e84dfcf94335606dd2b05a`.
The checkpoint validator was added later, so it ran from another HEAD. The
validator reused the training-launch authorization check and incorrectly
compared the original training authorization with the validator's current HEAD.
All eight validations therefore stopped before loading a checkpoint or
producing performance metrics.

## Provenance model

The training authorization remains bound to the Git HEAD recorded in both
`training_config.json` and `training_attempt.json`. The validation result records
that training HEAD and its own validation HEAD as separate fields. A mismatch
between the two training provenance artifacts still fails closed; a later
validation HEAD is allowed and made explicit.

## Execution and outputs

The recovery launcher creates a new runtime config that changes only validation
and gate output paths. It schedules all eight existing checkpoints on the
currently available physical GPUs 0-4, leaving GPU5 to the active Critic job.
No optimizer, backward pass, parameter update, Development TEST outcome, or new
final Evaluation outcome is used. The original failed validation directory and
gate are never overwritten.

## Verification

Focused unit tests cover training provenance agreement, rejection of provenance
conflicts, preservation of the validation-only path, complete checkpoint
assignment, and explicit runtime identity. The A100 launch requires a clean,
exact recovery HEAD and the existing terminal training/checkpoint package.
