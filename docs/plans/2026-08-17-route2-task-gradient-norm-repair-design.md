# Route 2 task-gradient norm repair design

Status: frozen Development-only exploratory design after the method-repair v2 screen NO-GO.

## Evidence and decision

The six-arm screen selected global-scaled at 9-task macro Spearman 0.131714, below the legacy 0.153287 reference and with only 3/9 task improvements. Edit-scaled reached 0.102854. Its CUDA TRAIN diagnostic found 11/21 negative shared-gradient pairs, but conflict did not cleanly separate improved from regressed tasks. The clearest actionable signal was gradient-norm imbalance: polyA was 0.906 versus 0.026–0.083 for the other tasks. A sampler audit then showed that 2,761/2,765 TRAIN batches contain only one task. Batch-level PCGrad would therefore be almost inactive; changing to a multi-task sampler would also change repeats, optimizer steps, or forward-equivalents.

## Selected method

Add `TRAIN_TASK_GRADIENT_NORM_CALIBRATED` as an explicit training-only optimization mode for the existing edit-scaled model. Before the first optimizer step, use the initialized model on at most 16 evenly spaced, complete source-group batches per TRAIN-supported task. Accumulate each task's mean gradient over the shared effect parameters, require finite non-zero norms, and freeze a loss multiplier equal to the geometric mean of the seven norms divided by that task's norm. Calibration must run on the declared CUDA device, read TRAIN only, and perform zero parameter updates.

Training then keeps the original sampler, records, forward-equivalents, AdamW, seed, epochs, architecture, parameter count, target scaling and checkpoint rule. A single-task batch uses its frozen multiplier. The four observed multi-task batches are split by task for loss construction and their scaled task losses are averaged; source groups remain intact. Task keys control training loss only and add no inference parameter or task-specific prediction head. The summary records calibration batches, record coverage, raw norms, multipliers and CUDA proof.

## Experimental gate

Run one seed-20260821 edit-scaled arm on a currently free GPU. It may advance only if it exceeds global-scaled and the legacy 0.153287 macro, has positive task median, improves at least 5/9 tasks over global-raw, and common train-robust macro MAE is not worse than 1.822073. Failure stops this hypothesis. Success authorizes matched source-only and exact-source permutation controls using the same optimization mode before any fresh-seed confirmation. Development TEST and Evaluation remain unopened throughout this screen.

