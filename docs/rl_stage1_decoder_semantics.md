# RL Stage 1: decoder semantics

## Old behavior

- Proposal-ranker training used `log(lambda_op * p_token)`, while decoder and proposal audit ranked raw intensities.
- `sample_mrna(model=..., backbone=...)` silently discarded both objects and ran random-safe editing.
- A non-empty legal pool forced one edit per requested budget step. There was no STOP/no-op action, no cycle state, and no immediate reverse-edit guard.
- Proposal-ranker checkpoints did not declare the task, region, or operations used by their teacher data. A 5-prime UTR teacher could therefore be used by a decoder whose default editable regions included 3-prime UTR.

## New behavior

- `rl/action_scoring.py::operation_log_score` is the sole score definition for `sub`, `ins`, `del`, and baseline `stop`. Ranker training, candidate enumeration, proposal audit, and cascade audit delegate to it.
- Decoder stochastic selection is `softmax(log_score / proposal_temperature)`. It no longer applies softmax to raw CTMC intensity.
- `sample_mrna()` routes a supplied `(model, backbone)` pair into `model_guided_edit_record()`; supplying only one raises. No-model calls retain random-safe behavior.
- `MRNARecord.metadata` records decoder provenance: `decoder_type`, checkpoint path/SHA when known, Oracle-guidance status, STOP outcome, applied/max edit count, cycle rejections, and action-space expansion status.

## Backward compatibility

- Existing calls that provide neither model nor backbone keep the random-safe API and return type.
- Existing calls to `model_guided_edit_record()` still return `MRNARecord`; decoder details are additive `metadata`.
- Existing Stage A checkpoints lack `trained_action_space` and retain the historical UTR default. This legacy fallback is explicit in the loader; new proposal-ranker checkpoints are fail-closed.
- Candidate `model_score` / `student_score` in proposal-ranking artifacts now mean log action score. Consumers that relied on raw intensity must migrate to `exp(model_score)` only when a raw intensity is specifically needed.

## Checkpoint metadata

New proposal-ranker checkpoints include:

```json
{
  "trained_task": "T5",
  "trained_editable_regions": ["utr5"],
  "trained_operations": ["sub", "ins", "del"],
  "trained_action_space": {
    "trained_task": "T5",
    "trained_editable_regions": ["utr5"],
    "trained_operations": ["sub", "ins", "del"]
  }
}
```

The nested form is canonical and the three top-level fields are mirrored for simple checkpoint consumers. The loader accepts either form and attaches this contract to the loaded model. Decoder requests inherit it by default. A request outside this task/region/operation domain raises unless `allow_action_space_expansion=True`; expanded outputs are marked `out_of_training_action_space=true`.

## STOP semantics

STOP leaves the record unchanged, costs zero edits, and has zero property delta. There is no learned STOP head in this stage: `stop_logit_bias` supplies the documented baseline score. `allow_stop=True` is the default. If every legal edit is below `stop_logit_bias + min_action_margin`, decoding terminates immediately. Otherwise STOP remains in the temperature-controlled action set and still consumes no budget.

## Cycle prevention

`DecoderState` stores SHA-256 hashes of all visited full sequences and an action history. A candidate returning to a visited sequence is rejected. Immediate reverse substitutions such as `A -> G` followed by `G -> A` are rejected by default. If all candidates are rejected, the decoder auto-STOPS and records `cycle_rejections`.

## Known limitations

- STOP is a configurable baseline, not a trained policy head; its calibration must be learned only in a later constrained policy-optimization stage.
- Current teacher provenance is 5-prime-UTR-local and one-step. Action-domain locking prevents silent expansion but does not create 3-prime-UTR or multi-step supervision.
- Oracle guidance remains a local proxy objective and must not be described as experimental translation efficiency, half-life, or assay evidence.
- Legacy Stage A checkpoints have no action-domain metadata. They are supported for compatibility but should be regenerated or wrapped with explicit metadata before scientific claims.

## Verification

Run:

```bash
cd /home/cunyuliu/mrna_editflow_goal/mrna_editflow
  /home/cunyuliu/miniconda3/envs/pc_cng/bin/python \
  -m pytest -q \
  mrna_editflow/tests/test_stage1_decoder_semantics.py \
  mrna_editflow/tests/test_training_sampling.py \
  mrna_editflow/tests/test_region_adapters.py \
  mrna_editflow/tests/test_baselines_ablation.py
```

Result: `77 passed, 1 warning in 13.85s` for the final Stage 1 semantics, sampling,
region-adapter, and baseline-ablation command above; additionally, `16 passed
in 90.06s` for `test_operators.py` and `test_protein_conditioned_cds.py`. The
warning is PyTorch's existing nested-tensor prototype warning; it is not a
biological-constraint failure. No performance conclusion follows from these
tests.

After adding the real ranker-checkpoint action-domain loading assertion, the
focused `test_stage1_decoder_semantics.py` + `test_training_sampling.py` rerun
also passed: `27 passed in 13.94s`.
