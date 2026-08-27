# Route A V3.3 Route 2 Goal 0 activation summary

## Outcome

Route 2 is configured as `ACTIVE_DUAL_TRACK_DEVELOPMENT` on branch
`route-a-v3-route2-20260815`. The branch starts from
`f94f3f193fc893cdfec1b6be2c472d4cdad86931`, which preserves the completed
81,794-parameter critic negative result.

This is an execution/configuration milestone, not a scientific result.
`scientific_claim_status` remains `NOT_ESTABLISHED`, the qualified state remains
`ordinary=1 / A1=1 / true-A2=0 / canonical=6,547`, and Edit Flow remains
`ACTIVE_IMPLEMENTATION_TARGET_NOT_YET_SCIENTIFICALLY_ESTABLISHED`.

## Active execution boundary

- Canonical conversion, baselines, Delta-predictor development, and Flow G0
  implementation are enabled.
- Any parameter update requires an NVIDIA GPU selected from physical GPU0-GPU5;
  CPU fallback is not allowed.
- GSE232572 and E-MTAB-10902 remain Evaluation-only. GSE246381 remains sealed
  and excluded.
- Historical successor, runtime-ledger, one-read, and resource-once workflows
  are not Route 2 prerequisites.
- `SUB + STOP` is the current action space. `INS/DEL` are not supported in V1.
- Guided XEditFlow remains `NOT_STARTED_DEPENDENCY_NOT_MET` until both
  `CRITIC_READY_FOR_GUIDANCE` and `FLOW_G0_READY` are established.

## Storage and preflight

The Route 2 artifact root is:

`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/`

The required directory structure was created without touching the dirty main
checkout. At activation time, GPU0-GPU5 all had active utilization from unrelated
jobs, so no Route 2 GPU training was started.

## Files

- `configs/route_a_v3_route2_v1.json`
- `scripts/route_a_v3/validate_route2_v1.py`
- `tests/route_a_v3/test_validate_route2_v1.py`

The validator checks only execution-changing boundaries: the frozen current
facts, exact 14-study roles, Development/Evaluation isolation, GPU-only training,
the removal of obsolete development prerequisites, generated-candidate credit,
and the two-readiness guided-generation dependency.

Both SetFlow V4 screen training arms have now terminated with summary artifacts,
but their performance payloads remain unread pending the frozen post-training
Validation package. A SetFlow-only V4.0.2 terminal coordinator has been added so
the eight preregistered checkpoint evaluations can run on GPUs 0–4 while GPU5
continues the independent Critic recovery. It reuses the frozen validator and
atomic SetFlow gate and does not reopen the already consumed historical Critic
failures. This is execution plumbing only; no recovery, diversity, NLL, model-
advantage, or publication-readiness claim has yet been established.
