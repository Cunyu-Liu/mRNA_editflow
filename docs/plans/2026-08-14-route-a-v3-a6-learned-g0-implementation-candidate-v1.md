# Route A V3 A6 learned base/value — G0 implementation candidate v1

## Status

This is a `DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL` and `NON_AUTHORITATIVE`
implementation candidate. It does not modify, promote, or supersede the parent
protocol draft. The parent draft continues to state
`PROTOCOL_SCHEMA_STATIC_VALIDATOR_AND_FOCUSED_TEST_ONLY`; therefore this code is
kept as a separate G0 review candidate and needs new explicit authority before
it can be registered as an active implementation.

Parameter updates need a further, separate owner authorization even after any
future implementation promotion. This candidate cannot train, run a model,
touch CUDA, read or write a checkpoint, read member rows or sequences, create a
runtime output file, select a model, change qualification/credit/canonical
state, assert A6 PASS/L3, or unlock A7.

## Implemented G0 surface

The candidate implements only pure, non-executing interfaces:

1. complete parent draft and candidate-config loading and validation;
2. a shape-only base/value architecture plan with the frozen 28/154/225 widths,
   separate base/value parameter ownership, and zero parameter construction;
3. a strict aggregate-only future input-contract manifest for ordinary-public
   role, qualification, split/leakage, rights, and scratch exposure;
4. an adapter that reuses the existing synthetic CPU exact-DAG kernel's state,
   hard-legality, STOP, source anchoring, alias aggregation, and budget
   transition interfaces in tests;
5. pure support-floor base-rate normalization, scalar-potential Doob-rate, and
   absorbing terminal-boundary formulae with supplied scalars only;
6. plan-only objective, independent-reference, approximation-gate, legality,
   trajectory, CUDA provenance, checkpoint-role, and output-manifest fields;
7. a `--validate-only` JSON report to standard output with zero data-row reads,
   model construction/forward, optimizer steps, parameter updates, CUDA probes,
   GPU runs, checkpoints, and runtime files;
8. authority barriers for train, optimizer-step, checkpoint-write, and CUDA
   preflight requests. The barrier rejects the request before the injected
   callback or any output operation.

## Explicit non-implementation

There is no Torch module or trainer in this candidate. “Architecture build” in
G0 means an internally checked shape/layer plan, not parameter allocation or a
forward pass. CUDA ownership preflight is an interface boundary only: under the
inactive parent it rejects before probing CUDA. A real Torch model, tensor/data
loader, optimizer, CUDA probe, checkpoint store, runtime publisher, exact
reference run, or training loop belongs only in a later promoted protocol and
parameter-update authority.

No scientific evidence is produced. The formal `FLOW_BASE_LEGAL_CTMC` task
remains `NOT_RUN`; A6 remains `IN_PROGRESS`; L3 remains `NOT_ESTABLISHED`; A7
remains locked; and every A1 count and canonical value remains unchanged.
