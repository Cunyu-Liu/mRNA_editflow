# MK0 state, action and replay specification

## Scope

The runtime process is a time-inhomogeneous CTMC on an extended state. It is not a CTMC only on sequence strings. External time is an input to the generator and is never incremented by a stochastic action update.

Use the names:

```text
x_src     fixed source sequence
x_current current sequence at external time t
x_target  paired/corrupted target used only to build training auxiliaries
```

The inference-visible state is `Y_t = (x_src, x_current, M_run, region, context, target_condition, B, b_t, h_run, q_t)`. The shorthand `S_t=(Y_t,t)` includes external time. `Z_aux=(z_src,z_t,z_target)` is not a state field and must never enter a rate or STOP head.

## Runtime-state invariants

`schemas/edit_state_v1.schema.json` validates the serialized envelope. Code must additionally enforce these cross-field invariants, which JSON Schema cannot express by itself:

1. `len(mapping_run.token_origins) == len(x_current)`.
2. `len(mapping_run.source_to_current) == len(x_src)`.
3. `len(mapping_run.gap_ids) == len(x_current) + 1`.
4. Every non-null source-to-current index points to the matching `SOURCE` origin and is unique.
5. Protected indices point only to protected source origins.
6. `remaining_budget == initial_budget - executed_action_count` for executed INS/SUB/DEL actions; STOP and forced termination do not change either value.
7. `history_run` is derivable only from executed actions and contains no target/alignment information.
8. ACTIVE has no termination event. HALTED has exactly one termination event and is absorbing.
9. State hashes are computed from canonical serialized runtime state; the `state_hash` field itself is excluded from the hash input.
10. A sequence, mapping, history or budget change after HALTED is a hard failure.

The runtime mapping preserves source token IDs, inserted-event IDs, gap IDs and protected flags through every indel. Gap IDs are stable logical identifiers, not bare integer positions; coordinates in an action are always pre-action current-state coordinates.

## Atomic actions

For a current sequence of length `L`:

- `INS(g,v)` uses a gap index `g` in `[0,L]` and `v∈{A,C,G,U}`;
- `SUB(i,v)` uses a token index `i` in `[0,L-1]`, with `v != x_current[i]`;
- `DEL(i)` uses a token index `i` in `[0,L-1]`;
- `STOP` has no coordinate or nucleotide.

`schemas/edit_action_v1.schema.json` fixes the serialized form. INS/SUB/DEL cost one atomic edit. An edit that later gets reversed, cycles back, or does not change final Levenshtein distance still consumed its budget at execution time.

## Deterministic update `T_Y`

All updates happen at a fixed external time:

| Action | Sequence update | Mapping/history update | Budget/status update |
|---|---|---|---|
| INS | insert nucleotide at the selected gap | create one inserted-event origin; split the logical gap deterministically; append action ID | decrement budget; remain ACTIVE |
| SUB | replace the selected nucleotide | retain its source or inserted origin and gap topology; append action ID | decrement budget; remain ACTIVE |
| DEL | remove the selected nucleotide | remove its current origin; set a deleted source token's source-to-current entry to null; merge adjacent gaps deterministically; append action ID | decrement budget; remain ACTIVE |
| STOP | sequence unchanged | mapping and edit history frozen | budget unchanged; enter HALTED with `LEARNED_STOP` |

After every edit, recompute the current-state coordinate view, protection mask and all action rates. A sampler must not reuse a stale source-only encoding.

## Hard legality before normalization

`m_C(S_t,a)` is applied before operation/token normalization. Illegal actions receive exact zero rate. The constructive mask covers at least:

- out-of-range current coordinates or cross-region edits;
- protected source anchors;
- length below/above configured limits;
- exhausted budget;
- identity substitution;
- invalid alphabet tokens;
- every edit action in HALTED; and
- task-specific UTR grammar constraints.

STOP legality and edit-action availability are separate. An empty INS/SUB/DEL set does not imply that STOP has zero hazard. A coupling target action forbidden by the mask is recorded in a rejected/repair ledger; dropping its loss term silently is forbidden.

## Inverse and replay semantics

Two meanings must not be conflated:

- **Audit undo:** a stored pre-state plus action record reconstructs the exact pre-state without consuming model budget. This is a verification operation, not a CTMC event.
- **Executed reverse edit:** a new INS/SUB/DEL that restores earlier sequence content is a new model event and consumes another budget unit.

Replay starts from the canonical initial state and consumes the recorded seed, time interval, hazards, random draws, candidate-action hash and selected action in order. Every intermediate state hash must match exactly. Replay success is the fraction of complete state-hash sequences that match, not only final-string equality.

## Termination and failure

The only termination reasons are those in `termination_event_v1.schema.json`:

```text
LEARNED_STOP
FORCED_BUDGET
FORCED_NO_LEGAL_EDIT_ACTION
FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
FORCED_TIME_HORIZON
FAILED_NUMERICAL
```

`FAILED_NUMERICAL` is outside the CTMC and does not produce a valid candidate. `NO_EVENT` is a trajectory step outcome, not a termination reason. A forced reason is never relabelled as learned STOP.

## Separately reported accounting

Every trajectory reports at least:

- cumulative executed INS/SUB/DEL action count;
- final source-to-output Levenshtein distance;
- source-token preservation/protection statistics;
- remaining budget; and
- termination reason.

These quantities answer different questions and must not be collapsed into one “edit count”.
