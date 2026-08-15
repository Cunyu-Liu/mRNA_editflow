# DEC028 machine authority repair Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Rebuild the DEC028 pending authority bundle from the valid EVT060 current authority so that the official authority validator and focused integrity tests accept the new pending state without changing scientific counts, claims, or execution locks.

**Architecture:** Preserve the validated 214ee9c DEC027 EVT060 projection as the historical predecessor. Add one explicit DEC028 pending-projection branch to the existing authority validator and registry-manifest closure, rather than weakening historical checks or treating the pending candidate as active. Every current-surface change must be declared in the DEC028 amendment, checked across config, qualification, supersession, interim, and registries, and be rejected if it enables data, CUDA, model, checkpoint, training, G1, A7, sealed access, a count change, or a scientific claim.

**Tech Stack:** Python standard library and PyYAML, pytest, Git worktrees; planning context from @brainstorming and @writing-plans.

---

### Batch 1: Explicit DEC028 pending projection

**Files:**

- Create: docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml
- Create: docs/contracts/candidates/mrna_xeditflow_route_a_v3_dec028_single_study_mainline_contract_v1.md
- Create: configs/route_a_v3_dec028_single_study_protocol_v1.json
- Create: configs/route_a_v3_dec028_successor_p0_schema_v1.json
- Modify: configs/route_a_v3.yaml
- Modify: configs/route_a_v3_a1_qualification.json
- Modify: docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml
- Modify: docs/execution/route_a_v3_a1_interim.yaml
- Modify: docs/execution/route_a_v3_a6_interim.yaml
- Modify: docs/execution/route_a_v3_data_role_registry.yaml
- Modify: docs/execution/route_a_v3_task_registry.yaml
- Modify: docs/execution/route_a_v3_split_registry.yaml
- Modify: docs/execution/route_a_v3_task_split_matrix.yaml
- Modify: docs/execution/route_a_v3_claim_evidence_matrix.yaml
- Modify: docs/execution/route_a_v3_baseline_registry.yaml
- Modify: docs/execution/route_a_v3_decision_log.yaml
- Modify: docs/execution/route_a_v3_registry_manifest.json

**Risk:** High — these files decide the active operational authority and could accidentally rewrite historical DEC027 facts or open a prohibited execution gate.

**Implementation:** Reapply only the contract's DEC028 pending successor data. Maintain DEC027 as the effective authority until a dynamic runtime sync; declare DEC028 as a pending amendment, preserve 1/1/0/6547 and NOT_ESTABLISHED, and require every data/CUDA/model/training/G1/A7/sealed permission to remain false.

**Minimum verification:** The DEC028-specific static validator and one source-level inspection must show pending status, DEC027 effective authority, unchanged counts, and all locks false.

**Independent review:** Yes — this is the action-changing authority surface described in the contract.

### Batch 2: Validator and historical-integrity forward-port

**Files:**

- Modify: scripts/route_a_v3/validate_a0_bundle.py
- Modify: tests/route_a_v3/test_a0_authority.py
- Modify: tests/route_a_v3/test_a0_integrity_guards.py
- Create: tests/route_a_v3/test_dec028_authority_bundle.py

**Risk:** High — an incorrect adaptation could either reject a valid append-only pending projection or weaken detection of historical count, lock, and claim drift.

**Implementation:** Add a closed DEC028-pending projection validator. Keep the DEC027 EVT060 historical validators intact for historical sections, but make their current-projection assertions recognize the contractually declared pending successor and its separately checked static leaves. Do not skip tests or accept arbitrary manifest rehashes.

**Minimum verification:** Run the official authority modules that previously failed, the new DEC028 module, and the DEC028 static validator. The repaired tree must have no authority issues; mutation fixtures must still catch count, lock, claim, runtime-event, and manifest bypasses.

**Independent review:** Yes — the validator is the final static authority gate before any runtime write.

### Batch 3: Runtime-sync implementation preparation

**Files:**

- Create: configs/route_a_v3_dec028_authority_runtime_sync_v1.json
- Create: scripts/route_a_v3/dec028_authority_runtime_sync.py
- Create: tests/route_a_v3/test_dec028_authority_runtime_sync.py

**Risk:** High — this is the first component allowed to append a live operational event.

**Implementation:** Build a two-step prepare/publish publisher that dynamically derives the successor event from the live predecessor, creates immutable snapshots before mutable updates, and accepts only exact publication prefixes for recovery. It will not be run until the authority bundle is bound, committed, pushed to the confirmed origin, and passes Batch 2.

**Minimum verification:** Synthetic temporary-runtime tests cover dynamic event derivation, frozen count/lock/claim preservation, prepared-directory no-overwrite, and rejection of a changed live predecessor.

**Independent review:** Yes — a distinct reviewer is needed before the live prepare/publish command.
