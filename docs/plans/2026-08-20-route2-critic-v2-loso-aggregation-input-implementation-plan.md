# Route 2 Critic V2 LOSO Aggregation Input Implementation Plan

> **Execution policy:** Freeze and test the V2 input path only. Do not read or aggregate real LOSO outcomes.

**Goal:** Turn the future 21 primary and 21 matched-baseline V2 LOSO terminal runs into the exact three seed-level inputs consumed by the shared aggregator and readiness gate.

**Architecture:** A prospective V2 aggregation protocol fixes identities and output roots. A V2-only builder uses runtime configs as the path/provenance authority, validates exact paired folds and terminal summaries, and emits the unchanged shared aggregation-input schema.

**Files:**

- Add the V2 LOSO aggregation protocol.
- Add a V2-only three-seed aggregation-input builder.
- Bind the aggregation protocol into the V2 readiness protocol/builder/adjudicator.
- Add focused builder plus shared-aggregator tests.
- Update execution, central-attempt and paper evidence records.

**Risk:** High. A mispaired fold would invalidate the cross-study readiness estimate.

**Minimum verification:** Exact three seeds and seven studies; 21+21 unique configs; exact study/seed/GPU/output pairing; terminal TEST-preserving summaries; shared aggregator accepts every generated payload; protocol drift and overwrite refusal; adjacent readiness regression.
