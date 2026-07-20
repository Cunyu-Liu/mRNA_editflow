# RL Protocol v1

**Status**: P1-14 — protocol frozen.

**Scope**: This document specifies the reinforcement learning protocol for mRNA-EditFlow, covering the action space, policy interface, reward computation, oracle tiers, budget accounting, and test set locking. All RL training and evaluation in the project must follow this protocol.

---

## 1. Action Space

### 1.1 Action Types

The mRNA design MDP has 4 action types:

| Action | Op | Position | Nucleotide | Description |
|---|---|---|---|---|
| Insertion | `ins` | `pos` (0..L-1) | `nt` (0..3) | Insert nucleotide `nt` after position `pos` |
| Substitution | `sub` | `pos` (0..L-1) | `nt` (0..3) | Replace nucleotide at `pos` with `nt` |
| Deletion | `del` | `pos` (0..L-1) | — | Delete nucleotide at `pos` |
| STOP | `stop` | -1 | -1 | Terminate the trajectory |

**Nucleotide encoding**: `0=A, 1=C, 2=G, 3=U` (RNA alphabet, V=4).

### 1.2 Legal Action Mask

The legal action mask restricts the action space based on biological constraints:

- **5'UTR / 3'UTR**: All 4 nucleotides allowed for ins/sub; all positions deletable.
- **CDS substitution**: Only synonymous nucleotides (via `synonymous_nt_sub_mask`).
- **CDS indels**: Forbidden by default (`codon_indel=False`) to preserve reading frame. If `codon_indel=True`, allowed only at codon-start positions (phase 0).
- **Identity substitutions**: Excluded (substituting a nucleotide with itself is not a real edit).

The STOP action is always legal (unless explicitly disabled).

### 1.3 Implementation

The action space is implemented in [rl/action_space.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/action_space.py):

```python
@dataclass(frozen=True)
class Action:
    op: str  # "ins", "sub", "del", "stop"
    pos: int = -1
    nt: int = -1
    def is_stop(self) -> bool: return self.op == "stop"

STOP_ACTION = Action(op="stop", pos=-1, nt=-1)

def build_legal_action_mask(record, device, *, codon_indel=False, allow_identity_sub=False) -> ActionMask
def apply_action(record, action, *, transcript_id=None) -> MRNARecord
```

---

## 2. Policy Interface

### 2.1 CTMC Policy

The policy wraps `MRNAEditFormer` and produces a normalized CTMC (Continuous-Time Markov Chain) action distribution:

```
q(action) = lambda_op · p_op(action)
Lambda(s) = sum of all q(action | s)
p(a | s) = q(a | s) / Lambda(s)
```

where:
- `lambda_op` = rate for operation `op` (ins/sub/del), from model output `rates[L, 3]`
- `p_op(action)` = token distribution for operation `op`, from `ins_probs[L, V]` / `sub_probs[L, V]`
- `STOP` probability = `1 - sum of all action probabilities` (external STOP)

### 2.2 Log-Probabilities

The policy computes log-probabilities for all legal actions:

```python
@dataclass
class ActionLogProbs:
    ins_logprobs: torch.Tensor  # [L, V] log p(ins at pos i, nt a)
    sub_logprobs: torch.Tensor  # [L, V] log p(sub at pos i, nt a)
    del_logprobs: torch.Tensor  # [L] log p(del at pos i)
    stop_logprob: float         # log p(STOP)
    log_partition: float        # log Lambda(s)
    raw_*: Optional              # pre-external-mask log-probs (for diagnostics)
```

### 2.3 Trajectory Log-Probability

The trajectory log-probability is the sum of per-step log-probabilities:

```
log p(τ) = sum_t log p(a_t | s_t)
```

This is differentiable (when `no_grad=False`) and can be used in REINFORCE.

### 2.4 STOP Action Handling

The STOP action is handled specially:

1. **External STOP**: The model outputs rates for ins/sub/del only. The STOP probability is computed as `1 - sum of action probabilities`.
2. **Differentiable STOP**: For REINFORCE, the STOP log-prob is recomputed as:
   ```python
   p(STOP) = 1 - sum(exp(ins_lp)) - sum(exp(sub_lp)) - sum(exp(del_lp))
   log p(STOP) = log(p(STOP))
   ```
3. **Budget-aware STOP**: The stop rate can be modulated by the remaining edit budget (optional, `stop_rate_strategy="budget_aware"`).

### 2.5 Quality Before/After Masking

The policy provides diagnostics for the quality of the action distribution before and after applying the legal action mask:

- **Raw distribution**: Model output normalized over all (op, pos, nt), including illegal ones.
- **Masked distribution**: Model output normalized over legal actions only.
- **Mass on illegal actions**: `raw_mass_on_illegal = 1 - sum of legal raw probabilities`.

High `raw_mass_on_illegal` indicates the model is wasting probability mass on illegal actions — a sign of poor training.

### 2.6 Implementation

The policy is implemented in [rl/policy.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/policy.py):

```python
class Policy:
    def legal_action_mask(self, record) -> ActionMask
    def action_logprobs(self, record, *, budget_remaining=None, budget_total=None,
                        return_raw=False, no_grad=True) -> ActionLogProbs
    def sample(self, record, *, budget_remaining=None, budget_total=None,
               generator=None) -> Tuple[Action, float]
    def trajectory_logprob(self, trajectory, *, budget_total=None) -> float
    def quality_before_after_masking(self, record, ...) -> dict
```

---

## 3. Oracle Tiers

The project uses three tiers of oracles, with increasing independence:

### 3.1 Tier 1: Development Oracle (Heuristic)

- **Type**: `LocalTranslationOracle` (heuristic, no learned components)
- **Independence**: None (uses the same codebase as training)
- **Use case**: Development mode (`run_mode="development"`)
- **Paper-eligible**: No
- **Manifest**: Not required

### 3.2 Tier 2: Cross-Fitted Predictor Ensemble (P1-04)

- **Type**: Ensemble of ≥2 architectures (e.g., CNN + Transformer) trained on ≥2 datasets (MPRA + MRL), with cross-fitting (each predictor trained on a fold where the evaluation records are held out).
- **Independence**: Partial (trained on different data than the policy, but same codebase)
- **Use case**: Training reward for RL
- **Paper-eligible**: Yes, for training-time claims
- **Manifest**: Required (`--oracle-manifest`)

### 3.3 Tier 3: Independent Final Oracle (P1-05)

- **Type**: Frozen, hidden oracle — a single predictor (or ensemble) trained on data that is NOT used for policy training or cross-fitted predictor training. The oracle is frozen before RL training begins and is not accessible to the policy during training.
- **Independence**: Full (different data, different training run, hidden from policy)
- **Use case**: Final evaluation and paper claims
- **Paper-eligible**: Yes, for all paper claims
- **Manifest**: Required, with `independent=true` and `independence_statement`

### 3.4 Oracle Contract

All oracle interactions are governed by the artifact contract in [eval/artifact_contract.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/eval/artifact_contract.py):

```python
def load_and_verify_oracle_manifest(path, *, run_mode) -> dict
```

The manifest must include:
- `schema_version: 1`
- `oracle_type`: e.g., "cross_fitted_ensemble" or "independent_final"
- `independent`: bool
- `source`: data source description
- `artifact_path`: path to the oracle artifact
- `artifact_sha256`: SHA-256 of the artifact
- `independence_statement`: human-readable statement of independence

In paper mode, the oracle must be independent (Tier 3). Using a heuristic oracle (Tier 1) in paper mode raises `OracleContractError`.

---

## 4. Budget and Query Accounting

### 4.1 Edit Budget

The edit budget is the maximum number of non-STOP actions per trajectory:

```
C(τ) = sum_t 1[a_t ≠ STOP] ≤ c_max
```

Default: `c_max = 3` (matching `--edit-budget 3`).

### 4.2 Oracle Query Budget

The oracle query budget is the maximum number of oracle evaluations allowed during RL training:

```
N_oracle_queries ≤ N_max
```

This is tracked to prevent overfitting to the oracle. Default: `N_max = 100,000` (for P2-08 large-scale RL).

### 4.3 Accounting

Every oracle query is logged with:
- `record_id`: the input record
- `query_time`: timestamp
- `oracle_score`: the returned score
- `oracle_version`: the oracle manifest SHA-256

The log is written to `benchmark/paper/oracle_query_log.jsonl` and verified at evaluation time.

### 4.4 CTO Hard Constraint

When using CTO (Innovation 1), the edit budget is a *hard constraint* — trajectories exceeding the budget are rejected during collection and masked out of the loss. This guarantees zero constraint violations at test time.

---

## 5. Test Set Lock

### 5.1 Test Set Isolation

The test set is locked at the beginning of the project and is NEVER used during training or hyperparameter tuning. The lock is enforced by:

1. **Split contract**: The split manifest (`data/reconstructed/p0_data_reconstruction_v1/.../split_manifest.json`) specifies which records belong to train/val/test roles.
2. **Exact-match fail-closed**: The `run_multiseed_benchmark.py` script verifies that the test records exactly match the split contract (via `records_content_digest`). Any mismatch causes `SystemExit`.
3. **Paper mode**: In paper mode (`run_mode="paper"`), the split manifest is required and the test role is enforced.

### 5.2 Test Set Evaluation

Test set evaluation is performed ONLY at the end of the project, using the frozen independent oracle (Tier 3). The evaluation is:

1. Load the frozen policy checkpoint.
2. Load the test split from the frozen split manifest.
3. Generate candidates for each test record (with edit budget = 3).
4. Score each candidate with the independent oracle.
5. Compute T1-T7 metrics (legal fraction, TE, MRL, novelty, protein identity, etc.).
6. Aggregate across decoder seeds with bootstrap CIs.

### 5.3 Test Lock Enforcement

The test lock is enforced by the `--train-idx/--val-idx/--test-idx` arguments in `train_backbone.py` (P1-10):

```python
def _verify_idx_files(train_idx, val_idx, test_idx, run_mode)
def _enforce_exact_match_fail_closed(records, split_contract, split_role)
```

In paper mode, these arguments are required and the exact-match check is performed. Any mismatch raises `SystemExit`.

---

## 6. RL Training Protocol

### 6.1 Training Loop

```
for episode in range(n_episodes):
    # 1. Collect trajectories (with CTO rejection sampling)
    trajectories = collect_feasible_batch(policy, mdp, batch_size)

    # 2. Compute synergy reward (if using Innovation 2)
    samples = collect_synergy_samples(policy, mdp, batch_size)

    # 3. Compute loss (feasibility-masked REINFORCE + synergy shaping)
    loss = compute_constrained_loss(trajectories) + alpha * compute_synergy_loss(samples)

    # 4. Update policy
    loss.backward()
    optimizer.step()

    # 5. Log metrics
    log(episode, loss, feasibility_rate, synergy_reward, oracle_queries)
```

### 6.2 Hyperparameters (Default)

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 0.01 | SGD |
| Batch size | 8 | Trajectories per update |
| Edit budget | 3 | Hard constraint (CTO) |
| Max trajectory length | 20 | Steps |
| Baseline | EMA | Decay = 0.9 |
| Gradient clipping | 10.0 | Max norm |
| Lambda schedule | warmup=20, anneal=30 | Synergy RL |
| Oracle query budget | 100,000 | Total queries |

### 6.3 Checkpointing

Checkpoints are saved every `checkpoint_interval` episodes to `benchmark/paper/rl_checkpoints/`. Each checkpoint includes:
- Policy weights (`policy.pt`)
- Optimizer state (`optimizer.pt`)
- Training metrics (`metrics.json`)
- Oracle query log (`oracle_queries.jsonl`)

---

## 7. Evaluation Protocol

### 7.1 Development Evaluation

- **Oracle**: Tier 1 (heuristic) or Tier 2 (cross-fitted ensemble)
- **Records**: Development set (not test)
- **Metrics**: T1-T7 (legal fraction, TE, MRL, novelty, protein identity, edit distance, reading frame)
- **Frequency**: Every `checkpoint_interval` episodes

### 7.2 Paper Evaluation

- **Oracle**: Tier 3 (independent final)
- **Records**: Test set (locked, frozen)
- **Metrics**: T1-T7 + multi-seed bootstrap CIs
- **Frequency**: Once, at the end of the project

### 7.3 Acceptance Gates

The RL training is considered complete when:
1. **Feasibility**: 100% of test trajectories satisfy the edit budget constraint.
2. **Synergy**: Synergy score > 0 (if using Innovation 2).
3. **Return**: Mean test return ≥ baseline (non-RL decoder).
4. **Oracle queries**: Total queries ≤ budget.

---

## 8. Files

- [rl/action_space.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/action_space.py) — Action types and legal masks
- [rl/policy.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/policy.py) — CTMC policy
- [rl/tiny_mdp.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/tiny_mdp.py) — Tiny MDP for testing
- [rl/cto.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/cto.py) — Innovation 1 (CTO)
- [rl/synergy.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/synergy.py) — Innovation 2 (Synergy RL)
- [eval/artifact_contract.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/eval/artifact_contract.py) — Oracle and split contract
- [train_backbone.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/train_backbone.py) — Training entry (split enforcement)
- [eval/run_multiseed_benchmark.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/eval/run_multiseed_benchmark.py) — Multi-seed benchmark (exact-match fail-closed)

---

## 9. Protocol Freeze

This protocol is **frozen** as of 2026-07-19. Changes require:
1. Documenting the change in `docs/rl_protocol_v1_changelog.md`.
2. Re-freezing with a new version number (v2).
3. Re-running all affected experiments.

**Frozen parameters**:
- Action space: 4 types (ins/sub/del/STOP)
- Nucleotide encoding: V=4 (A,C,G,U)
- CDS constraints: synonymous-only sub, no indels (default)
- Oracle tiers: 3 (development / cross-fitted / independent final)
- Edit budget: 3
- Test lock: exact-match fail-closed

---

**Document SHA-256**: to be computed after final review.
**Authors**: mRNA-EditFlow team
**Last updated**: 2026-07-19
