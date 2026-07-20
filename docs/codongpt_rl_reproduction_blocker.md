# P1-09 CodonGPT REINFORCE Reproduction: Measured Report + Blocker

**Status**: Pretrained checkpoint verified. RL training code NOT available → **degraded to pretrained-checkpoint-only** per degradation path.
**Date**: 2026-07-19
**Author**: trae agent (autonomous execution, P1-09)
**Degradation**: Per P1 plan degradation path — "P1-09 受阻: 降级为 pretrained checkpoint only, 记录 docs/codongpt_rl_reproduction_blocker.md"

---

## 1. Executive Summary

This report documents the P1-09 task: reproduce CodonGPT REINFORCE training on public cargos (ACTB, HLA-A).

**What was accomplished**:
- ✅ CodonGPT pretrained checkpoint loaded and verified (3.4M params, GPT-2 architecture, codon-level tokenizer).
- ✅ `SynonymMaskingLogitsProcessor` verified — enforces synonymous codon substitutions, matching our P1-07 action space design.
- ✅ Inference test on ACTB (20 aa prefix): 17/20 codons changed, all synonymous, 15.1s.
- ✅ Inference test on HLA-A (20 aa prefix): 11/20 codons changed, all synonymous, 16.8s.

**What was NOT accomplished**:
- ❌ Public RL training notebooks (github.com/NanilTx/codonGPT_pub) are **not accessible** — git clone failed (`could not read Username for 'https://github.com'`). The repo either does not exist, is private, or the server has no internet access to github.com.
- ❌ REINFORCE training on top of CodonGPT was **not executed** — the public RL training code is the reference implementation, and without it, reproducing the exact training loop (reward function, rollout strategy, hyperparameters) is not possible without guesswork.

**Verdict**: Per the degradation path, P1-09 is **degraded to pretrained checkpoint only**. The blocker is documented below, along with a reproduction plan using our P1-08 REINFORCE infrastructure.

---

## 2. Pretrained Checkpoint Verification

### 2.1 Checkpoint Location

```
/home/cunyuliu/mrna_editflow_goal/mrna_editflow/external_tools/codonGPT_hf_ee7017c4/
```

### 2.2 Files and SHA-256

| File | SHA-256 | Size |
|---|---|---|
| `README.md` | `0b54674c3d2d8d835637efca687ee0e4a61d468e86accd8e2cde47e309538330` | 8,771 B |
| `config.json` | `4f62874bda276cefcb133633f27a58a673a93b1f2ba47f1b5f4f7d7c732c0cc6` | 803 B |
| `generation_config.json` | `ba98ce436484dcaf16b25cd8a9774913652b869557c22466f33d6759c9c98315` | 119 B |
| `pytorch_model.bin` | `df41546883e31ba13598d5ae74044666502a89ba34630d6f6c32943836e6f454` | 17,968,165 B |
| `synonymous_logit_processor.py` | `e7a384e277380b15c63db53c45b297e304335ce23b6bb22181150163a9f0365a` | 3,912 B |
| `tokenizer.py` | (in manifest) | (in manifest) |
| `model_manifest.json` | (in manifest) | (in manifest) |

### 2.3 Model Architecture

- **Architecture**: GPT-2 (causal LM)
- **Parameters**: 3.4M
- **Tokenizer**: `CodonTokenizer` (extends `GPT2Tokenizer`), vocab_size = 67
  - Special tokens: BOS=1, EOS=2
  - Codon tokens: 64 codons (AAA, AAC, ..., TTT) + special tokens
- **Dtype**: float32
- **Forward signature**: standard `transformers.AutoModelForCausalLM.forward(input_ids, ...)`

### 2.4 Action Space (via `SynonymMaskingLogitsProcessor`)

The legal action space for CodonGPT is **synonymous codon substitution** at each position:

```python
class SynonymMaskingLogitsProcessor(LogitsProcessor):
    def __init__(self, current_aa, tokenizer, aa_to_codon=None):
        self.current_aa = current_aa
        # aa_to_codon_human maps each amino acid to its synonymous codons
        # e.g., 'A' -> ['GCT', 'GCC', 'GCA', 'GCG']

    def __call__(self, input_ids, scores):
        synonymous_codons = self.aa_to_codon.get(self.current_aa, [])
        synonym_token_ids = self.tokenizer.convert_tokens_to_ids(synonymous_codons)
        mask = torch.ones_like(scores) * -float('inf')
        mask[:, synonym_token_ids] = 0
        return scores + mask
```

This is **exactly the same action space** as our P1-07 CDS synonymous substitution (`synonymous_nt_sub_mask` in [rl/action_space.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/action_space.py)), but at the **codon level** (CodonGPT generates one codon per step) rather than the **nucleotide level** (our MEF policy generates one nt edit per step).

---

## 3. Inference Test Results

### 3.1 Test Script

[scripts/test_codongpt_inference.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/scripts/test_codongpt_inference.py) (uploaded to server as `/tmp/test_codongpt_inference.py`)

### 3.2 ACTB (beta-actin) — 20 aa prefix

- **Protein prefix** (20 aa): `MDDDIAALVVDNGSGMCKAG`
- **Initial codons** (first synonymous codon per aa):
  ```
  ATG GAT GAT GAT ATT GCT GCT TTA GTT GTT GAT AAT GGT TCT GGT ATG TGT AAA GCT GGT
  ```
- **CodonGPT-optimized codons**:
  ```
  ATG GAC GAC GAC ATC GCC GCC CTG GTG GTG GAC AAC GGC TCC GGG ATG TGC AAG GCT GGC
  ```
- **Codon changes**: 17/20 (85%)
- **Synonymous check**: ✅ `MDDDIAALVVDNGSGMCKAG` == `MDDDIAALVVDNGSGMCKAG`
- **Elapsed**: 15.10s (CPU, no GPU)

**Observation**: CodonGPT shifts toward "optimal" codons (e.g., GAT→GAC for D, ATT→ATC for I, GCT→GCC for A), which are known to have higher tRNA abundance in human cells. This is the expected behavior for a codon optimization model.

### 3.3 HLA-A (MHC class I) — 20 aa prefix

- **Protein prefix** (20 aa): `GSHSMRYFFTSVSRPGRGEP`
- **Initial codons**:
  ```
  GGT TCT CAT TCT ATG CGT TAT TTT TTT ACT TCT GTT TCT CGT CCT GGT CGT GGT GAA CCT
  ```
- **CodonGPT-optimized codons**:
  ```
  GGT TCA CAT TCA ATG CGC TAC TTT TTT ACG TCA GTT TCT AGA CCC GGA CGA GGA GAA CCT
  ```
- **Codon changes**: 11/20 (55%)
- **Synonymous check**: ✅ `GSHSMRYFFTSVSRPGRGEP` == `GSHSMRYFFTSVSRPGRGEP`
- **Elapsed**: 16.79s (CPU)

**Observation**: Fewer changes than ACTB (11 vs 17), possibly because HLA-A's native codons are already closer to optimal.

### 3.4 Environment

- **Python**: 3.10 (conda env `codongpt`)
- **transformers**: 4.57.6
- **biopython**: installed via pip (for `Bio.Seq.translate`)
- **Device**: CPU (no GPU needed for inference)
- **Runtime**: ~16s per 20-codon prefix (would scale linearly to ~280s for full 376-aa ACTB)

---

## 4. Blocker: Public RL Training Code Not Available

### 4.1 What Was Expected

Per the [next_steps_sota_roadmap.md](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/docs/next_steps_sota_roadmap.md) Section 11:
> codonGPT public RL notebooks: https://github.com/NanilTx/codonGPT_pub

The public RL notebooks were expected to contain:
- REINFORCE training loop on top of CodonGPT
- Reward function (likely CAI or a proxy)
- Rollout strategy (sequential codon generation with synonym masking)
- Hyperparameters (learning rate, batch size, reward scaling)
- Public cargo benchmarks (ACTB, HLA-A)

### 4.2 What Was Attempted

```bash
cd /tmp && timeout 30 git clone https://github.com/NanilTx/codonGPT_pub.git
```

**Result**: `fatal: could not read Username for 'https://github.com': No such device or address`

### 4.3 Root Cause

One of:
1. The repository `NanilTx/codonGPT_pub` does not exist (404).
2. The repository is private (requires authentication).
3. The server (`36.137.135.49`) has no outbound internet access to `github.com` (firewall/proxy).

### 4.4 Impact

Without the reference RL training code, we cannot:
- Verify the exact reward function used by CodonGPT's RL training.
- Reproduce the exact hyperparameters and training schedule.
- Compare our P1-08 REINFORCE implementation against the reference on the same cargo set.

### 4.5 Mitigation

Per the degradation path, P1-09 is **degraded to pretrained checkpoint only**. The pretrained checkpoint is verified (Section 2-3), and the action space is confirmed to match our P1-07 design. The RL reproduction is deferred until either:
- (a) The public RL notebooks become accessible, or
- (b) We implement our own REINFORCE training on top of CodonGPT using P1-08 infrastructure (see Section 5).

---

## 5. Reproduction Plan (Using P1-08 REINFORCE)

If the public RL notebooks remain unavailable, we can reproduce CodonGPT REINFORCE using our own infrastructure:

### 5.1 Architecture Mapping

| CodonGPT concept | Our P1-07/P1-08 equivalent |
|---|---|
| Codon-level generation | Nt-level edits (our MEF policy) |
| `SynonymMaskingLogitsProcessor` | `synonymous_nt_sub_mask` in `build_legal_action_mask` |
| GPT-2 forward pass | `MRNAEditFormer.forward` |
| CAI reward (assumed) | `LocalTranslationOracle.score_record` or CAI metric |
| REINFORCE training loop | `REINFORCE` class in [rl/tiny_mdp.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/rl/tiny_mdp.py) |

### 5.2 Implementation Plan

1. **Wrap CodonGPT as a `Policy`**: Create a `CodonGPTPolicy` class that wraps the GPT-2 model and exposes `sample()`, `action_logprobs()`, `legal_action_mask()` matching our P1-07 `Policy` interface.
2. **Define a `CodonMDP`**: State = current codon sequence; action = next codon (synonymous); reward = CAI(terminal) or per-step CAI delta.
3. **REINFORCE training**: Use our `REINFORCE` class with the `CodonGPTPolicy` and `CodonMDP`.
4. **Benchmark cargos**: ACTB (376 aa), HLA-A (360 aa) — full sequences.
5. **Metrics**: pre-RL vs post-RL CAI, MFE, codon diversity, training reward curve.

### 5.3 Estimated Effort

- `CodonGPTPolicy` wrapper: ~100 lines
- `CodonMDP`: ~80 lines
- Training script: ~50 lines
- Tests: ~100 lines
- Total: ~330 lines, ~2-3 hours of implementation + testing.

### 5.4 Decision

**Deferred** — this is not blocking for 壁垒 2/4/1/3. The P1-08 REINFORCE infrastructure is already validated on our MEF policy (23 tests pass). Re-implementing it for CodonGPT is a benchmarking exercise, not a barrier. If the public RL notebooks become available, we will use them directly; otherwise, the reproduction plan above can be executed in P2.

---

## 6. Artifacts

### 6.1 Code

| File | SHA-256 | Description |
|---|---|---|
| `/tmp/test_codongpt_inference.py` | (uploaded) | Inference test script |
| `external_tools/codonGPT_hf_ee7017c4/` | (see Section 2.2) | Pretrained checkpoint (read-only) |

### 6.2 Results

| Cargo | Prefix length | Codon changes | Synonymous | Elapsed |
|---|---|---|---|---|
| ACTB | 20 aa | 17/20 (85%) | ✅ | 15.10s |
| HLA-A | 20 aa | 11/20 (55%) | ✅ | 16.79s |

### 6.3 Environment

- **Python**: 3.10 (conda env `codongpt`)
- **transformers**: 4.57.6
- **biopython**: 1.87 (installed during test)
- **torch**: (inherited from codongpt env)
- **Device**: CPU

---

## 7. Conclusion

P1-09 is **partially complete**:
- ✅ Pretrained checkpoint verified (loads, inference works, synonymous action space confirmed).
- ❌ RL training not reproduced (public notebooks not accessible).

Per the degradation path, this is acceptable: the pretrained checkpoint is sufficient for use as a SOTA baseline (A3 or A4 in the P1-13 panel). The RL reproduction is deferred to P2, pending either (a) access to the public notebooks, or (b) execution of the reproduction plan in Section 5.

---

**End of report.**
