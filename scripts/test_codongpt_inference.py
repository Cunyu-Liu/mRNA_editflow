"""P1-09: CodonGPT checkpoint load + inference test on ACTB / HLA-A.

Verifies the pretrained CodonGPT checkpoint loads correctly and can
generate synonymous codon sequences for public cargos.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

CODONGPT_DIR = Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow/external_tools/codonGPT_hf_ee7017c4")
sys.path.insert(0, str(CODONGPT_DIR))


def main() -> int:
    print(f"[codongpt] Checkpoint dir: {CODONGPT_DIR}")
    print(f"[codongpt] Files: {sorted(os.listdir(CODONGPT_DIR))}")

    # Load tokenizer.
    from tokenizer import CodonTokenizer
    tok = CodonTokenizer.from_pretrained(str(CODONGPT_DIR))
    print(f"[codongpt] Tokenizer loaded. vocab_size={tok.vocab_size}, bos={tok.bos_token_id}, eos={tok.eos_token_id}")

    # Load model.
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(CODONGPT_DIR),
        torch_dtype=torch.float32,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[codongpt] Model loaded. params={n_params/1e6:.1f}M, dtype={next(model.parameters()).dtype}")

    # Load synonymous logit processor.
    from synonymous_logit_processor import (
        SynonymMaskingLogitsProcessor,
        aa_to_codon_human,
        generate_candidate_codons_with_generate,
    )
    print(f"[codongpt] SynonymMaskingLogitsProcessor loaded. aa_to_codon keys: {len(aa_to_codon_human)}")

    # Test cargos: ACTB (1131 nt CDS, 376 aa) and HLA-A (1083 nt CDS, 360 aa).
    # Use a short prefix for the test (first 20 codons).
    cargos = {
        "ACTB_prefix20": "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPTLLTEAPLNPKANREKMTQIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAAASSSSLEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHRKCF",
        "HLA_A_prefix20": "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRW",
    }

    # Use first 20 codons (60 aa) for quick test.
    for cargo_name, full_protein in cargos.items():
        protein_prefix = full_protein[:20]  # first 20 aa
        print(f"\n[codongpt] Cargo: {cargo_name}")
        print(f"[codongpt]   protein prefix (20 aa): {protein_prefix}")

        # Initial codons: use the first synonymous codon for each aa (simplest choice).
        initial_codons = [aa_to_codon_human[aa][0] for aa in protein_prefix]
        print(f"[codongpt]   initial codons: {' '.join(initial_codons)}")

        # Generate optimized codons with the model.
        t0 = time.time()
        try:
            optimized = generate_candidate_codons_with_generate(
                initial_codons=initial_codons,
                temperature=1.0,
                model=model,
                tokenizer=tok,
            )
            elapsed = time.time() - t0
            print(f"[codongpt]   optimized codons: {' '.join(optimized)}")
            print(f"[codongpt]   elapsed: {elapsed:.2f}s")

            # Verify synonymous.
            from Bio.Seq import Seq
            initial_aa = str(Seq("".join(initial_codons)).translate())
            optimized_aa = str(Seq("".join(optimized)).translate())
            print(f"[codongpt]   initial aa:   {initial_aa}")
            print(f"[codongpt]   optimized aa: {optimized_aa}")
            print(f"[codongpt]   synonymous: {initial_aa == optimized_aa}")

            # Count codon changes.
            n_changes = sum(1 for a, b in zip(initial_codons, optimized) if a != b)
            print(f"[codongpt]   codon changes: {n_changes}/{len(initial_codons)}")
        except Exception as e:
            print(f"[codongpt]   ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n[codongpt] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
