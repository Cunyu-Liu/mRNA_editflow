# Flow-loss correctness note

The intended Stage A path is:

`make_hybrid_batch -> one bridge draw z_t -> rm_gap_tokens_with_aux -> forward -> edit_flow_loss`

`z_t` is one Monte Carlo draw from the conditional bridge for the batch.  The
same draw must be used for token removal, the model forward pass, and all edit
loss terms.  Drawing a second bridge sample inside one loss call would make the
model inputs and supervision refer to different stochastic states.

The source snapshot reviewed for P0 already called `sample_cond_pt` exactly
once in `_flow_batch_loss`; the earlier duplicate-call roadmap observation was
stale.  A deterministic regression test now protects that invariant and checks
loss and gradient reproducibility on CPU after resetting model and RNG state.
