# Autoresearch classic loop: Critic V4 RNG replay recovery

Primary Critic acceptance signals:

- the 170.48M full model passes bit-exact CUDA BF16 RNG replay with activation
  checkpointing and a nonzero finite gradient;
- final-pass Development Validation task-macro Spearman is at least 0.30 and at
  least 0.10 above the matched C0-V4 result;
- task-macro standardized MAE is at most 1.7 and no worse than C0-V4;
- at least eight of nine task correlations are positive and at least six tasks
  beat C0-V4;
- later control completion must satisfy the frozen candidate-information and
  mechanism-ablation margins before a formal screen PASS.

The first iteration repairs technical execution only. Performance is judged
from terminal Development Validation summaries. Development TEST and new final
Evaluation outcomes remain unread during this loop.

