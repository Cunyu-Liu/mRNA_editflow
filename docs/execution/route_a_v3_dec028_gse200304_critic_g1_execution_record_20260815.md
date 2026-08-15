# DEC028 GSE200304 source-relative critic G1 execution record

Status: `TERMINATED_SAFELY_WITH_EVIDENCE_NO_RETRY`

The one authority-bound G1 execution was launched once on 2026-08-15.  It
passed repository, implementation, SS3 row, SS4 split, Python, PyTorch and
physical-GPU identity checks; read the exact 6,547 private development rows and
6,547 frozen split assignments; constructed one critic and one AdamW optimizer;
and entered the first TRAIN batch.  The reverse encoder forward pass raised
`torch.OutOfMemoryError` before loss, backward or optimizer update.

The host inventory had reported a 40 GiB A100 with about 37 GiB free, while the
bound PyTorch process reported only about 4.75 GiB total visible capacity and
18.81 MiB free at failure.  This discrepancy is recorded as an environment
capacity failure, not a data-gate or scientific-result failure.

Execution counts:

- authorized execution / launched execution: `1 / 1`;
- model constructions: `1`;
- optimizer constructions / fit attempts: `1 / 1`;
- parameter updates: `0`;
- checkpoints: `0`;
- private prediction files: `0`;
- aggregate success results: `0`;
- failure records: `1`;
- retries authorized or executed: `0 / 0`.

The only published file is the aggregate-safe failure record at:

`/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/DEC028_GSE200304_SOURCE_RELATIVE_CRITIC_G1_ONE_RUN_20260815/GSE200304_SOURCE_RELATIVE_CRITIC_G1_FAILURE.json`

It is 1,152 bytes with SHA-256
`5fa222e5417229bb841d412045e824fa8841840512876aae6d37172a673039ab`.
It contains no member payload and keeps the scientific claim
`NOT_ESTABLISHED`.  The exactly-one authority is consumed.  Changing GPU,
batch size, architecture, attention backend or runtime would require a new
prospective successor decision; this contract authorizes no retry.
