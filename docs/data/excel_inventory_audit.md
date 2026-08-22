# Excel Inventory Audit (D0-01)

> Historical audit retained as a small Git summary. The current producer now
> defaults future generated Parquet output to
> `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/data_registry/excel_inventory.parquet`;
> the output line below records the legacy run and does not authorize migration
> or removal of the currently tracked file.

- input: `data/raw/codonflow_integrated_dataset_catalog_ranked.xlsx`
- input sha256: `4c9dabb961de01d24c4ebf674b99784d5dec612c1bff4fa464d2c208d3b583d8`
- output: `data_registry/excel_inventory.parquet`
- total Excel data rows classified: 184

## Sheet inventory

| sheet | data rows | role |
|---|---|---|
| 模型适配排序 | 78 | model adaptation ranking |
| 数据集资源排序 | 14 | dataset resource ranking |
| Sources | 92 | per-model verification sources |

## Mapping rules (frozen)

- `模型适配排序` -> `MODEL` (model entry: name + paper title + verifiable ID).
- `数据集资源排序` -> `DATABASE` if the resource name contains a known
  database token (RefSeq/Ensembl/GENCODE/NCBI/GTEx/ENCODE/FANTOM5/4DN/
  RNAcentral/Rfam/PDB/RNASolo/CELLxGENE/GEO/HCA/hECA), else `DATASET`.
- `Sources` -> `PAPER` (bibliographic/code verification source).

## Inventory kind counts

| inventory_kind | rows |
|---|---|
| DATASET | 5 |
| MODEL | 78 |
| DATABASE | 9 |
| PAPER | 92 |
| AUXILIARY_RESOURCE | 0 |
| NOT_RELEVANT | 0 |

## Model sheet coverage

- model rows: 78
- rows with Sources verification URL: 78
- adapt_level distribution: A=8, B=33, C=27, D=1, S=9
- category distribution: RNA 序列 LM=21, DNA/多组学=17, RNA 结构模型=17, 单细胞 RNA-seq=16, RNA 序列 LM; RNA 结构模型=4, RNA 序列 LM; DNA/多组学=3

## Resource sheet mapping (all 14 rows)

| excel_row | usage_role | resource | inventory_kind |
|---|---|---|---|
| 2 | 主干 | RefSeq / Ensembl transcript-CDS-protein + GFF/GTF | DATABASE |
| 3 | 剪接监督 | GENCODE canonical transcript / splice annotation | DATABASE |
| 4 | mRNA/CDS 生成主语料 | NCBI mRNA/CDS coding sequences | DATABASE |
| 5 | full-length mRNA 区域数据 | full-length mRNA 区域数据(5'UTR / CDS / 3'UTR) | DATASET |
| 6 | full-length mRNA/UTR/CDS 评测与微调 | mRNABERT downstream datasets (Zenodo) | DATASET |
| 7 | UTR 设计与评测 | 5'UTR / 3'UTR 序列数据 | DATASET |
| 8 | 组织表达条件 | GTEx expression / RNA-seq | DATABASE |
| 9 | 调控 RNA signal 评测 | ENCODE / FANTOM5 / 4DN 中的 RNA 相关 tracks | DATABASE |
| 10 | 病毒/原核 codon prior | SARS-CoV-2 RNA viral genomes | DATASET |
| 11 | RNA prior / 表征预训练 | RNAcentral ncRNA | DATABASE |
| 12 | RNA 家族/同源先验 | Rfam families / MSA | DATABASE |
| 13 | RNA 二级结构评测 | bpRNA / RNAStralign / ArchiveII 二级结构数据 | DATASET |
| 14 | RNA 3D 结构辅助 | PDB RNA / RNASolo 3D 结构 | DATABASE |
| 15 | 细胞/组织条件 | CELLxGENE / GEO / HCA / hECA 等 scRNA-seq 表达语料 | DATABASE |

## Acceptance

| check | status | detail |
|---|---|---|
| 78 model entries mapped | PASS | model rows=78 (expected 78), all kind=MODEL: True |
| 14 resource classes mapped | PASS | resource rows=14 (expected 14), kinds=['DATABASE', 'DATASET'] |
| no unexplained entries | PASS | unexplained=0, invalid_kind=0, empty_name=0, total_rows=184 |
