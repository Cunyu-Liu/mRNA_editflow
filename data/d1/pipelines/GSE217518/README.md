# GSE217518 D1 pipeline

The official Figure4 `SHdiNT_U3.csv` and `SHdiNT_U5.csv` tables are paired by
removing terminal `_(Ref|Mut)_[0-9]+` from `seqName`; only exactly one Ref and
one Mut endpoint are admitted. Boundary removal follows observed table
content: U3 fixed suffix 20 nt, U5 fixed prefix 20 nt plus suffix 13 nt.
Raw oligos remain preserved and canonical inserts are stored separately.

Half-life values are provided processed labels. Canonical edit actions describe
endpoint differences, not observed edit order.
