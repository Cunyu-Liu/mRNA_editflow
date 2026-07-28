# GSE200304 D1 pipeline

The production extractor joins the official construct and processed-label
tables only by exact `construct.merged_id == labels.Barcode`, then pairs IDs by
removing terminal `_WT` or `_Mutant`. It keeps 6,120 both-labelled pairs
eligible for measured evaluation and separately audits one-sided, unlabelled,
and control constructs. Deposited `Freq` is `PROVIDED_LABEL_ONLY`.
