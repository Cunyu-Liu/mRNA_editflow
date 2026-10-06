#!/usr/bin/env bash
# Batch-4 (amendment v3) monitor: status + auto-summary when both jobs complete.
set -u
E=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments
D2=$E/analysis_generalist_rows_v2/batch4_newrows_v1_gpu2
D4=$E/analysis_generalist_rows_v2/batch4_newrows_v1_gpu4
LOG=$E/analysis_generalist_rows_v2/batch4_launch_20261006.log
echo "===== batch4 check $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
alive=$(pgrep -fc "run_route2_frozen_delta_generalist_newrows_v1" || true)
echo "alive processes: $alive"
for D in $D2 $D4; do
  if [ -d "$D" ]; then
    n=$(find "$D" -name run_detail.json | wc -l)
    echo "$D: $n/16 run_detail files"
    ls "$D" | grep -c "__" >/dev/null 2>&1 || true
  else
    echo "$D: not started"
  fi
done
if [ "$alive" -eq 0 ] && [ -d "$D2" ] && [ -d "$D4" ]; then
  n2=$(find "$D2" -name run_detail.json | wc -l)
  n4=$(find "$D4" -name run_detail.json | wc -l)
  total=$((n2 + n4))
  echo "jobs finished: $total run_detail files"
  if [ "$total" -ge 28 ] && [ ! -f "$E/analysis_generalist_rows_v2/batch4_newrows_matrix_v1.json" ]; then
    PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
    $PY - <<PYINNER
import json, glob
rows = {}
for f in sorted(glob.glob("$D2/*__*/run_detail.json") + glob.glob("$D4/*__*/run_detail.json")):
    j = json.load(open(f))
    tag = f.split("/")[-2]
    task, model = tag.rsplit("__", 1)
    rows.setdefault(model, {})[task] = j.get("task_macro_spearman")
out = {
    "artifact": "amendment v3 batch-4 reporting-only matrix (11 backbones x 4 new-row columns scheduled; this file covers launched jobs)",
    "rows": rows,
    "note": "probe = same-endpoint task TRAIN fit; eval = NEW_EVAL_ROW full surface; reporting-only, append-only",
}
json.dump(out, open("$E/analysis_generalist_rows_v2/batch4_newrows_matrix_v1.json", "w"), indent=1)
print("matrix written:", len(rows), "models")
PYINNER
  fi
fi
tail -2 "$LOG" 2>/dev/null | grep -v Warning || true
echo "===== end ====="
