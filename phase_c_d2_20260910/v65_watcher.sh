#!/usr/bin/env bash
# V6.5 explore-then-select watcher:
#   1) B512 arm, B1024 arm (unguided, trajectory-count override)
#   2) B256 seed replicas (seed 20260916 / 20260917)
#   3) final harvest: all gates (amendment v2) + journal + commit
set -u
W=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901
X=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5
D=$X/v65_explore_select_20260910
LOG=$D/watcher.log
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
FREE_GB=12
DONE=$D/done_v65.json
mkdir -p "$D"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

gpu_ok() {
    local idx=$1 used util
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -le $((40960 - FREE_GB * 1024)) ] && [ "$util" -lt 40 ]
}
has_project_pid() {
    local idx=$1 pid owner
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$idx" 2>/dev/null | tr -d ' '); do
        [ -z "$pid" ] && continue
        owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        [ "$owner" = "cunyuliu" ] && return 0
    done
    return 1
}

run_arm() {
    local tag=$1 traj=$2 seed=$3
    local OUT="$D/${tag}"
    [ -f "$OUT/guided_run_summary.json" ] && { log "$tag terminal; skip"; return 0; }
    log "waiting GPU for $tag"
    while true; do
        for idx in 0 1 2 3 4 5; do
            if gpu_ok "$idx" && ! has_project_pid "$idx"; then
                log "$tag on GPU$idx"
                cd "$W" && $PY scripts/route_a_v3/run_route2_guided_xeditsetflow_v5_v1.py \
                    --config configs/route_a_v3_route2_xeditsetflow_v5_screen_v1.json \
                    --run-id b_fix2 --checkpoint-pass 2 --physical-gpu-index "$idx" \
                    --output-dir "$OUT" \
                    --screen-gate "$X/screen_20260915/screen_gate.json" \
                    --arms unguided --trajectory-count "$traj" \
                    ${seed:+--decoder-seed-base "$seed"} \
                    >> "$D/${tag}.log" 2>&1
                log "$tag rc=$?"
                return 0
            fi
        done
        sleep 300
    done
}

log "V6.5 watcher started (pid $$)"
run_arm B512_unguided 512 ""
run_arm B1024_unguided 1024 ""
run_arm B256_seed16 256 2026091601
run_arm B256_seed17 256 2026091701

log "all arms done; harvesting"
$PY /tmp/v65_harvest.py >> "$LOG" 2>&1
rc=$?
cat >> /home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902/docs/training_journal/TRAINING_LOG_202609.md <<EOF

### 批次六十七（自动 V6.5 收割：全臂终态，\`$(date '+%F %T')\`）

- v65 watcher 全臂终态（B512/B1024/seed 副本×2）→ 自动执行 v65_harvest.py（rc=$rc）：产物 \`v65_explore_select_20260910/v65_harvest.json\`（amendment v2 全门：B2 Δ门 3-seed 稳健 + B512/1024 边际曲线 + B2-I 独立口径 + B3 绝对线）。
- 待会话复核终判。
EOF
J=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
(cd "$J" && git add docs/training_journal/TRAINING_LOG_202609.md && git commit -q -m "train journal: auto batch - V6.5 explore-select harvest rc=$rc" && git push -q origin route-a-v3-w0-diagnosis-20260902) >> "$LOG" 2>&1
(cd "$W" && git add phase_c_d2_20260910/ 2>/dev/null; git commit -q -m "analysis(route2): V6.5 harvest products" 2>/dev/null; git push -q origin route-a-v3-setflow-v5-base-fix-20260901) >> "$LOG" 2>&1
echo "{\"harvest_rc\":$rc,\"timestamp\":\"$(date -Iseconds)\"}" > "$DONE"
log "V6.5 complete; exiting"
exit $rc
