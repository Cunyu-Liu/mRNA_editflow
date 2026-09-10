#!/usr/bin/env bash
# D2 watcher: when a full A100 (0..5) is free (>=12GB free, util<40%, no
# project pid), runs the smoke (8 src) then the calib100 arm A (pure
# retrieval, beta sweep 0.25/0.5/1/2), then exits.
set -u
W=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901
X=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5
D=$X/d2_retrieval_20260910
LOG=$D/watcher.log
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
FREE_GB=12
DONE=$D/done_d2_calib.json

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

gpu_free() {
    local idx=$1
    local used util
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ -n "$util" ] && [ "$used" -le $((40960 - FREE_GB * 1024)) ] && [ "$util" -lt 40 ]
}

gpu_has_project_pid() {
    local idx=$1 pid owner
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$idx" 2>/dev/null | tr -d ' '); do
        [ -z "$pid" ] && continue
        owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        [ "$owner" = "cunyuliu" ] && return 0
    done
    return 1
}

log "D2 watcher started (pid $$)"
# 1) smoke first (reuse if terminal)
if [ ! -f "$D/smoke8/guided_run_summary.json" ]; then
    log "waiting for a free GPU for smoke"
    while true; do
        for idx in 0 1 2 3 4 5; do
            if gpu_free "$idx" && ! gpu_has_project_pid "$idx"; then
                log "smoke on GPU$idx"
                cd "$W" && $PY scripts/route_a_v3/run_route2_guided_xeditsetflow_v5_v1.py \
                    --config configs/route_a_v3_route2_xeditsetflow_v5_screen_v1.json \
                    --run-id b_fix2 --checkpoint-pass 2 --physical-gpu-index "$idx" \
                    --output-dir "$D/smoke8" \
                    --screen-gate "$X/screen_20260915/screen_gate.json" \
                    --arms guided --beta 1.0 --critic-kind retrieval --source-limit 8 \
                    >> "$D/smoke8.log" 2>&1
                log "smoke rc=$?"
                break 2
            fi
        done
        sleep 300
    done
fi
if [ -f "$DONE" ]; then log "done marker present; exiting"; exit 0; fi

# 2) calib100 arm A: beta sweep, serial
for BETA in 0.25 0.5 1 2; do
    OUT="$D/calib100_retr_beta_${BETA}"
    if [ -f "$OUT/guided_run_summary.json" ]; then
        log "beta $BETA already terminal; skip"
        continue
    fi
    log "waiting GPU for calib beta=$BETA"
    while true; do
        for idx in 0 1 2 3 4 5; do
            if gpu_free "$idx" && ! gpu_has_project_pid "$idx"; then
                log "calib beta=$BETA on GPU$idx"
                cd "$W" && $PY scripts/route_a_v3/run_route2_guided_xeditsetflow_v5_v1.py \
                    --config configs/route_a_v3_route2_xeditsetflow_v5_screen_v1.json \
                    --run-id b_fix2 --checkpoint-pass 2 --physical-gpu-index "$idx" \
                    --output-dir "$OUT" \
                    --screen-gate "$X/screen_20260915/screen_gate.json" \
                    --arms guided --beta "$BETA" --critic-kind retrieval \
                    --source-subset-file "$X/beta_sweep_20260908/calib100_keys.txt" \
                    >> "$D/calib_beta_${BETA}.log" 2>&1
                log "calib beta=$BETA rc=$?"
                break 2
            fi
        done
        sleep 300
    done
done
echo "{\"d2_calib_complete\":true,\"timestamp\":\"$(date -Iseconds)\"}" > "$DONE"
log "D2 calib100 arm A complete; exiting"
