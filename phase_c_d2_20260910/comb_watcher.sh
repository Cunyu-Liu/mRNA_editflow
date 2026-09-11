#!/usr/bin/env bash
# COMB mechanism experiment watcher:
#   tier 1: D arm (guided beta 0.25, B=256) on calib100
#   (positive direction -> tier 2: 891 full + seed replicas x2, per prereg)
set -u
W=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901
X=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5
D=$X/comb_mechanism_20260911
LOG=$D/watcher.log
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
FREE_GB=12
DONE1=$D/done_tier1.json
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
run_on_free_gpu() {
    local outdir=$1; shift
    while true; do
        for idx in 0 1 2 3 4 5; do
            if gpu_ok "$idx" && ! has_project_pid "$idx"; then
                log "arm on GPU$idx: $outdir"
                cd "$W" && $PY scripts/route_a_v3/run_route2_guided_xeditsetflow_v5_v1.py \
                    --config configs/route_a_v3_route2_xeditsetflow_v5_screen_v1.json \
                    --run-id b_fix2 --checkpoint-pass 2 --physical-gpu-index "$idx" \
                    --output-dir "$outdir" \
                    --screen-gate "$X/screen_20260915/screen_gate.json" \
                    "$@" >> "$D/$(basename "$outdir").log" 2>&1
                log "rc=$? : $outdir"
                return 0
            fi
        done
        sleep 300
    done
}

log "COMB watcher started (pid $$)"

# ---- tier 1: D arm calib100 ----
if [ ! -f "$D/calib100_D_comb/guided_run_summary.json" ] && [ ! -f "$DONE1" ]; then
    run_on_free_gpu "$D/calib100_D_comb" \
        --arms guided --beta 0.25 --critic-kind v5 --trajectory-count 256 \
        --source-subset-file "$X/beta_sweep_20260908/calib100_keys.txt"
fi
echo "{\"tier1_done\":true,\"timestamp\":\"$(date -Iseconds)\"}" > "$DONE1"
log "tier 1 complete; tier-2 gating deferred to session review (prereg: positive direction -> 891 + 3 seeds)"
