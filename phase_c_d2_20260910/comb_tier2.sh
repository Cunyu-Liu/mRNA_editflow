#!/usr/bin/env bash
# COMB tier-2: 891 full D arm + 2 seed replicas, then three-calibre harvest.
set -u
W=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901
X=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5
D=$X/comb_mechanism_20260911
LOG=$D/tier2_watcher.log
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
FREE_GB=12
DONE=$D/done_tier2.json

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
    local outdir=$1; shift
    [ -f "$outdir/guided_run_summary.json" ] && { log "skip terminal: $outdir"; return 0; }
    log "waiting GPU: $outdir"
    while true; do
        for idx in 0 1 2 3 4 5; do
            if gpu_ok "$idx" && ! has_project_pid "$idx"; then
                log "launch GPU$idx: $outdir"
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

log "tier-2 watcher started (pid $$)"
# main arm (seed base default 2026091501), then replicas with distinct seed bases
run_arm "$D/D891_main" --arms guided --beta 0.25 --critic-kind v5 --trajectory-count 256
run_arm "$D/D891_seed16" --arms guided --beta 0.25 --critic-kind v5 --trajectory-count 256 --decoder-seed-base 2026091601
run_arm "$D/D891_seed17" --arms guided --beta 0.25 --critic-kind v5 --trajectory-count 256 --decoder-seed-base 2026091701
echo "{\"tier2_done\":true,\"timestamp\":\"$(date -Iseconds)\"}" > "$DONE"
log "tier-2 all arms complete; exiting (harvest on session review)"
