#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# phaseA_a3_a5_watcher.sh — Low-peak GPU watcher for SetFlow V5 Phase A.
#
# On a shared A100 cluster (load ~94, 64 users, six full cards often busy, GPU
# 6/7 are MIG slices) this daemon polls nvidia-smi every 5 minutes and, when a
# genuinely FREE full card (GPU 0..5) appears, runs the sequence:
#
#   1) A3   : full-891 critic mixed-pool probe (a3_critic_mixed_pool_probe.py)
#             on --physical-gpu-index <GPU>
#   2) A5   : b_fix2 snapshot-stability, the two post-baseline saved passes
#             (pass 4 and pass 6) — pass 2 is the baseline kept from existing
#             frozen data, never re-run.
#
# FREE = Free-or-estimated memory >= 15 GB AND util < 30% AND no in-flight
# mrna-editflow / setflow compute PID on that card (conservative; NEVER grabs a
# card hosting project training). GPU 6/7 (MIG) are always excluded.
#
# Each step writes a done-marker JSON into analysis_phaseA_20260907/ logging
# timestamp, GPU and exit code.  A step is skipped once its marker reports
# exit_code == 0.  When every step is complete the watcher exits 0.
#
# Run (background):
#   nohup bash monitor/phaseA_a3_a5_watcher.sh >> monitor/phaseA_a3_a5_watcher.log 2>&1 &
# ---------------------------------------------------------------------------
set -u

WORKTREE=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901
ANALYSIS="$WORKTREE/analysis_phaseA_20260907"
MONITOR="$WORKTREE/monitor"
PY=python3

FREE_MEM_GB=15
FREE_MEM_MB=$((FREE_MEM_GB * 1024))
UTIL_MAX=30
POLL_SECONDS=300

log() { echo "[$(date '+%F %T')] $*"; }

done_marker() { echo "$ANALYSIS/done_$1.json"; }

step_ok() {
    local m="$(done_marker "$1")"
    [ -f "$m" ] || return 1
    local rc
    rc="$($PY -c "import json,sys;print(json.load(open('$m')).get('exit_code',1))" 2>/dev/null || echo 1)"
    [ "$rc" = "0" ]
}

write_marker() {
    local step="$1" gpu="$2" rc="$3"
    echo "{\"step\":\"$step\",\"gpu\":$gpu,\"timestamp\":\"$(date -Iseconds)\",\"exit_code\":$rc}" \
        > "$(done_marker "$step")"
}

# Project (mrna-editflow/setflow) process ids currently residing on GPU <idx>.
# We treat a card as busy if ANY of its resident compute PIDs is one of OUR
# OWN (user cunyuliu) python compute jobs (training / validation / generation /
# critic / p-phase orchestrators such as p4_train.py).  This conservatively and
# reliably keeps us from ever preempting our own in-flight work on a card —
# regardless of what the subprocess is named.
gpu_has_project_pid() {
    local idx="$1" pid owner cmd
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$idx" 2>/dev/null | tr -d ' '); do
        [ -z "$pid" ] && continue
        owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        [ "$owner" = "cunyuliu" ] || continue
        cmd="$(ps -o command= -p "$pid" 2>/dev/null | tr -d ' ')"
        case "$cmd" in
            python|python3|/usr/bin/python*|/home/*/python*)
                return 0 ;;
        esac
    done
    return 1
}

# Echo "<idx> <free_mb> <util>" for the first free full GPU (0..5), else fail.
find_free_gpu() {
    local idx util mem
    for idx in $(seq 0 5); do
        util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
        mem="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')"
        [ -z "${mem:-}" ] && mem=0
        [ -z "${util:-}" ] && util=100
        if [ "$mem" -ge "$FREE_MEM_MB" ] && [ "$util" -lt "$UTIL_MAX" ] \
           && ! gpu_has_project_pid "$idx"; then
            # confirm the card is full (non-MIG resumes to full memory) — cheap guard
            echo "$idx $mem $util"
            return 0
        fi
    done
    return 1
}

run_a3() {
    local gpu="$1" rc
    log "A3_FULL  : launch full-891 critic probe on GPU $gpu"
    "$PY" "$ANALYSIS/a3_critic_mixed_pool_probe.py" \
        --output "$ANALYSIS/a3_critic_mixed_pool_probe_full.json" \
        --physical-gpu-index "$gpu"
    rc=$?
    write_marker A3_FULL "$gpu" "$rc"
    log "A3_FULL  : done rc=$rc (marker $(done_marker A3_FULL))"
    return "$rc"
}

run_a5() {
    local gpu="$1" pass="$2" rc
    log "A5_PASS$pass : launch b_fix2 snapshot-stability (pass $pass) GPU=$gpu"
    "$PY" "$ANALYSIS/a5_snapshot_stability.py" \
        --output "$ANALYSIS/a5_snapshot_stability.json" \
        --pass "$pass" \
        --physical-gpu-index "$gpu" \
        --allow-regeneration \
        --scratch-dir "$ANALYSIS/scratch_a5"
    rc=$?
    write_marker "A5_PASS$pass" "$gpu" "$rc"
    log "A5_PASS$pass : done rc=$rc (marker $(done_marker A5_PASS$pass))"
    return "$rc"
}

main() {
    mkdir -p "$MONITOR" "$ANALYSIS"
    log "phaseA_a3_a5_watcher starting (worktree=$WORKTREE, poll=${POLL_SECONDS}s)"
    log "saved checkpoints: b_fix2 pass {2 baseline,4,6}; A3 full 891 then A5 pass4/6"

    while :; do
        if step_ok A3_FULL && step_ok A5_PASS4 && step_ok A5_PASS6; then
            log "all steps complete; watcher exiting 0"
            exit 0
        fi

        if ! free_info="$(find_free_gpu)"; then
            log "no free full GPU (all busy / MIG-only); sleep ${POLL_SECONDS}s"
            sleep "$POLL_SECONDS"
            continue
        fi

        gpu="${free_info%% *}"
        log "free GPU found idx=$gpu (free_mem_mb/util): $free_info"

        if ! step_ok A3_FULL; then
            run_a3 "$gpu"
        fi

        # A5 only after A3 has succeeded (task: A3 -> then A5).
        if step_ok A3_FULL; then
            if ! step_ok A5_PASS4; then run_a5 "$gpu" 4; fi
            if ! step_ok A5_PASS6; then run_a5 "$gpu" 6; fi
        else
            log "A3 not successful yet; A5 deferred to a later free window"
        fi

        sleep "$POLL_SECONDS"
    done
}

main