#!/bin/bash
# P0-1 V5 polyA 3-seed — OOM retry for seed 20260922 (attempt 1, 2026-09-28 01:48Z).
#
# What happened: both seeds raced onto GPU4 inside the shared-GPU watcher (2 x V5
# + external users on one 40GB card => CUDA OOM). Seed 20260921 survived; seed
# 20260922 died with a proper failure.json. This retry (a) archives the failed
# attempt directory (evidence preserved, append-only outputs untouched), then
# (b) relaunches the SAME frozen runner/seed on a different card with an
# independent free-VRAM gate. No prereg change: seed, config, runner, output
# schema are identical.
set -u
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
WT=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
RUNNER=$WT/scripts/route_a_v3/run_route2_xeditcritic_v5_polya_3seed_v1.py
LOGDIR=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v5_polya_3seed
SEED=20260922
EXCLUDE_GPU=4          # seed 20260921 is training there
REQUIRED_FREE=10000    # MiB; V5 peak measured 8.3GB -> 10GB gate with margin
TS=$(date -u +%Y%m%dT%H%M%SZ)

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [retry] $1" | tee -a "$LOGDIR/launch_watcher.log"; }
emit() {
  "$PY" - "$1" "$2" "$3" <<'PYEOF'
import json, sys, datetime
status, phase, note = sys.argv[1], sys.argv[2], sys.argv[3]
out = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v5_polya_3seed/watcher_heartbeat.json"
try:
    cur = json.load(open(out))
except Exception:
    cur = {}
cur.update({"status": status, "phase": phase, "note": note,
            "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "schema_version": "route_a_v3_v5_polya_3seed_watcher_heartbeat.v1"})
open(out, "w").write(json.dumps(cur, indent=1, sort_keys=True))
print("watcher heartbeat:", status, "|", phase)
PYEOF
}

# 1) archive the failed attempt (evidence preserved) --------------------------
if [ -d "$LOGDIR/seed_$SEED" ]; then
  mv "$LOGDIR/seed_$SEED" "$LOGDIR/seed_${SEED}_failed_oom_attempt1_${TS}"
  log "archived failed attempt -> seed_${SEED}_failed_oom_attempt1_${TS}"
fi

# 2) pick a card (excluding GPU4) and relaunch with grace re-check ------------
while true; do
  BEST=-1; BESTFREE=-1
  for i in 0 1 2 3 5; do
    [ "$i" = "$EXCLUDE_GPU" ] && continue
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" | tr -d ' ')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$i" | tr -d ' ')
    free=$(( ${total:-0} - ${used:-99999} ))
    if [ "$free" -gt "$BESTFREE" ]; then BEST=$i; BESTFREE=$free; fi
  done
  if [ "$BESTFREE" -ge "$REQUIRED_FREE" ]; then
    log "retry target GPU$BEST (free ${BESTFREE}MiB); 120s grace re-check"
    emit LAUNCHING "SEED_${SEED}_RETRY" "GPU$BEST free ${BESTFREE}MiB; grace window"
    sleep 120
    used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$BEST" | tr -d ' ')
    total2=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$BEST" | tr -d ' ')
    free2=$(( ${total2:-0} - ${used2:-99999} ))
    if [ "$free2" -ge "$REQUIRED_FREE" ]; then
      cd "$WT"
      nohup "$PY" -u "$RUNNER" --seed "$SEED" --physical-gpu-index "$BEST" \
        > "$LOGDIR/polya_3seed_seed${SEED}_attempt2.log" 2>&1 &
      echo $! > "$LOGDIR/polya_3seed_seed${SEED}_pid.txt"
      log "RELAUNCH attempt2 seed=$SEED pid=$! on GPU$BEST (grace free ${free2}MiB)"
      emit RUNNING "SEED_${SEED}" "attempt2 launched on GPU$BEST (pid $!)"
      break
    fi
    log "grace re-check failed (free ${free2}MiB < ${REQUIRED_FREE}); re-queue"
  else
    log "no eligible card >=${REQUIRED_FREE}MiB free (best GPU$BEST=${BESTFREE}); retry in 300s"
    emit WAITING "SEED_${SEED}_RETRY_QUEUE" "no card >=${REQUIRED_FREE}MiB free; polling 300s"
    sleep 300
  fi
done
log "retry watcher exit (seed $SEED attempt2 launched)"