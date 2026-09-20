#!/bin/bash
# Task 3.4 M1 intervention arm: queue-then-launch watcher (W0 discipline: full idle card on GPU 0-5 only)
# Reads frozen prereg docs/paper/m1_intervention_arm_amendment_v1.md. No gate edits after launch.
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
WT=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
OUT=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_m1_intervention/seed_20260920
LOGDIR=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_m1_intervention
mkdir -p "$OUT" "$LOGDIR"

emit_wait() {
  python3 - "$1" <<'PYEOF'
import json, sys, datetime
status = sys.argv[1]
msg = {
 "WAITING": "GPU queue: no idle full card on GPU 0-5; watcher polling every 600s",
 "LAUNCHING": "idle full card detected; handing off to training runner"
}[status]
out = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_m1_intervention/seed_20260920/heartbeat.json"
try:
    cur = json.load(open(out))
except Exception:
    cur = {}
cur.update({"status": status, "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": msg, "schema_version": "route_a_v3_m1_intervention_heartbeat.v1"})
open(out, "w").write(json.dumps(cur, indent=1, sort_keys=True))
print("heartbeat:", cur["status"])
PYEOF
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watcher start (task 3.4 M1 intervention, seed 20260920)" | tee -a "$LOGDIR/launch_watcher.log"
emit_wait WAITING

while true; do
  picked=-1
  for i in 0 1 2 3 4 5; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" | tr -d ' ')
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" | grep -c .)
    if [ "${used:-40960}" -lt 2000 ] && [ "${procs:-1}" -eq 0 ]; then
      picked=$i
      break
    fi
  done
  if [ "$picked" -ge 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) idle full card: GPU$picked -> launching" | tee -a "$LOGDIR/launch_watcher.log"
    emit_wait LAUNCHING
    sleep 120
    used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$picked" | tr -d ' ')
    if [ "${used2:-40960}" -ge 2000 ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU$picked re-occupied during grace window; returning to WAITING" | tee -a "$LOGDIR/launch_watcher.log"
      emit_wait WAITING
      continue
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) final check passed; exec training on GPU$picked" | tee -a "$LOGDIR/launch_watcher.log"
    cd "$WT"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) LAUNCH gpu=$picked seed=20260920" >> "$LOGDIR/launch_watcher.log"
    nohup "$PY" -u scripts/route_a_v3/run_route2_m1_intervention_fullft_v1.py --physical-gpu-index "$picked" \
      > "$LOGDIR/m1_intervention_seed20260920.log" 2>&1 &
    echo "$!" > "$OUT/training_pid.txt"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launched pid=$(cat "$OUT/training_pid.txt") on GPU$picked" | tee -a "$LOGDIR/launch_watcher.log"
    break
  fi
  sleep 600
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watcher exit" | tee -a "$LOGDIR/launch_watcher.log"
