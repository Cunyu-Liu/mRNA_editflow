#!/bin/bash
# P0-1 V5 polyA 3-seed supplement: serialized queue-then-launch watcher.
# Modeled on launch_m1_intervention_watcher.sh with the serialization rule:
# Phase A - poll the M1 intervention heartbeat until M1 training has LAUNCHED
#           (status in TRAINING/EPOCH_DONE/DONE, i.e. M1 watcher grabbed its
#           card); until then we never compete for GPU 0-5.
# Phase B - after M1 is running, poll GPU 0-5 for an idle full card every 600s
#           (mem < 2000 MiB AND zero compute processes), 120s grace re-check.
# Phase C - launch seed 20260921 training, wait for terminal state
#           (heartbeat DONE/FAILED or pid gone), then Phase B again for seed
#           20260922. Strictly serial: only one polyA seed training at a time.
# Prereg: docs/paper/polya_3seed_mini_prereg_v1.md (frozen before launch).
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
WT=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
RUNNER=$WT/scripts/route_a_v3/run_route2_xeditcritic_v5_polya_3seed_v1.py
M1_HEARTBEAT=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_m1_intervention/seed_20260920/heartbeat.json
LOGDIR=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v5_polya_3seed
SEEDS=(20260921 20260922)
mkdir -p "$LOGDIR"

emit() {
  python3 - "$1" "$2" "$3" <<'PYEOF'
import json, sys, datetime
status, phase, note = sys.argv[1], sys.argv[2], sys.argv[3]
out = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v5_polya_3seed/watcher_heartbeat.json"
try:
    cur = json.load(open(out))
except Exception:
    cur = {}
cur.update({"status": status, "phase": phase, "note": note, "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "schema_version": "route_a_v3_v5_polya_3seed_watcher_heartbeat.v1"})
open(out, "w").write(json.dumps(cur, indent=1, sort_keys=True))
print("watcher heartbeat:", status, "|", phase)
PYEOF
}

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" | tee -a "$LOGDIR/launch_watcher.log"; }

m1_status() {
  python3 - "$M1_HEARTBEAT" <<'PYEOF'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("status", "ABSENT"))
except Exception:
    print("ABSENT")
PYEOF
}

m1_launched() {
  s=$(m1_status)
  case "$s" in
    TRAINING|EPOCH_DONE|DONE|FAILED) return 0 ;;
    *) return 1 ;;
  esac
}

seed_terminal() {
  local seed=$1
  local hb="$LOGDIR/seed_${seed}/heartbeat.json"
  local pidfile="$LOGDIR/seed_${seed}/training_pid.txt"
  if [ -f "$hb" ]; then
    local st
    st=$(python3 -c "import json;print(json.load(open('$hb')).get('status',''))" 2>/dev/null)
    case "$st" in
      DONE|FAILED) return 0 ;;
    esac
  fi
  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile")
    if ! kill -0 "$pid" 2>/dev/null; then
      log "seed $seed pid $pid no longer alive without terminal heartbeat - treating as terminal"
      return 0
    fi
  fi
  return 1
}

log "watcher start (P0-1 V5 polyA 3-seed, seeds ${SEEDS[*]}, serialized after M1)"
emit WAITING PHASE_M1_GATE "M1 intervention watcher (pid 3494364) still queuing; polyA watcher will not touch GPU 0-5 until M1 training has launched"

# ---- Phase A: wait until M1 training has actually launched ----------------
until m1_launched; do
  sleep 600
  emit WAITING PHASE_M1_GATE "M1 heartbeat=$(m1_status); holding off GPU 0-5"
done
log "M1 training launched (status=$(m1_status)); polyA queue logic now armed"
emit WAITING PHASE_GPU_QUEUE "M1 running; polling GPU 0-5 for idle full card every 600s"

# ---- Phase B/C: per-seed serial launch -------------------------------------
for seed in "${SEEDS[@]}"; do
  if seed_terminal "$seed"; then
    log "seed $seed already terminal; skipping"
    continue
  fi
  if [ -d "$LOGDIR/seed_${seed}" ]; then
    log "seed $seed directory exists but not terminal - skipping (append-only; inspect manually)"
    emit WAITING "SEED_${seed}_SKIPPED" "existing non-terminal seed dir; refusing double launch"
    continue
  fi
  picked=-1
  while [ "$picked" -lt 0 ]; do
    if ! m1_launched; then
      log "M1 left RUNNING state (status=$(m1_status)) before seed $seed launch; re-arming M1 gate"
      until m1_launched; do sleep 600; done
    fi
    for i in 0 1 2 3 4 5; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" | tr -d ' ')
      procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" | grep -c .)
      if [ "${used:-40960}" -lt 2000 ] && [ "${procs:-1}" -eq 0 ]; then
        picked=$i
        break
      fi
    done
    if [ "$picked" -lt 0 ]; then
      sleep 600
      emit WAITING PHASE_GPU_QUEUE "no idle full card on GPU 0-5; seed $seed pending; polling every 600s"
    fi
  done
  log "idle full card: GPU$picked -> launching seed $seed (120s grace re-check)"
  emit LAUNCHING "SEED_${seed}" "idle full card GPU$picked detected; grace window"
  sleep 120
  used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$picked" | tr -d ' ')
  if [ "${used2:-40960}" -ge 2000 ]; then
    log "GPU$picked re-occupied during grace window; returning to WAITING for seed $seed"
    emit WAITING PHASE_GPU_QUEUE "GPU$picked re-occupied during grace; seed $seed back to queue"
    picked=-1
    while [ "$picked" -lt 0 ]; do
      sleep 600
      for i in 0 1 2 3 4 5; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" | tr -d ' ')
        procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$i" | grep -c .)
        if [ "${used:-40960}" -lt 2000 ] && [ "${procs:-1}" -eq 0 ]; then
          picked=$i
          break
        fi
      done
      if [ "$picked" -lt 0 ]; then
        emit WAITING PHASE_GPU_QUEUE "still no idle card; seed $seed pending"
      fi
    done
    log "re-acquired GPU$picked after grace re-check for seed $seed"
  fi
  mkdir -p "$LOGDIR/seed_${seed}"
  cd "$WT"
  log "LAUNCH gpu=$picked seed=$seed"
  nohup "$PY" -u "$RUNNER" --seed "$seed" --physical-gpu-index "$picked" \
    > "$LOGDIR/polya_3seed_seed${seed}.log" 2>&1 &
  echo "$!" > "$LOGDIR/seed_${seed}/training_pid.txt"
  log "launched seed $seed pid=$(cat "$LOGDIR/seed_${seed}/training_pid.txt") on GPU$picked"
  emit RUNNING "SEED_${seed}" "training running on GPU$picked (pid $(cat "$LOGDIR/seed_${seed}/training_pid.txt"))"

  # ---- wait for terminal state of this seed before queuing the next ----
  while ! seed_terminal "$seed"; do
    sleep 600
    hb="$LOGDIR/seed_${seed}/heartbeat.json"
    st="?"
    [ -f "$hb" ] && st=$(python3 -c "import json;print(json.load(open('$hb')).get('status','?'))" 2>/dev/null)
    emit RUNNING "SEED_${seed}" "training heartbeat status=$st; waiting for terminal"
  done
  hb="$LOGDIR/seed_${seed}/heartbeat.json"
  st="?"
  [ -f "$hb" ] && st=$(python3 -c "import json;print(json.load(open('$hb')).get('status','?'))" 2>/dev/null)
  log "seed $seed terminal (status=$st)"
  emit WAITING "SEED_${seed}_TERMINAL" "seed $seed finished with status=$st; proceeding to next seed"
done

log "all seeds terminal; watcher exit"
emit DONE ALL_SEEDS_TERMINAL "both polyA 3-seed supplement seeds reached terminal state"
