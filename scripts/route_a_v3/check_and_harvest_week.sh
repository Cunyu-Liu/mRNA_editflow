#!/bin/bash
# Weekly follow-up monitor + auto-harvest (M1 intervention arm + V5 polyA 3-seed).
# Prints a compact status report; runs the frozen harvest scripts as soon as the
# corresponding training reaches terminal state (append-only outputs; idempotent).
#
# Usage: bash check_and_harvest_week.sh [--no-harvest]
# Exit: 0 always (report script); individual harvest outcomes are printed.

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
WT=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
MNT=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2
# Path overrides exist so the gate/launch logic can be exercised on throwaway fixtures without
# touching the evidence directories. Defaults are the production paths.
M1BASE="${MRNA_MON_M1BASE:-$MNT/experiments/xeditcritic_m1_intervention}"
M1="$M1BASE/seed_20260920"
POLYA="${MRNA_MON_POLYA:-$MNT/experiments/xeditcritic_v5_polya_3seed}"
HARVEST_M1="${MRNA_MON_HARVEST_M1:-$WT/scripts/route_a_v3/harvest_route2_m1_intervention_v1.py}"
HARVEST_M1X="${MRNA_MON_HARVEST_M1X:-$WT/scripts/route_a_v3/harvest_route2_m1_intervention_3seed_v1.py}"
HARVEST_P3="${MRNA_MON_HARVEST_P3:-$WT/scripts/route_a_v3/harvest_route2_v5_polya_3seed_v1.py}"
LAUNCH_EXT="${MRNA_MON_LAUNCHER:-$WT/scripts/route_a_v3/launch_m1_intervention_extension_v2.sh}"
EXT_SEEDS="20260904 20260905"
LOGDIR="$WT/docs/training_journal"

DO_HARVEST=1
[ "$1" = "--no-harvest" ] && DO_HARVEST=0

hb() { # hb <file> <key>
  $PY - "$1" "$2" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get(sys.argv[2], "?"))
except Exception:
    print("ABSENT")
EOF
}

echo "===== mRNA week check $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="

echo "--- M1 intervention arm (seed 20260920) ---"
echo "status=$(hb "$M1/heartbeat.json" status) epoch=$(hb "$M1/heartbeat.json" epoch) step=$(hb "$M1/heartbeat.json" step)/$(hb "$M1/heartbeat.json" total_steps) elapsed_s=$(hb "$M1/heartbeat.json" elapsed_s)"
if [ -f "$M1/training_pid.txt" ]; then
  PID=$(cat "$M1/training_pid.txt")
  if kill -0 "$PID" 2>/dev/null; then echo "pid $PID ALIVE"; else echo "pid $PID NOT ALIVE"; fi
fi
tail -2 "$M1/training_losses.jsonl" 2>/dev/null | sed 's/^/loss: /'
[ -f "$M1/run_summary.json" ] && echo "run_summary: PRESENT (terminal)" || echo "run_summary: absent"

echo "--- M1 extension arms (amendment v2, seeds $EXT_SEEDS) ---"
for s in $EXT_SEEDS; do
  d="$M1BASE/seed_$s"
  if [ -d "$d" ]; then
    pid=""
    [ -f "$d/training_pid.txt" ] && pid=$(cat "$d/training_pid.txt")
    [ -z "$pid" ] && pid="-"
    alive="?"
    [ "$pid" != "-" ] && { kill -0 "$pid" 2>/dev/null && alive=ALIVE || alive=NOT_ALIVE; }
    term="absent"; [ -f "$d/run_summary.json" ] && term="PRESENT"
    echo "seed_$s: status=$(hb "$d/heartbeat.json" status) epoch=$(hb "$d/heartbeat.json" epoch) step=$(hb "$d/heartbeat.json" step)/$(hb "$d/heartbeat.json" total_steps) pid=$pid($alive) run_summary=$term"
  else
    echo "seed_$s: not launched"
  fi
done

echo "--- V5 polyA 3-seed ---"
echo "watcher=$(hb "$POLYA/watcher_heartbeat.json" status) phase=$(hb "$POLYA/watcher_heartbeat.json" phase) note=$(hb "$POLYA/watcher_heartbeat.json" note)"
for s in 20260921 20260922; do
  d="$POLYA/seed_$s"
  if [ -d "$d" ]; then
    st=$(hb "$d/heartbeat.json" status)
    pid=""
    [ -f "$d/training_pid.txt" ] && pid=$(cat "$d/training_pid.txt")
    [ -z "$pid" ] && [ -f "$POLYA/polya_3seed_seed${s}_pid.txt" ] && pid=$(cat "$POLYA/polya_3seed_seed${s}_pid.txt")
    [ -z "$pid" ] && pid="-"
    alive="?"
    [ "$pid" != "-" ] && { kill -0 "$pid" 2>/dev/null && alive=ALIVE || alive=NOT_ALIVE; }
    final="absent"
    [ -f "$d/final_validation_predictions.jsonl" ] && final="PRESENT"
    echo "seed_$s: status=$st pid=$pid($alive) final_predictions=$final"
  else
    echo "seed_$s: not launched"
  fi
done

echo "--- GPU snapshot (0-5 full cards; 6/7 MIG) ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null

pick_gpu() {
  BEST=-1; BEST_FREE=-1
  for i in 0 1 2 3 4 5; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null | tr -d ' ')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$i" 2>/dev/null | tr -d ' ')
    free=$(( ${total:-0} - ${used:-99999} ))
    if [ "$free" -gt "$BEST_FREE" ]; then BEST=$i; BEST_FREE=$free; fi
  done
  if [ "$BEST_FREE" -ge 6000 ]; then echo "$BEST"; else echo "-1"; fi
}

mig_uuid() { # first 3g.20gb MIG UUID on GPU 7 (fallback inference device)
  nvidia-smi -L | awk '/MIG 3g.20gb/{gsub("UUID: ","",$0); gsub(")","",$0); print $NF; exit}'
}

if [ "$DO_HARVEST" = "1" ]; then
  echo "--- auto-harvest ---"

  # 0) extension-arm launch pass: idempotent; uses any full card with enough free memory
  #    (amendment v2 pre-committed training). Never blocks on the lock for long.
  if [ -x "$LAUNCH_EXT" ]; then
    EXT_OUT=$(MRNA_M1_EXT_LOCK_WAIT=30 bash "$LAUNCH_EXT" 2>&1)
    echo "$EXT_OUT" | sed 's/^/[launch] /'
  else
    echo "[launch] extension launcher not found at $LAUNCH_EXT"
  fi

  OUT_M1="${MRNA_MON_OUT_M1:-$M1BASE/adjudication_v1}"
  mkdir -p "$OUT_M1" 2>/dev/null
  if [ -f "$M1/run_summary.json" ] && [ ! -f "$OUT_M1/m1_adjudication_v1.json" ]; then
    GPU=$(pick_gpu)
    if [ "$GPU" != "-1" ]; then
      echo "[harvest] M1 full adjudication on GPU$GPU ..."
      $PY "$HARVEST_M1" --gpu-index "$GPU" >> "$OUT_M1/harvest_run.log" 2>&1 \
        && echo "[harvest] M1 OK -> $OUT_M1/m1_adjudication_v1.json" \
        || echo "[harvest] M1 FAILED (see $OUT_M1/harvest_run.log)"
    else
      MU=$(mig_uuid)
      if [ -n "$MU" ]; then
        echo "[harvest] M1 full adjudication on MIG 3g.20gb ($MU) ..."
        CUDA_VISIBLE_DEVICES="$MU" $PY "$HARVEST_M1" --gpu-index 0 >> "$OUT_M1/harvest_run.log" 2>&1 \
          && echo "[harvest] M1 OK (MIG) -> $OUT_M1/m1_adjudication_v1.json" \
          || echo "[harvest] M1 FAILED on MIG (see $OUT_M1/harvest_run.log)"
      else
        echo "[harvest] M1 deferred: no GPU with >=6GB free"
        $PY "$HARVEST_M1" --gpu-index 0 --skip-gpu >> "$OUT_M1/harvest_g1.log" 2>&1 \
          && echo "[harvest] M1 G1/G3-only written (GPU deferred)"
      fi
    fi
  else
    [ -f "$OUT_M1/m1_adjudication_v1.json" ] && echo "[harvest] M1 already harvested"
  fi

  # 2) M1 3-seed ensemble harvest (amendment v2): requires G1 direction POSITIVE *and* all three
  #    arms terminal. If G1 is not positive the extension arms stay trained-not-analyzed.
  OUT_M1X="$OUT_M1/m1_3seed_ensemble_v1.json"
  if [ ! -f "$OUT_M1X" ]; then
    if [ ! -f "$OUT_M1/m1_adjudication_v1.json" ]; then
      echo "[harvest] M1 3-seed: deferred (single-arm adjudication absent)"
    else
      G1DIR=$($PY - "$OUT_M1/m1_adjudication_v1.json" <<'EOF'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("G1", {}).get("direction_positive"))
except Exception:
    print("ABSENT")
EOF
)
      ALLTERM=1
      for s in $EXT_SEEDS; do
        [ -f "$M1BASE/seed_$s/run_summary.json" ] || ALLTERM=0
      done
      if [ "$G1DIR" != "True" ]; then
        echo "[harvest] M1 3-seed: SKIPPED (G1 direction_positive=$G1DIR; extension arms remain trained-not-analyzed per amendment v2)"
      elif [ "$ALLTERM" != "1" ]; then
        echo "[harvest] M1 3-seed: deferred (G1 positive but extension arms not all terminal)"
      else
        echo "[harvest] M1 3-seed ensemble harvest (CPU) ..."
        $PY "$HARVEST_M1X" >> "$OUT_M1/harvest_3seed_run.log" 2>&1 \
          && echo "[harvest] M1 3-seed OK -> $OUT_M1X" \
          || echo "[harvest] M1 3-seed FAILED (see $OUT_M1/harvest_3seed_run.log)"
      fi
    fi
  else
    echo "[harvest] M1 3-seed already harvested"
  fi

  OUT_P3="$POLYA/harvest_v1"
  if [ -f "$POLYA/seed_20260921/final_validation_predictions.jsonl" ] && \
     [ -f "$POLYA/seed_20260922/final_validation_predictions.jsonl" ] && \
     [ ! -f "$OUT_P3/polya_3seed_harvest_v1.json" ]; then
    echo "[harvest] polyA 3-seed aggregation (CPU) ..."
    mkdir -p "$OUT_P3"
    $PY "$HARVEST_P3" >> "$OUT_P3/harvest_run.log" 2>&1 \
      && echo "[harvest] polyA 3-seed OK -> $OUT_P3/polya_3seed_harvest_v1.json" \
      || echo "[harvest] polyA 3-seed deferred/failed (see $OUT_P3/harvest_run.log)"
  else
    if [ -f "$OUT_P3/polya_3seed_harvest_v1.json" ]; then
      echo "[harvest] polyA 3-seed already harvested"
    else
      echo "[harvest] polyA 3-seed: deferred (final per-seed predictions absent)"
    fi
  fi
fi
echo "===== end ====="