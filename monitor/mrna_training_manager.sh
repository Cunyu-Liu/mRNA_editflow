#!/usr/bin/env bash
# mRNA EditFlow training manager (replaces apa_relaunch_watcher.sh).
# Launches APA polyA pre-finetune + V8 Stage 1 arms (S/H) on whichever full A100
# (GPU0-5) has the most free VRAM, batch sized to fit with margin. MIG slices
# (GPU 6/7) are excluded by project discipline. Auto-heals: on death without
# terminal marker, requeues; batch halves per repeated launch (floor 32).
#
# Prereg amendments (logged, pre-launch): batch adaptively reduced from the
# preregistered 128 on contended shared cards (user instruction: use any GPU
# with VRAM; no ≥30GiB gate). All other prereg fields unchanged.
set -u
MDIR=/home/cunyuliu/mrna_editflow_goal/monitor
LOG=$MDIR/training_manager.log
RT=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments
APA_OUT=$RT/xeditcritic_route_a/apa_3p5m_prefinetune_20260903
V8_OUT=$RT/xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904
W0=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
V8WT=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
APA_LOG=$RT/xeditcritic_route_a/apa_3p5m_prefinetune_20260903_relaunch.log
V8S_LOG=$RT/xeditcritic_route_a/v8_stage1_s_arm.log
V8H_LOG=$RT/xeditcritic_route_a/v8_stage1_h_arm.log

log() { echo "$(date '+%F %T') $*" >>"$LOG"; }

# launch counters per job (halve batch on repeated failed launches)
n_launch() { local f=$MDIR/state_$1.n; [ -f "$f" ] && cat "$f" || echo 0; }
bump() { local f=$MDIR/state_$1.n; echo "$(($(n_launch "$1") + 1))" > "$f"; }

# "idx free_mib" lines for GPU0-5
free_mib() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$1<=5 {print $1, $2}'
}

batch_for() {
  local f=$1
  if [ "$f" -ge 28672 ]; then echo 128
  elif [ "$f" -ge 20480 ]; then echo 96
  elif [ "$f" -ge 15360 ]; then echo 64
  elif [ "$f" -ge 11264 ]; then echo 48
  elif [ "$f" -ge 7168 ]; then echo 32
  else echo 0; fi
}

# pick (gpu batch) for a job with `attempts` prior launches
pick_slot() {
  local attempts=$1 best_gpu=-1 best_free=0
  while read -r idx free; do
    if [ -n "${free:-}" ] && [ "$free" -gt "$best_free" ]; then
      best_free=$free; best_gpu=$idx
    fi
  done < <(free_mib)
  if [ "$best_gpu" -lt 0 ]; then echo "-1 0"; return; fi
  local b
  b=$(batch_for "$best_free")
  while [ "$attempts" -gt 0 ] && [ "$b" -gt 32 ]; do b=$((b/2)); attempts=$((attempts-1)); done
  echo "$best_gpu $b"
}

job_apa() {
  local gpu=$1 batch=$2
  reserve "$gpu"
  (cd "$W0" && nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/route_a_v3/run_route2_mrnabert_apa_3p5m_prefinetune_v1.py \
    --physical-gpu-index "$gpu" --epochs 6 --seed 20260903 --batch "$batch" \
    >"$APA_LOG" 2>&1 &)
  log "LAUNCH apa gpu=$gpu batch=$batch pid=$!"
  bump apa
}

job_v8s() {
  local gpu=$1 batch=$2
  reserve "$gpu"
  (cd "$V8WT" && nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/route_a_v3/run_route2_v8_stage1_joint_prefinetune_v1.py \
    --arch s --libraries mrl,polya --physical-gpu-index "$gpu" --seed 20260903 \
    --epochs 2 --batch "$batch" >"$V8S_LOG" 2>&1 &)
  log "LAUNCH v8_s gpu=$gpu batch=$batch pid=$!"
  bump v8s
}

job_v8h() {
  local gpu=$1 batch=$2
  reserve "$gpu"
  (cd "$V8WT" && nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/route_a_v3/run_route2_v8_stage1_joint_prefinetune_v1.py \
    --arch h --libraries mrl,polya --physical-gpu-index "$gpu" --seed 20260903 \
    --epochs 2 --batch "$batch" >"$V8H_LOG" 2>&1 &)
  log "LAUNCH v8_h gpu=$gpu batch=$batch pid=$!"
  bump v8h
}

# terminal markers
apa_terminal() { [ -f "$APA_OUT/frozen_delta_results.json" ]; }
v8s_terminal() { [ -f "$V8_OUT/s_mrl-polya/run_report.json" ]; }
v8h_terminal() { [ -f "$V8_OUT/h_mrl-polya/run_report.json" ]; }

apa_running()  { pgrep -f "run_route2_mrnabert_apa_3p5m_prefinetune_v1.py" >/dev/null; }
v8s_running()  { pgrep -f "run_route2_v8_stage1_joint_prefinetune_v1.py --arch s" >/dev/null; }
v8h_running()  { pgrep -f "run_route2_v8_stage1_joint_prefinetune_v1.py --arch h" >/dev/null; }

log "training manager started (adaptive batch, no 30GiB gate)"

# APA harvest-once (after terminal)
APA_HARVESTED=$MDIR/apa_harvested.marker

while true; do
  if apa_terminal && [ ! -f "$APA_HARVESTED" ]; then
    log "APA terminal detected - running harvest chain"
    if bash "$MDIR/harvest_polya.sh" >>"$LOG" 2>&1; then
      if (cd "$W0" && git add docs/training_journal/TRAINING_LOG_202609.md \
          && git commit -m "analysis(route2): polyA Route A terminal adjudication (training manager auto-harvest)" \
          && git push origin HEAD) >>"$LOG" 2>&1; then
        log "polyA journal committed+pushed"
      else
        log "polyA journal commit FAILED - NEEDS HUMAN"
      fi
      touch "$APA_HARVESTED"
      log "APA harvest done"
    else
      log "APA harvest FAILED - NEEDS HUMAN"
    fi
  fi

  if ! apa_terminal && ! apa_running; then
    read -r gpu batch < <(pick_slot "$(n_launch apa)")
    if [ "$gpu" -ge 0 ] && [ "$batch" -ge 32 ]; then job_apa "$gpu" "$batch"; fi
  fi
  if ! v8s_terminal && ! v8s_running; then
    read -r gpu batch < <(pick_slot "$(n_launch v8s)")
    if [ "$gpu" -ge 0 ] && [ "$batch" -ge 32 ]; then job_v8s "$gpu" "$batch"; fi
  fi
  if ! v8h_terminal && ! v8h_running; then
    read -r gpu batch < <(pick_slot "$(n_launch v8h)")
    if [ "$gpu" -ge 0 ] && [ "$batch" -ge 32 ]; then job_v8h "$gpu" "$batch"; fi
  fi
  sleep 120
done
