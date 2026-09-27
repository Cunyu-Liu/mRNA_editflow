#!/bin/bash
# Launch the M1 intervention 3-seed extension arms (amendment v2, ACTIVE/FROZEN 2026-09-27T19:06Z).
# Idempotent + flock-serialised + auto card-picking: skips arms that are running/terminal, picks the
# full card (0-5) with the most free memory, and refuses to land on a card that hosts another
# extension arm (spread rule; the 2026-09-27 same-card OOM lesson). Safe to call repeatedly from the
# 2h cron monitor so that free cluster memory is used as soon as it appears (no memory-side gate on
# our side beyond the safety margin below).
#
# Usage: bash launch_m1_intervention_extension_v2.sh [--dry-run]
# Env:   MRNA_M1_EXT_MINFREE  minimum free MiB required (default 14000 = ~9.5GB job + headroom)
# Exit:  0 = all arms running/terminal or launched; 1 = at least one arm could not be launched now.
set -u

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
WT=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
MNT=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2
BASE="$MNT/experiments/xeditcritic_m1_intervention"
RUNNER="scripts/route_a_v3/run_route2_m1_intervention_fullft_v1.py"
# Lock file path carries a version suffix: a training child launched by launcher v1 inherited the old
# lock FD and holds it for the whole run, so re-using that path would block every later invocation.
LOCK=/tmp/mrna_m1_ext_launch_v2.lock
LOCK_WAIT="${MRNA_M1_EXT_LOCK_WAIT:-60}"
MINFREE="${MRNA_M1_EXT_MINFREE:-14000}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
HOSTED_CARDS=""   # full-card indices already hosting an extension arm in this run

cd "$WT" || exit 1
exec 9>"$LOCK" || exit 1
# bounded wait: the monitor must never hang on the lock (children get 9>&- below, so no stale holder)
if ! flock -w "$LOCK_WAIT" 9; then echo "launch lock busy (>${LOCK_WAIT}s) - deferring to next monitor pass"; exit 1; fi

card_free() { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

pick_card() { # pick_card <excluded_csv> -> "<index> <free_mib>" or "-1 0"
  local excl=",$1," best=-1 bestfree=-1 free
  for i in 0 1 2 3 4 5; do
    case "$excl" in *",$i,"*) continue;; esac
    free=$(card_free "$i")
    if [ -n "$free" ] && [ "$free" -gt "$bestfree" ]; then best=$i; bestfree=$free; fi
  done
  if [ "$bestfree" -ge "$MINFREE" ]; then echo "$best $bestfree"; else echo "-1 0"; fi
}

launch() { # launch <seed>
  local seed="$1" out pid card free
  out="$BASE/seed_$seed"
  mkdir -p "$out"
  if [ -f "$out/run_summary.json" ]; then echo "seed $seed: TERMINAL (run_summary present) - skip"; return 0; fi
  if [ -f "$out/training_pid.txt" ]; then
    pid=$(cat "$out/training_pid.txt")
    if kill -0 "$pid" 2>/dev/null; then echo "seed $seed: already RUNNING pid $pid - skip"; return 0; fi
  fi
  read -r card free <<<"$(pick_card "$HOSTED_CARDS")"
  if [ "$card" = "-1" ]; then
    echo "seed $seed: no full card with >=${MINFREE}MiB free (best available below margin) - NOT launched, retry on next monitor pass"
    return 1
  fi
  if [ "$DRY" = "1" ]; then echo "seed $seed: DRY-RUN would launch on GPU$card (free ${free}MiB)"; HOSTED_CARDS="$HOSTED_CARDS,$card"; return 0; fi
  # setsid: fully detach from this (possibly short-lived, ssh-owned) session so a client-side
  # disconnect cannot take the training process down. Per-attempt console log + attempts.jsonl:
  # the previous design reused one log file and overwrote the death evidence of earlier attempts.
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  setsid env M1_INTERVENTION_OUT_DIR="$out" nohup "$PY" -u "$RUNNER" --physical-gpu-index "$card" --seed "$seed" \
    > "$out/train_console_${stamp}.log" 2>&1 9>&- < /dev/null &
  pid=$!
  echo "$pid" > "$out/training_pid.txt"
  $PY - "$out/attempts.jsonl" "$stamp" "$seed" "$card" "$pid" "$free" <<'EOF'
import json, sys
path, stamp, seed, card, pid, free = sys.argv[1:7]
with open(path, "a") as handle:
    handle.write(json.dumps({"utc": stamp, "seed": int(seed), "physical_gpu_index": int(card),
                             "pid": int(pid), "free_mib_at_launch": int(free),
                             "log": f"train_console_{stamp}.log"}, sort_keys=True) + "\n")
EOF
  HOSTED_CARDS="$HOSTED_CARDS,$card"
  # post-launch assertion: $! must be the runner itself (setsid execs in place when the shell has no
  # job control); a mismatch would silently break the pid-guard for later monitor passes.
  sleep 2
  if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q run_route2_m1_intervention_fullft_v1; then
    echo "seed $seed: LAUNCHED on GPU$card pid $pid (free was ${free}MiB, log $out/train_console_${stamp}.log)"
  else
    echo "seed $seed: WARNING pid $pid is not the runner (setsid forked?); inspect $out/train_console_${stamp}.log"
  fi
  return 0
}

RC=0
launch 20260904 || RC=1
launch 20260905 || RC=1
exit "$RC"