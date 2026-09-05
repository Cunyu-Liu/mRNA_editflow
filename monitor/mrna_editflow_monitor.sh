#!/usr/bin/env bash
# mRNA EditFlow - every-2h monitoring (2026-09-05 v3)
# Tracks: training manager + APA polyA pre-finetune + V8 Stage 1 arms (S/H).
set -u
MDIR=/home/cunyuliu/mrna_editflow_goal/monitor
MONLOG=$MDIR/status.log
FLAG=$MDIR/needs_attention.txt
mkdir -p "$MDIR"
stamp=$(date '+%Y-%m-%d %H:%M:%S')
NV=$(command -v nvidia-smi || echo /usr/bin/nvidia-smi)
RT=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments
APA_OUT=$RT/xeditcritic_route_a/apa_3p5m_prefinetune_20260903
V8_OUT=$RT/xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904

mgr_pid=$(pgrep -f mrna_training_manager.sh | head -1)
apa_pid=$(pgrep -f run_route2_mrnabert_apa_3p5m_prefinetune | head -1)
v8s_pid=$(pgrep -f "v8_stage1_joint_prefinetune_v1.py --arch s" | head -1)
v8h_pid=$(pgrep -f "v8_stage1_joint_prefinetune_v1.py --arch h" | head -1)
gpu=$("$NV" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')
apa_last=$(tail -1 "$RT/xeditcritic_route_a/apa_3p5m_prefinetune_20260903_relaunch.log" 2>/dev/null | cut -c1-160)
apa_met=$(tail -1 "$APA_OUT/epoch_frozen_delta_metrics.jsonl" 2>/dev/null)
v8s_last=$(tail -1 "$RT/xeditcritic_route_a/v8_stage1_s_arm.log" 2>/dev/null | cut -c1-160)
v8h_last=$(tail -1 "$RT/xeditcritic_route_a/v8_stage1_h_arm.log" 2>/dev/null | cut -c1-160)

echo "$stamp MGR alive=${mgr_pid:+1} | $apa_last" >>"$MONLOG"
echo "$stamp APA alive=${apa_pid:+1} | $apa_met" >>"$MONLOG"
echo "$stamp V8S alive=${v8s_pid:+1} | $v8s_last" >>"$MONLOG"
echo "$stamp V8H alive=${v8h_pid:+1} | $v8h_last" >>"$MONLOG"
echo "$stamp GPU {$gpu}" >>"$MONLOG"

att=0
if [ -z "$apa_pid" ] && [ ! -f "$APA_OUT/frozen_delta_results.json" ]; then
  echo "$stamp APA NOT RUNNING AND NOT TERMINAL - INVESTIGATE" >>"$FLAG"; att=1
fi
if [ -z "$v8s_pid" ] && [ ! -f "$V8_OUT/s_mrl-polya/run_report.json" ]; then
  echo "$stamp V8S NOT RUNNING AND NOT TERMINAL - INVESTIGATE" >>"$FLAG"; att=1
fi
if [ -z "$v8h_pid" ] && [ ! -f "$V8_OUT/h_mrl-polya/run_report.json" ]; then
  echo "$stamp V8H NOT RUNNING AND NOT TERMINAL - INVESTIGATE" >>"$FLAG"; att=1
fi
if [ -z "$mgr_pid" ]; then
  echo "$stamp TRAINING MANAGER NOT RUNNING - INVESTIGATE" >>"$FLAG"; att=1
fi
if [ "$att" = "0" ]; then rm -f "$FLAG"; fi
