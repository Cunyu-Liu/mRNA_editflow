#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
MINIMUM_FREE_MB="${MINIMUM_FREE_MB:-4096}"
CONTROL_ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_critic_v2_task_study_macro_controls_adjudication_v1.json"
THREE_SEED_ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_critic_v2_task_study_macro_three_seed_adjudication_v1.json"

while [[ ! -f "${THREE_SEED_ADJUDICATION}" ]]; do
  if [[ -f "${CONTROL_ADJUDICATION}" ]]; then
    supports_seeds=$("${PYTHON}" -c \
      'import json,sys; p=json.load(open(sys.argv[1])); print(str(p.get("status") == "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS" and p.get("supports_three_frozen_seeds") is True).lower())' \
      "${CONTROL_ADJUDICATION}")
    if [[ "${supports_seeds}" != "true" ]]; then
      printf '%s critic_v2_control_gate_terminal_no_go_post_confirmation_not_started\n' "$(date -Is)"
      exit 0
    fi
  fi
  printf '%s waiting_for_critic_v2_three_seed_adjudication\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

supports_test=$("${PYTHON}" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); print(str(p.get("status") == "CRITIC_V2_THREE_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST" and p.get("supports_single_frozen_development_test") is True).lower())' \
  "${THREE_SEED_ADJUDICATION}")
if [[ "${supports_test}" != "true" ]]; then
  printf '%s critic_v2_three_seed_gate_terminal_no_go_post_confirmation_not_started\n' "$(date -Is)"
  exit 0
fi

cd "${REPO_ROOT}"
printf '%s starting_critic_v2_post_confirmation_development_pipeline\n' "$(date -Is)"
"${PYTHON}" -u scripts/route_a_v3/run_route2_mrnabert_critic_v2_post_confirmation_stage_v1.py \
  --minimum-free-mb "${MINIMUM_FREE_MB}" \
  --poll-seconds "${POLL_SECONDS}"
printf '%s finished_critic_v2_post_confirmation_development_pipeline\n' "$(date -Is)"
