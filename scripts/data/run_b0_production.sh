#!/usr/bin/env bash
#
# Execute the full D1-bound B0 benchmark as audited nodes 00 through 13.
# The attempt root is an approved fresh external directory.  Code and D1
# inputs are checked before and after every applicable node.  Any failure
# preserves the attempt and terminates without claiming stage completion.

set -Eeuo pipefail
umask 027
unset PYTHONHOME PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

EX_USAGE=64
EX_DATAERR=65
EX_CANTCREAT=73
EX_GUARD=74
DEFAULT_MINIMUM_FREE_BYTES=1073741824

usage() {
  printf '%s\n' \
    "usage: $0 \\" \
    "  --isolated-worktree ABSOLUTE_PATH \\" \
    "  --d1-acceptance ABSOLUTE_PATH \\" \
    "  --b0-attempt-root ABSOLUTE_FRESH_PATH \\" \
    "  --python-bin ABSOLUTE_VENV_LAUNCHER \\" \
    "  --runtime-manifest ABSOLUTE_JSON \\" \
    "  --expected-commit 40_HEX \\" \
    "  --expected-driver-sha256 64_HEX \\" \
    "  --expected-dirty-state-sha256 64_HEX \\" \
    "  --expected-runtime-prefix ABSOLUTE_PATH \\" \
    "  --expected-runtime-manifest-sha256 64_HEX \\" \
    "  --approved-b0-parent ABSOLUTE_EXTERNAL_PATH \\" \
    "  [--minimum-free-bytes POSITIVE_INTEGER]"
}

usage_error() {
  printf 'B0 driver invocation error: %s\n' "$1" >&2
  usage >&2
  exit "$EX_USAGE"
}

require_flag_value() {
  local flag="$1"
  local remaining="$2"
  if [[ "$remaining" -lt 2 ]]; then
    usage_error "missing value for $flag"
  fi
}

set_once() {
  local variable="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "${!variable:-}" ]]; then
    usage_error "duplicate option: $flag"
  fi
  if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    usage_error "empty or control-character value for $flag"
  fi
  printf -v "$variable" '%s' "$value"
}

PROJECT_ROOT_RAW=""
D1_ACCEPTANCE_RAW=""
B0_ATTEMPT_ROOT_RAW=""
PYTHON_BIN_RAW=""
RUNTIME_MANIFEST_RAW=""
EXPECTED_COMMIT=""
EXPECTED_DRIVER_SHA256=""
EXPECTED_DIRTY_STATE_SHA256=""
EXPECTED_RUNTIME_PREFIX_RAW=""
EXPECTED_RUNTIME_MANIFEST_SHA256=""
APPROVED_B0_PARENT_RAW=""
MINIMUM_FREE_BYTES="$DEFAULT_MINIMUM_FREE_BYTES"
MINIMUM_FREE_BYTES_SEEN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --isolated-worktree)
      require_flag_value "$1" "$#"
      set_once PROJECT_ROOT_RAW "$1" "$2"
      shift 2
      ;;
    --d1-acceptance)
      require_flag_value "$1" "$#"
      set_once D1_ACCEPTANCE_RAW "$1" "$2"
      shift 2
      ;;
    --b0-attempt-root)
      require_flag_value "$1" "$#"
      set_once B0_ATTEMPT_ROOT_RAW "$1" "$2"
      shift 2
      ;;
    --python-bin)
      require_flag_value "$1" "$#"
      set_once PYTHON_BIN_RAW "$1" "$2"
      shift 2
      ;;
    --runtime-manifest)
      require_flag_value "$1" "$#"
      set_once RUNTIME_MANIFEST_RAW "$1" "$2"
      shift 2
      ;;
    --expected-commit)
      require_flag_value "$1" "$#"
      set_once EXPECTED_COMMIT "$1" "$2"
      shift 2
      ;;
    --expected-driver-sha256)
      require_flag_value "$1" "$#"
      set_once EXPECTED_DRIVER_SHA256 "$1" "$2"
      shift 2
      ;;
    --expected-dirty-state-sha256)
      require_flag_value "$1" "$#"
      set_once EXPECTED_DIRTY_STATE_SHA256 "$1" "$2"
      shift 2
      ;;
    --expected-runtime-prefix)
      require_flag_value "$1" "$#"
      set_once EXPECTED_RUNTIME_PREFIX_RAW "$1" "$2"
      shift 2
      ;;
    --expected-runtime-manifest-sha256)
      require_flag_value "$1" "$#"
      set_once EXPECTED_RUNTIME_MANIFEST_SHA256 "$1" "$2"
      shift 2
      ;;
    --approved-b0-parent)
      require_flag_value "$1" "$#"
      set_once APPROVED_B0_PARENT_RAW "$1" "$2"
      shift 2
      ;;
    --minimum-free-bytes)
      require_flag_value "$1" "$#"
      if [[ "$MINIMUM_FREE_BYTES_SEEN" -eq 1 ]]; then
        usage_error "duplicate option: $1"
      fi
      MINIMUM_FREE_BYTES="$2"
      MINIMUM_FREE_BYTES_SEEN=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage_error "unknown argument: $1"
      ;;
  esac
done

for required_name in \
  PROJECT_ROOT_RAW \
  D1_ACCEPTANCE_RAW \
  B0_ATTEMPT_ROOT_RAW \
  PYTHON_BIN_RAW \
  RUNTIME_MANIFEST_RAW \
  EXPECTED_COMMIT \
  EXPECTED_DRIVER_SHA256 \
  EXPECTED_DIRTY_STATE_SHA256 \
  EXPECTED_RUNTIME_PREFIX_RAW \
  EXPECTED_RUNTIME_MANIFEST_SHA256 \
  APPROVED_B0_PARENT_RAW
do
  if [[ -z "${!required_name}" ]]; then
    usage_error "missing required option for $required_name"
  fi
done

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  usage_error "--expected-commit must be 40 lowercase hexadecimal characters"
fi
if [[ ! "$EXPECTED_DRIVER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  usage_error \
    "--expected-driver-sha256 must be 64 lowercase hexadecimal characters"
fi
if [[ ! "$EXPECTED_DIRTY_STATE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  usage_error \
    "--expected-dirty-state-sha256 must be 64 lowercase hexadecimal characters"
fi
if [[ ! "$EXPECTED_RUNTIME_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  usage_error \
    "--expected-runtime-manifest-sha256 must be 64 lowercase hexadecimal characters"
fi
if [[ ! "$MINIMUM_FREE_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  usage_error "--minimum-free-bytes must be a positive integer"
fi
for absolute_value in \
  "$PROJECT_ROOT_RAW" \
  "$D1_ACCEPTANCE_RAW" \
  "$B0_ATTEMPT_ROOT_RAW" \
  "$PYTHON_BIN_RAW" \
  "$RUNTIME_MANIFEST_RAW" \
  "$EXPECTED_RUNTIME_PREFIX_RAW" \
  "$APPROVED_B0_PARENT_RAW"
do
  if [[ "$absolute_value" != /* ]]; then
    usage_error "all supplied paths must be absolute: $absolute_value"
  fi
done

for bootstrap_tool in git realpath dirname basename mkdir; do
  if ! command -v "$bootstrap_tool" >/dev/null 2>&1; then
    printf 'required bootstrap tool is unavailable: %s\n' "$bootstrap_tool" >&2
    exit "$EX_DATAERR"
  fi
done

PROJECT_ROOT="$(realpath "$PROJECT_ROOT_RAW")"
D1_ACCEPTANCE="$(realpath "$D1_ACCEPTANCE_RAW")"
APPROVED_B0_PARENT="$(realpath "$APPROVED_B0_PARENT_RAW")"
EXPECTED_RUNTIME_PREFIX="$(realpath "$EXPECTED_RUNTIME_PREFIX_RAW")"
PYTHON_BIN_DIR="$(realpath "$(dirname "$PYTHON_BIN_RAW")")"
PYTHON_BIN="$PYTHON_BIN_DIR/$(basename "$PYTHON_BIN_RAW")"
RUNTIME_MANIFEST="$(realpath "$RUNTIME_MANIFEST_RAW")"
B0_REQUESTED_PARENT="$(realpath "$(dirname "$B0_ATTEMPT_ROOT_RAW")")"
B0_ATTEMPT_BASENAME="$(basename "$B0_ATTEMPT_ROOT_RAW")"
B0_ATTEMPT_ROOT="$APPROVED_B0_PARENT/$B0_ATTEMPT_BASENAME"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  printf 'isolated worktree is missing: %s\n' "$PROJECT_ROOT" >&2
  exit "$EX_DATAERR"
fi
if [[ ! -f "$D1_ACCEPTANCE" ]]; then
  printf 'D1 acceptance is missing: %s\n' "$D1_ACCEPTANCE" >&2
  exit "$EX_DATAERR"
fi
if [[ ! -d "$APPROVED_B0_PARENT" ]]; then
  printf 'approved B0 parent is missing: %s\n' "$APPROVED_B0_PARENT" >&2
  exit "$EX_DATAERR"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python launcher is unavailable: %s\n' "$PYTHON_BIN" >&2
  exit "$EX_DATAERR"
fi
if [[ ! -f "$RUNTIME_MANIFEST" || -L "$RUNTIME_MANIFEST" ]]; then
  printf 'runtime manifest is unavailable or unsafe: %s\n' \
    "$RUNTIME_MANIFEST" >&2
  exit "$EX_DATAERR"
fi
if [[ "$B0_REQUESTED_PARENT" != "$APPROVED_B0_PARENT" ]]; then
  printf 'attempt root is not a direct child of approved B0 parent\n' >&2
  exit "$EX_DATAERR"
fi
if [[ "$B0_ATTEMPT_BASENAME" == "." || "$B0_ATTEMPT_BASENAME" == ".." ]]; then
  printf 'attempt root basename is unsafe: %s\n' "$B0_ATTEMPT_BASENAME" >&2
  exit "$EX_DATAERR"
fi
if [[ -e "$B0_ATTEMPT_ROOT" ]]; then
  printf 'refusing existing B0 attempt root: %s\n' "$B0_ATTEMPT_ROOT" >&2
  exit "$EX_CANTCREAT"
fi

GIT_TOP_LEVEL="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel)"
GIT_TOP_LEVEL="$(realpath "$GIT_TOP_LEVEL")"
if [[ "$GIT_TOP_LEVEL" != "$PROJECT_ROOT" ]]; then
  printf 'approved isolated worktree is not exact Git top-level: %s\n' \
    "$PROJECT_ROOT" >&2
  exit "$EX_DATAERR"
fi

paths_overlap() {
  local left="${1%/}"
  local right="${2%/}"
  [[ "$left" == "$right" || "$left" == "$right/"* || "$right" == "$left/"* ]]
}

if paths_overlap "$APPROVED_B0_PARENT" "$PROJECT_ROOT"; then
  printf 'approved B0 parent must be external to and disjoint from repository\n' >&2
  exit "$EX_DATAERR"
fi

D1_ROOT_BOOTSTRAP="$(
  "$PYTHON_BIN" -c '
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get("stage_d1_root")
if not isinstance(value, str) or not pathlib.Path(value).is_absolute():
    raise SystemExit("D1 stage root is not an absolute JSON string")
print(pathlib.Path(value).resolve(strict=True))
' "$D1_ACCEPTANCE"
)"
if paths_overlap "$B0_ATTEMPT_ROOT" "$D1_ROOT_BOOTSTRAP"; then
  printf 'B0 attempt root must be disjoint from D1 stage root\n' >&2
  exit "$EX_DATAERR"
fi

DRIVER_PATH="$(realpath "${BASH_SOURCE[0]}")"
EXPECTED_DRIVER_PATH="$PROJECT_ROOT/scripts/data/run_b0_production.sh"
if [[ "$DRIVER_PATH" != "$EXPECTED_DRIVER_PATH" ]]; then
  printf 'driver is not executing from approved isolated worktree: %s\n' \
    "$DRIVER_PATH" >&2
  exit "$EX_DATAERR"
fi
GUARD="$PROJECT_ROOT/scripts/execution/b0_driver_guard.py"
AUDIT_WRAPPER="$PROJECT_ROOT/scripts/execution/run_audited_command.py"
ACCEPTANCE_SEMANTICS="$PROJECT_ROOT/scripts/execution/acceptance_semantics.py"
for required_entry in "$GUARD" "$AUDIT_WRAPPER" "$ACCEPTANCE_SEMANTICS"; do
  if [[ ! -f "$required_entry" || -L "$required_entry" ]]; then
    printf 'required execution entry is missing: %s\n' "$required_entry" >&2
    exit "$EX_DATAERR"
  fi
done

CURRENT_HEAD="$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD)"
if [[ "$CURRENT_HEAD" != "$EXPECTED_COMMIT" ]]; then
  printf 'HEAD differs from caller-approved commit before bootstrap\n' >&2
  exit "$EX_DATAERR"
fi
"$PYTHON_BIN" -c '
import hashlib
import pathlib
import subprocess
import sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
for relative in sys.argv[2:]:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe bootstrap entry: {relative}")
    tag = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-v", "--", relative],
        text=True,
    ).rstrip("\n")
    if tag != f"H {relative}":
        raise SystemExit(f"nonstandard bootstrap index flag: {tag!r}")
    head = subprocess.check_output(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
    )
    live = path.read_bytes()
    if hashlib.sha256(live).digest() != hashlib.sha256(head).digest():
        raise SystemExit(f"bootstrap entry differs from HEAD: {relative}")
' "$PROJECT_ROOT" \
  "scripts/data/run_b0_production.sh" \
  "scripts/execution/b0_driver_guard.py" \
  "scripts/execution/run_audited_command.py" \
  "scripts/execution/acceptance_semantics.py"

ATTEMPT_INITIALIZED=0
ATTEMPT_COMPLETE=0
FAILURE_FINALIZED=0
CURRENT_NODE=""
CURRENT_WRAPPER_PID=""
WATCHDOG_PID=""
LAST_FAILURE_REASON=""

stop_watchdog() {
  if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  WATCHDOG_PID=""
}

finalize_failure() {
  local exit_code="$1"
  local reason="$2"
  local signal_name="${3:-}"
  local line="${4:-}"
  local command_text="${5:-}"
  if [[ "$ATTEMPT_COMPLETE" -eq 1 ||
        "$FAILURE_FINALIZED" -eq 1 ]]; then
    return 0
  fi
  if [[ ! -f "$B0_ATTEMPT_ROOT/status.json" ||
        ! -f "$B0_ATTEMPT_ROOT/logs/events.jsonl" ||
        ! -f "$B0_ATTEMPT_ROOT/terminal.lock" ]]; then
    return 0
  fi
  FAILURE_FINALIZED=1
  LAST_FAILURE_REASON="$reason"
  stop_watchdog
  local failure_args=(
    "$GUARD" failure
    --attempt-root "$B0_ATTEMPT_ROOT"
    --exit-code "$exit_code"
    --reason "$reason"
  )
  if [[ -n "$CURRENT_NODE" ]]; then
    failure_args+=(--node "$CURRENT_NODE")
  fi
  if [[ -n "$signal_name" ]]; then
    failure_args+=(--signal "$signal_name")
  fi
  if [[ -n "$line" ]]; then
    failure_args+=(--line "$line")
  fi
  if [[ -n "$command_text" ]]; then
    failure_args+=(--command "$command_text")
  fi
  if [[ -n "$CURRENT_WRAPPER_PID" ]]; then
    failure_args+=(--wrapper-pid "$CURRENT_WRAPPER_PID")
  fi
  "$PYTHON_BIN" "${failure_args[@]}" || true
}

abort_attempt() {
  local reason="$1"
  local exit_code="${2:-$EX_GUARD}"
  finalize_failure "$exit_code" "$reason" "" "${BASH_LINENO[0]:-}" \
    "${BASH_COMMAND:-}"
  exit "$exit_code"
}

on_error() {
  local exit_code="$?"
  local line="${BASH_LINENO[0]:-${LINENO}}"
  local command_text="${BASH_COMMAND:-UNKNOWN_COMMAND}"
  finalize_failure "$exit_code" "UNHANDLED_DRIVER_ERROR" "" "$line" "$command_text"
  return "$exit_code"
}

on_exit() {
  local exit_code="$?"
  if [[ "$ATTEMPT_COMPLETE" -ne 1 && "$FAILURE_FINALIZED" -ne 1 ]]; then
    finalize_failure "$exit_code" "DRIVER_EXIT_BEFORE_COMPLETION"
  fi
}

on_signal() {
  local signal_name="$1"
  local signal_number="$2"
  local exit_code="$((128 + signal_number))"
  trap - ERR INT TERM HUP
  if "$PYTHON_BIN" "$GUARD" terminal-success \
    --attempt-root "$B0_ATTEMPT_ROOT" >/dev/null 2>&1
  then
    ATTEMPT_COMPLETE=1
    exit 0
  fi
  if [[ -n "$CURRENT_WRAPPER_PID" ]] &&
     kill -0 "$CURRENT_WRAPPER_PID" 2>/dev/null; then
    kill -s "$signal_name" "$CURRENT_WRAPPER_PID" 2>/dev/null || true
    wait "$CURRENT_WRAPPER_PID" 2>/dev/null || true
  fi
  finalize_failure "$exit_code" "SIGNAL_${signal_name}" "$signal_name"
  exit "$exit_code"
}

trap on_error ERR
trap on_exit EXIT
trap 'on_signal INT 2' INT
trap 'on_signal TERM 15' TERM
trap 'on_signal HUP 1' HUP

if ATTEMPT_MANIFEST_SHA256="$(
  "$PYTHON_BIN" "$GUARD" init \
  --attempt-root "$B0_ATTEMPT_ROOT" \
  --project-root "$PROJECT_ROOT" \
  --d1-acceptance "$D1_ACCEPTANCE" \
  --approved-b0-parent "$APPROVED_B0_PARENT" \
  --python-launcher "$PYTHON_BIN" \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --driver "$DRIVER_PATH" \
  --guard "$GUARD" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-driver-sha256 "$EXPECTED_DRIVER_SHA256" \
  --expected-dirty-state-sha256 "$EXPECTED_DIRTY_STATE_SHA256" \
  --expected-runtime-prefix "$EXPECTED_RUNTIME_PREFIX" \
  --expected-runtime-manifest-sha256 \
    "$EXPECTED_RUNTIME_MANIFEST_SHA256" \
  --minimum-free-bytes "$MINIMUM_FREE_BYTES"
)"; then
  :
else
  bootstrap_rc="$?"
  finalize_failure "$bootstrap_rc" "EARLY_DRIVER_BOOTSTRAP_FAILURE"
  exit "$bootstrap_rc"
fi
ATTEMPT_INITIALIZED=1

CODE_MANIFEST="$B0_ATTEMPT_ROOT/provenance/code_manifest.json"
if CODE_MANIFEST_SHA256="$(
  "$PYTHON_BIN" "$GUARD" fingerprint \
  --project-root "$PROJECT_ROOT" \
  --driver-path "$DRIVER_PATH" \
  --output "$CODE_MANIFEST" \
  --sha256-output "$B0_ATTEMPT_ROOT/provenance/code_manifest.sha256" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-driver-sha256 "$EXPECTED_DRIVER_SHA256" \
  --expected-dirty-state-sha256 "$EXPECTED_DIRTY_STATE_SHA256"
)"; then
  :
else
  bootstrap_rc="$?"
  finalize_failure "$bootstrap_rc" "EARLY_DRIVER_BOOTSTRAP_FAILURE"
  exit "$bootstrap_rc"
fi

BASELINE_HEAD="$EXPECTED_COMMIT"
BASELINE_DIRTY_STATE_SHA256="$EXPECTED_DIRTY_STATE_SHA256"

"$PYTHON_BIN" "$GUARD" watchdog-once \
  --attempt-root "$B0_ATTEMPT_ROOT" \
  --d1-acceptance "$D1_ACCEPTANCE"
"$PYTHON_BIN" "$GUARD" watchdog \
  --attempt-root "$B0_ATTEMPT_ROOT" \
  --d1-acceptance "$D1_ACCEPTANCE" \
  --parent-pid "$$" &
WATCHDOG_PID="$!"

guard_event() {
  local event="$1"
  shift
  "$PYTHON_BIN" "$GUARD" event \
    --attempt-root "$B0_ATTEMPT_ROOT" \
    --event "$event" \
    "$@"
}

guard_status() {
  local state="$1"
  shift
  "$PYTHON_BIN" "$GUARD" status \
    --attempt-root "$B0_ATTEMPT_ROOT" \
    --state "$state" \
    "$@"
}

assert_code_fingerprint() {
  local node="$1"
  local timing="$2"
  local output="$B0_ATTEMPT_ROOT/provenance/fingerprints/$node.$timing.json"
  if ! "$PYTHON_BIN" "$GUARD" assert-fingerprint \
    --baseline "$CODE_MANIFEST" \
    --expected-baseline-sha256 "$CODE_MANIFEST_SHA256" \
    --project-root "$PROJECT_ROOT" \
    --driver-path "$DRIVER_PATH" \
    --label "$node:$timing" \
    --observed-output "$output"
  then
    abort_attempt "CODE_FINGERPRINT_DRIFT_${node}_${timing}" "$EX_GUARD"
  fi
}

assert_d1_inputs() {
  local node="$1"
  local timing="$2"
  local output="$B0_ATTEMPT_ROOT/provenance/input_checks/$node.$timing.json"
  if ! "$PYTHON_BIN" "$GUARD" assert-inputs \
    --input-manifest "$INPUT_MANIFEST" \
    --expected-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
    --label "$node:$timing" \
    --output "$output"
  then
    abort_attempt "D1_INPUT_DRIFT_${node}_${timing}" "$EX_GUARD"
  fi
}

run_named_gate() {
  local gate_name="$1"
  local predicate="$2"
  local artifact="$3"
  shift 3
  local gate_rc
  guard_event "GATE_START" --detail "$gate_name"
  if "$PYTHON_BIN" "$GUARD" validate-gate \
    --gate "$predicate" \
    --label "$gate_name" \
    --artifact "$artifact" \
    --evidence-output "$B0_ATTEMPT_ROOT/provenance/gates/$gate_name.json" \
    "$@"
  then
    gate_rc=0
  else
    gate_rc="$?"
    finalize_failure "$gate_rc" "SEMANTIC_GATE_FAILED_${gate_name}"
    exit "$gate_rc"
  fi
  guard_event "GATE_PASSED" --detail "$gate_name"
}

run_node() {
  local node="$1"
  shift
  local run_root="$B0_ATTEMPT_ROOT/audit/$node"
  local wrapper_rc
  local post_rc=0

  CURRENT_NODE="$node"
  if [[ -e "$run_root" ]]; then
    abort_attempt "REFUSING_EXISTING_AUDIT_NODE_${node}" "$EX_CANTCREAT"
  fi
  assert_code_fingerprint "$node" before
  if [[ "$node" != "00_preflight" ]]; then
    assert_d1_inputs "$node" before
  fi
  guard_event "NODE_START" --node "$node"
  guard_status RUNNING --node "$node"

  "$PYTHON_BIN" "$AUDIT_WRAPPER" \
    --run-root "$run_root" \
    --project-root "$PROJECT_ROOT" \
    --working-directory "$PROJECT_ROOT" \
    --workload-class NON_NEURAL_DATA_BENCHMARK \
    --expected-git-head "$BASELINE_HEAD" \
    --expected-git-dirty-state-sha256 "$BASELINE_DIRTY_STATE_SHA256" \
    -- "$@" &
  CURRENT_WRAPPER_PID="$!"
  guard_status RUNNING --node "$node" --wrapper-pid "$CURRENT_WRAPPER_PID"

  if wait "$CURRENT_WRAPPER_PID"; then
    wrapper_rc=0
  else
    wrapper_rc="$?"
  fi
  CURRENT_WRAPPER_PID=""
  guard_status RUNNING --node "$node"

  if "$PYTHON_BIN" "$GUARD" assert-fingerprint \
    --baseline "$CODE_MANIFEST" \
    --expected-baseline-sha256 "$CODE_MANIFEST_SHA256" \
    --project-root "$PROJECT_ROOT" \
    --driver-path "$DRIVER_PATH" \
    --label "$node:after" \
    --observed-output \
      "$B0_ATTEMPT_ROOT/provenance/fingerprints/$node.after.json"
  then
    post_rc=0
  else
    post_rc="$?"
  fi
  if [[ "$node" != "00_preflight" && -f "$PREFLIGHT_REPORT" ]]; then
    if "$PYTHON_BIN" "$GUARD" assert-inputs \
      --input-manifest "$INPUT_MANIFEST" \
      --expected-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
      --label "$node:after" \
      --output "$B0_ATTEMPT_ROOT/provenance/input_checks/$node.after.json"
    then
      :
    elif [[ "$post_rc" -eq 0 ]]; then
      post_rc="$EX_GUARD"
    fi
  fi

  if [[ "$wrapper_rc" -ne 0 ]]; then
    finalize_failure "$wrapper_rc" "AUDITED_NODE_FAILED_${node}"
    exit "$wrapper_rc"
  fi
  if [[ "$post_rc" -ne 0 ]]; then
    finalize_failure "$post_rc" "POST_NODE_FREEZE_DRIFT_${node}"
    exit "$post_rc"
  fi

  run_named_gate "audit_completion_${node}" \
    audit-completion "$run_root/completion.json"
  run_named_gate "audit_git_binding_${node}" \
    audit-git-binding "$run_root/audit_manifest.json" \
    --expected-head "$BASELINE_HEAD" \
    --expected-dirty-state-sha256 "$BASELINE_DIRTY_STATE_SHA256"
  guard_event "NODE_COMPLETED" --node "$node"
}

PREFLIGHT_REPORT="$B0_ATTEMPT_ROOT/artifacts/preflight.json"
run_node 00_preflight \
  "$PYTHON_BIN" "$GUARD" preflight \
  --manifest "$B0_ATTEMPT_ROOT/attempt_manifest.json" \
  --expected-manifest-sha256 "$ATTEMPT_MANIFEST_SHA256" \
  --output "$PREFLIGHT_REPORT"

run_named_gate preflight preflight "$PREFLIGHT_REPORT"
INPUT_MANIFEST="$B0_ATTEMPT_ROOT/provenance/input_manifest.json"
INPUT_MANIFEST_SHA256="$(
  "$PYTHON_BIN" "$GUARD" freeze-inputs \
    --preflight "$PREFLIGHT_REPORT" \
    --node00-stdout "$B0_ATTEMPT_ROOT/audit/00_preflight/logs/stdout.log" \
    --output "$INPUT_MANIFEST" \
    --sha256-output "$B0_ATTEMPT_ROOT/provenance/input_manifest.sha256"
)"
assert_d1_inputs 00_preflight after
guard_status PREFLIGHT_PASSED

CANONICAL="$(
  "$PYTHON_BIN" -c '
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["d1"]["canonical_label_store"]["path"])
' "$PREFLIGHT_REPORT"
)"
STRUCTURAL="$(
  "$PYTHON_BIN" -c '
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["d1"]["sealed_label_free_candidate_store"]["path"])
' "$PREFLIGHT_REPORT"
)"
LEDGER="$(
  "$PYTHON_BIN" -c '
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["d1"]["exposure_ledger"]["path"])
' "$PREFLIGHT_REPORT"
)"

CANONICAL_VALIDATION="$B0_ATTEMPT_ROOT/artifacts/canonical_validation.json"
S5="$B0_ATTEMPT_ROOT/artifacts/splits/5utr_source_disjoint.json"
T5="$B0_ATTEMPT_ROOT/artifacts/splits/5utr_study_disjoint.json"
S3="$B0_ATTEMPT_ROOT/artifacts/splits/3utr_source_disjoint.json"
T3="$B0_ATTEMPT_ROOT/artifacts/splits/3utr_study_disjoint.json"
CROSS="$B0_ATTEMPT_ROOT/artifacts/splits/cross_region_transfer.json"
L5S="$B0_ATTEMPT_ROOT/artifacts/leakage/5utr_source_disjoint.json"
L5T="$B0_ATTEMPT_ROOT/artifacts/leakage/5utr_study_disjoint.json"
L3S="$B0_ATTEMPT_ROOT/artifacts/leakage/3utr_source_disjoint.json"
L3T="$B0_ATTEMPT_ROOT/artifacts/leakage/3utr_study_disjoint.json"
LCROSS="$B0_ATTEMPT_ROOT/artifacts/leakage/cross_region_transfer.json"
BUNDLE="$B0_ATTEMPT_ROOT/artifacts/bundle"
FINAL_ACCEPTANCE="$B0_ATTEMPT_ROOT/artifacts/acceptance.json"

run_node 01_canonical_validation \
  "$PYTHON_BIN" scripts/data/build_b0_splits.py \
  --records "$CANONICAL" \
  --validate-canonical-only \
  --d1-acceptance "$D1_ACCEPTANCE" \
  --output "$CANONICAL_VALIDATION"

run_named_gate canonical_validation canonical-validation "$CANONICAL_VALIDATION"

run_node 02_split_5utr_source \
  "$PYTHON_BIN" scripts/data/build_b0_splits.py \
  --records "$STRUCTURAL" \
  --split-kind source_disjoint \
  --region five_utr \
  --canonical-validation-report "$CANONICAL_VALIDATION" \
  --output "$S5"

run_node 03_split_5utr_study \
  "$PYTHON_BIN" scripts/data/build_b0_splits.py \
  --records "$STRUCTURAL" \
  --split-kind study_disjoint \
  --region five_utr \
  --canonical-validation-report "$CANONICAL_VALIDATION" \
  --output "$T5"

run_node 04_split_3utr_source \
  "$PYTHON_BIN" scripts/data/build_b0_splits.py \
  --records "$STRUCTURAL" \
  --split-kind source_disjoint \
  --region three_utr \
  --canonical-validation-report "$CANONICAL_VALIDATION" \
  --output "$S3"

run_node 05_split_3utr_study \
  "$PYTHON_BIN" scripts/data/build_b0_splits.py \
  --records "$STRUCTURAL" \
  --split-kind study_disjoint \
  --region three_utr \
  --canonical-validation-report "$CANONICAL_VALIDATION" \
  --output "$T3"

run_node 06_split_cross_region \
  "$PYTHON_BIN" scripts/data/build_b0_splits.py \
  --records "$STRUCTURAL" \
  --split-kind cross_region_transfer \
  --source-region five_utr \
  --target-region three_utr \
  --canonical-validation-report "$CANONICAL_VALIDATION" \
  --output "$CROSS"

for split in "$S5" "$T5" "$S3" "$T3" "$CROSS"; do
  run_named_gate "split_common_$(basename "$split")" \
    split-common "$split" \
    --expected-d1-acceptance "$D1_ACCEPTANCE" \
    --expected-canonical-validation "$CANONICAL_VALIDATION"
done
run_named_gate split_5utr_source_role split-5utr-source-role "$S5"
run_named_gate split_5utr_study_role split-5utr-study-role "$T5"
run_named_gate split_3utr_source_role split-3utr-source-role "$S3"
run_named_gate split_3utr_study_role split-3utr-study-role "$T3"
run_named_gate split_cross_region_role split-cross-region-role "$CROSS"

run_node 07_leakage_5utr_source \
  "$PYTHON_BIN" scripts/data/audit_b0_leakage.py \
  --records "$STRUCTURAL" \
  --split-manifest "$S5" \
  --output "$L5S"

run_node 08_leakage_5utr_study \
  "$PYTHON_BIN" scripts/data/audit_b0_leakage.py \
  --records "$STRUCTURAL" \
  --split-manifest "$T5" \
  --output "$L5T"

run_node 09_leakage_3utr_source \
  "$PYTHON_BIN" scripts/data/audit_b0_leakage.py \
  --records "$STRUCTURAL" \
  --split-manifest "$S3" \
  --output "$L3S"

run_node 10_leakage_3utr_study \
  "$PYTHON_BIN" scripts/data/audit_b0_leakage.py \
  --records "$STRUCTURAL" \
  --split-manifest "$T3" \
  --output "$L3T"

run_node 11_leakage_cross_region \
  "$PYTHON_BIN" scripts/data/audit_b0_leakage.py \
  --records "$STRUCTURAL" \
  --split-manifest "$CROSS" \
  --output "$LCROSS"

for report in "$L5S" "$L5T" "$L3S" "$L3T" "$LCROSS"; do
  run_named_gate "leakage_$(basename "$report")" leakage "$report"
done

SPLIT_ARGS=(
  --split-manifest "$S5"
  --split-manifest "$T5"
  --split-manifest "$S3"
  --split-manifest "$T3"
  --split-manifest "$CROSS"
)
REPORT_ARGS=(
  --leakage-report "$L5S"
  --leakage-report "$L5T"
  --leakage-report "$L3S"
  --leakage-report "$L3T"
  --leakage-report "$LCROSS"
)

if [[ -e "$BUNDLE" ]]; then
  abort_attempt "REFUSING_EXISTING_BUNDLE" "$EX_CANTCREAT"
fi
run_node 12_evaluation_bundle \
  "$PYTHON_BIN" scripts/data/build_b0_evaluation_artifacts.py \
  --d1-acceptance "$D1_ACCEPTANCE" \
  --canonical-records "$CANONICAL" \
  --structural-records "$STRUCTURAL" \
  "${SPLIT_ARGS[@]}" \
  "${REPORT_ARGS[@]}" \
  --exposure-ledger "$LEDGER" \
  --output-root "$BUNDLE"

BUNDLE_MANIFEST="$BUNDLE/build_manifest.json"
run_named_gate evaluation_bundle evaluation-bundle "$BUNDLE_MANIFEST"
BUNDLE_BINDING="$B0_ATTEMPT_ROOT/provenance/bundle_binding.json"
BUNDLE_BINDING_SHA256="$(
  "$PYTHON_BIN" "$GUARD" bind-result \
    --kind bundle \
    --artifact "$BUNDLE_MANIFEST" \
    --output "$BUNDLE_BINDING" \
    --sha256-output "$B0_ATTEMPT_ROOT/provenance/bundle_binding.sha256"
)"

TRACK_ARGS=(
  --track-manifest "$BUNDLE/evaluation/tracks/closed_measured_pool.yaml"
  --track-manifest "$BUNDLE/evaluation/tracks/heldout_generative.yaml"
  --track-manifest "$BUNDLE/evaluation/tracks/open_legal_generation.yaml"
)
run_node 13_final_acceptance \
  "$PYTHON_BIN" scripts/data/validate_b0_acceptance.py \
  --records "$STRUCTURAL" \
  --d1-acceptance "$D1_ACCEPTANCE" \
  --exposure-ledger "$LEDGER" \
  --exposure-identity-level dataset_id \
  "${SPLIT_ARGS[@]}" \
  "${REPORT_ARGS[@]}" \
  "${TRACK_ARGS[@]}" \
  --artifact-bindings "$BUNDLE/artifact_bindings.json" \
  --output "$FINAL_ACCEPTANCE"

run_named_gate final_acceptance final-acceptance "$FINAL_ACCEPTANCE"
ACCEPTANCE_BINDING="$B0_ATTEMPT_ROOT/provenance/acceptance_binding.json"
ACCEPTANCE_BINDING_SHA256="$(
  "$PYTHON_BIN" "$GUARD" bind-result \
    --kind acceptance \
    --artifact "$FINAL_ACCEPTANCE" \
    --output "$ACCEPTANCE_BINDING" \
    --sha256-output "$B0_ATTEMPT_ROOT/provenance/acceptance_binding.sha256"
)"

assert_code_fingerprint 13_final_seal before
assert_d1_inputs 13_final_seal before
guard_event "B0_ACCEPTANCE_PASSED_PENDING_DRIVER_SEAL" \
  --detail "$FINAL_ACCEPTANCE"
stop_watchdog
if "$PYTHON_BIN" "$GUARD" seal \
    --attempt-root "$B0_ATTEMPT_ROOT" \
    --final-acceptance "$FINAL_ACCEPTANCE" \
    --bundle-manifest "$BUNDLE_MANIFEST" \
    --code-manifest "$CODE_MANIFEST" \
    --input-manifest "$INPUT_MANIFEST" \
    --acceptance-binding "$ACCEPTANCE_BINDING" \
    --bundle-binding "$BUNDLE_BINDING" \
    --expected-attempt-manifest-sha256 "$ATTEMPT_MANIFEST_SHA256" \
    --expected-code-manifest-sha256 "$CODE_MANIFEST_SHA256" \
    --expected-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
    --expected-acceptance-binding-sha256 "$ACCEPTANCE_BINDING_SHA256" \
    --expected-bundle-binding-sha256 "$BUNDLE_BINDING_SHA256"
then
  :
else
  seal_rc="$?"
  if "$PYTHON_BIN" "$GUARD" terminal-success \
    --attempt-root "$B0_ATTEMPT_ROOT" >/dev/null 2>&1
  then
    ATTEMPT_COMPLETE=1
    CURRENT_NODE=""
    trap - ERR INT TERM HUP
    exit 0
  fi
  finalize_failure "$seal_rc" "DRIVER_SEAL_FAILED"
  exit "$seal_rc"
fi
ATTEMPT_COMPLETE=1
CURRENT_NODE=""
trap - ERR INT TERM HUP
exit 0
