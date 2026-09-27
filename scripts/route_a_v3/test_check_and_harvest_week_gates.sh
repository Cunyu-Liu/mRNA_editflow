#!/bin/bash
# Fixture test for the reconfigured dispatch logic in check_and_harvest_week.sh.
# Exercises every branch of the two gated harvests (M1 single-arm, M1 3-seed, polyA 3-seed)
# on throwaway directories + stub harvesters, so the evidence directories are never touched.
#
# Note: the monitor invokes harvesters as `$PY <script> ...`, so the stubs are *python* scripts.
# Run: bash test_check_and_harvest_week_gates.sh
set -u

PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
WT=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
MON="$WT/scripts/route_a_v3/check_and_harvest_week.sh"
T=$(mktemp -d /tmp/mrna_mon_fixture_XXXXXX)
trap 'rm -rf "$T"' EXIT
export STUB_LOG="$T/stub_calls.log"
FAIL=0
pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; FAIL=1; }

write_m1_stub() { # write_m1_stub <python-bool-literal: True|False>
  cat > "$T/stub_m1.py" <<EOF
import json, os, pathlib, sys
pathlib.Path(os.environ["STUB_LOG"]).open("a").write("m1 " + " ".join(sys.argv[1:]) + "\n")
out = pathlib.Path(os.environ["MRNA_MON_OUT_M1"]); out.mkdir(parents=True, exist_ok=True)
json.dump({"G1": {"direction_positive": $1}}, (out / "m1_adjudication_v1.json").open("w"))
EOF
}

cat > "$T/stub_m1x.py" <<'EOF'
import json, os, pathlib, sys
pathlib.Path(os.environ["STUB_LOG"]).open("a").write("m1x " + " ".join(sys.argv[1:]) + "\n")
out = pathlib.Path(os.environ["MRNA_MON_OUT_M1"]); out.mkdir(parents=True, exist_ok=True)
json.dump({"stub": "m1_3seed_ensemble"}, (out / "m1_3seed_ensemble_v1.json").open("w"))
EOF

cat > "$T/stub_p3.py" <<'EOF'
import json, os, pathlib, sys
pathlib.Path(os.environ["STUB_LOG"]).open("a").write("p3 " + " ".join(sys.argv[1:]) + "\n")
out = pathlib.Path(os.environ["MRNA_MON_POLYA"]) / "harvest_v1"; out.mkdir(parents=True, exist_ok=True)
json.dump({"stub": "polya_3seed"}, (out / "polya_3seed_harvest_v1.json").open("w"))
EOF

cat > "$T/stub_launcher.sh" <<'EOF'
#!/bin/bash
echo "[stub-launcher] called with: $*"
EOF
chmod +x "$T/stub_launcher.sh"

run_mon() { # run_mon <label> -> writes $T/out_<label>.txt
  MRNA_MON_M1BASE="$T/base" MRNA_MON_POLYA="$T/polya" MRNA_MON_OUT_M1="$T/out_m1" \
  MRNA_MON_HARVEST_M1="$T/stub_m1.py" MRNA_MON_HARVEST_M1X="$T/stub_m1x.py" \
  MRNA_MON_HARVEST_P3="$T/stub_p3.py" MRNA_MON_LAUNCHER="$T/stub_launcher.sh" \
    bash "$MON" > "$T/out_$1.txt" 2>&1
}
quiet() { grep -q "$1" "$T/out_$2.txt"; }

echo "== fixture root: $T =="

echo "[branch 1] nothing terminal, no adjudication -> all deferred"
mkdir -p "$T/base/seed_20260920" "$T/polya" "$T/out_m1"
run_mon b1
quiet "\[harvest\] M1 3-seed: deferred (single-arm adjudication absent)" b1 \
  && pass "3-seed deferred when adjudication absent" || fail "3-seed absent-branch message missing"
quiet "\[harvest\] polyA 3-seed: deferred (final per-seed predictions absent)" b1 \
  && pass "polyA deferred message present" || fail "polyA deferred message missing"
quiet "\[stub-launcher\] called" b1 && pass "launcher pass invoked" || fail "launcher not invoked"
[ ! -s "$STUB_LOG" ] && pass "no harvester invoked when gates closed" || fail "harvester invoked prematurely"

echo "[branch 2] M1 arm terminal + G1 negative -> single-arm harvest runs, 3-seed SKIPPED"
echo '{"terminal": true}' > "$T/base/seed_20260920/run_summary.json"
write_m1_stub False
run_mon b2
quiet "\[harvest\] M1 OK" b2 && pass "single-arm harvest ran" || fail "single-arm harvest did not run"
quiet "M1 3-seed: SKIPPED (G1 direction_positive=False" b2 \
  && pass "3-seed skipped on negative direction" || fail "negative-direction skip message missing"
grep -q "^m1x" "$STUB_LOG" 2>/dev/null && fail "3-seed harvester ran despite negative G1" \
  || pass "3-seed harvester NOT run on negative G1"

echo "[branch 3] G1 positive but extension arms not terminal -> deferred"
rm -f "$T/out_m1/m1_adjudication_v1.json"
write_m1_stub True
run_mon b3
quiet "M1 3-seed: deferred (G1 positive but extension arms not all terminal)" b3 \
  && pass "3-seed deferred while arms unfinished" || fail "positive-but-unfinished message missing"
grep -q "^m1x" "$STUB_LOG" 2>/dev/null && fail "3-seed harvester ran with unfinished arms" \
  || pass "3-seed harvester NOT run while arms unfinished"

echo "[branch 4] G1 positive + all three arms terminal -> 3-seed harvest runs, then idempotent"
mkdir -p "$T/base/seed_20260904" "$T/base/seed_20260905"
echo '{"terminal": true}' > "$T/base/seed_20260904/run_summary.json"
echo '{"terminal": true}' > "$T/base/seed_20260905/run_summary.json"
rm -f "$T/out_m1/m1_adjudication_v1.json"
run_mon b4
quiet "\[harvest\] M1 3-seed OK" b4 && pass "3-seed harvest ran" || fail "3-seed harvest did not run"
grep -q "^m1x" "$STUB_LOG" 2>/dev/null && pass "3-seed harvester invoked through the gate" || fail "3-seed harvester never invoked"
[ -f "$T/out_m1/m1_3seed_ensemble_v1.json" ] && pass "3-seed output written" || fail "3-seed output missing"
run_mon b4b
quiet "M1 3-seed already harvested" b4b && pass "idempotent re-run" || fail "re-run not idempotent"

echo "[branch 5] polyA finals present -> polyA aggregation runs, then idempotent"
mkdir -p "$T/polya/seed_20260921" "$T/polya/seed_20260922"
touch "$T/polya/seed_20260921/final_validation_predictions.jsonl" "$T/polya/seed_20260922/final_validation_predictions.jsonl"
run_mon b5
quiet "\[harvest\] polyA 3-seed OK" b5 && pass "polyA aggregation ran" || fail "polyA aggregation did not run"
run_mon b5b
quiet "polyA 3-seed already harvested" b5b && pass "polyA idempotent re-run" || fail "polyA re-run not idempotent"

echo "== stub invocation log =="
cat "$STUB_LOG" 2>/dev/null | sed 's/^/  /'
echo "== RESULT: $([ "$FAIL" = "0" ] && echo ALL_PASS || echo FAILURES) =="
exit "$FAIL"