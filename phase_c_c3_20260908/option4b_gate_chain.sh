#!/usr/bin/env bash
# Option 4b gate: after full891 harvest lands, if beta 0.25 passes the B2 delta
# gate at 891 scale, auto-launch B=256 guided beta*=0.25; then auto-harvest.
set -u
W=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901
J=/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902
X=/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5
D=$X/beta_full891_20260909
Q=$X/pool256_guided_20260909
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
LOG=$Q/gate_watch.log
DONE=$Q/done_gate.json
mkdir -p "$Q"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "option-4b gate watcher started (pid $$)"
while true; do
    [ -f "$DONE" ] && { log "done marker present; exiting"; exit 0; }
    H=$D/full891_confirm_harvest.json
    if [ ! -f "$H" ]; then
        log "harvest not yet; waiting"
        sleep 1800
        continue
    fi
    log "harvest present; evaluating 4b gate"
    PASS=$("$PY" - <<PYEOF
import json
d = json.load(open("$H"))
r = d["rows"].get("beta_0.25", {})
pt = r.get("d_sc_hit1_point")
ci = r.get("d_sc_hit1_ci95")
ok = bool(pt is not None and ci is not None and pt >= 0.03 and ci[0] > 0)
print("1" if ok else "0")
PYEOF
)
    log "gate evaluation: beta_0.25 d_sc_hit1 pass=$PASS"
    if [ "$PASS" != "1" ]; then
        log "beta_0.25 did not pass B2 delta gate at 891 scale -> option 4b NOT launched (negative result path, see amendment honesty clause); exiting"
        cat >> "$J/docs/training_journal/TRAINING_LOG_202609.md" <<EOF

### 批次五十九（自动 4b 门裁决：β=0.25 891 尺度未过 B2 Δ门 → 4b 不发射，负结果路径收口）

- full891 收割后自动门评估：β=0.25 Δsc-hit@1 未达 +0.03 且 CI 不跨零 → calib100 过门系子集偏差。选项 4b（B=256 guided）不发射（amendment 诚实性条款：不追加第六方向）。Phase C 全链终态 = 负结果收口（β 杠杆存在但 calib100 尺度不外推），详档 full891_confirm_harvest.json。
EOF
        (cd "$J" && git add docs/training_journal/TRAINING_LOG_202609.md && git commit -q -m "train journal: auto - option 4b gate NEGATIVE (beta 0.25 fails B2 delta gate at 891 scale); negative-result closure path" && git push -q origin route-a-v3-w0-diagnosis-20260902) >> "$LOG" 2>&1
        echo "{\"4b_launched\":false,\"timestamp\":\"$(date -Iseconds)\"}" > "$DONE"
        exit 0
    fi
    break
done

# Gate passed -> launch B=256 guided beta 0.25 (unguided B=256 baseline already
# exists from option 4a: pool256_unguided_20260908).
log "launching option 4b: B=256 guided beta=0.25"
OUT=$Q/b256_guided_beta025
if [ ! -f "$OUT/guided_run_summary.json" ]; then
    cd "$W" && $PY scripts/route_a_v3/run_route2_guided_xeditsetflow_v5_v1.py \
        --config configs/route_a_v3_route2_xeditsetflow_v5_screen_v1.json \
        --run-id b_fix2 --checkpoint-pass 2 --physical-gpu-index 4 \
        --output-dir "$OUT" \
        --screen-gate "$X/screen_20260915/screen_gate.json" \
        --arms guided --beta 0.25 --trajectory-count 256 \
        >> "$Q/b256_guided_beta025.log" 2>&1
    rc=$?
    log "4b run rc=$rc"
else
    log "4b already terminal; skip"
fi

# Final harvest: B=256 guided vs B=256 unguided (option 4a) + B=32 guided beta .25
log "running 4b final comparison harvest"
$PY - <<PYEOF >> "$LOG" 2>&1
import json
from collections import defaultdict
import random

X = "$X"
Q = "$Q"
MEAS = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/development_validation_v1/measured_neighborhood.private.jsonl"

def norm(s):
    return str(s).upper().replace("T", "U")

measured = defaultdict(set)
for line in open(MEAS):
    r = json.loads(line)
    measured[r["source_key"]].add(norm(r["candidate_sequence"]))

def per_source(path):
    gen = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        sk = r["source_key"]
        sq = norm(r["candidate_sequence"])
        s = float(r.get("generation_score") or 0.0)
        if sq not in gen[sk] or gen[sk][sq] < s:
            gen[sk][sq] = s
    out = {}
    for sk, seqs in gen.items():
        ms = measured.get(sk, set())
        covered = [q for q in ms if q in seqs]
        h1 = 0.0
        if covered:
            ranked = sorted(seqs.items(), key=lambda kv: -kv[1])
            top = ranked[0][1]
            block = [sq for sq, s in ranked if s == top]
            tie = len([sq for sq in block if sq in ms])
            h1 = tie / len(block)
        out[sk] = {"support": 1.0 if covered else 0.0,
                   "recovery": (len(covered)/len(ms)) if ms else 0.0,
                   "sc_hit1": h1 if covered else None}
    return out

def bci(d, iters=2000, seed=20260816):
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(iters):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s)/len(s))
    means.sort()
    return [means[int(0.025*iters)], means[int(0.975*iters)]]

base = per_source(X + "/pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl")
guided = per_source(Q + "/b256_guided_beta025/guided/generated_candidates.private.jsonl")
vs = [v["sc_hit1"] for v in guided.values() if v["sc_hit1"] is not None]
d_sc = [(guided[k]["sc_hit1"] or 0.0) - (base.get(k, {}).get("sc_hit1") or 0.0)
        for k in guided
        if (guided[k]["sc_hit1"] is not None or base.get(k, {}).get("sc_hit1") is not None)]
d_sup = [guided[k]["support"] - base.get(k, {}).get("support", 0.0) for k in guided]
b_vs = [v["sc_hit1"] for v in base.values() if v["sc_hit1"] is not None]
res = {
    "schema_version": "route_a_v3_phase_c_4b_final.v1",
    "comparison": "B256 guided beta0.25 vs B256 unguided (option 4a baseline)",
    "gate": "amendment A-tier: sc_hit1 >= 0.30 AND d_support >= +0.10 (vs B32), dual-report",
    "B256_unguided": {"support": sum(v["support"] for v in base.values())/len(base),
                      "recovery": sum(v["recovery"] for v in base.values())/len(base),
                      "sc_hit1_mean": (sum(b_vs)/len(b_vs)) if b_vs else None,
                      "sc_hit1_n": len(b_vs)},
    "B256_guided_beta025": {"support": sum(v["support"] for v in guided.values())/len(guided),
                            "recovery": sum(v["recovery"] for v in guided.values())/len(guided),
                            "sc_hit1_mean": (sum(vs)/len(vs)) if vs else None,
                            "sc_hit1_n": len(vs)},
    "d_support_point": sum(d_sup)/len(d_sup),
    "d_support_ci95": bci(d_sup),
    "d_sc_hit1_point": (sum(d_sc)/len(d_sc)) if d_sc else None,
    "d_sc_hit1_ci95": bci(d_sc) if d_sc else None,
}
json.dump(res, open(Q + "/final_4b_comparison.json", "w"), indent=1)
print(json.dumps(res, indent=1))
PYEOF
rc2=$?
cat >> "$J/docs/training_journal/TRAINING_LOG_202609.md" <<EOF

### 批次六十（自动 4b 收割：B=256 guided β=0.25 vs B=256 unguided 终态对位）

- 4b 门通过（β=0.25 891 过门）→ B=256 guided 自动发射并终态 → \`final_4b_comparison.json\`（sc-hit@1/support/recovery + ΔCI，amendment A 档口径：A 档 sc-hit@1 ≥ 0.30 且 Δsupport(vs B=32) ≥ +0.10）。**A 档完整判定含 vs B=32 的 Δsupport——以 full891_confirm_harvest.json 的 beta_0.25 support 与 4a unguided B=32 support(0.242) 差值核对**。待会话复核终判。
EOF
(cd "$J" && git add docs/training_journal/TRAINING_LOG_202609.md && git commit -q -m "train journal: auto batch - option 4b final harvest (B256 guided vs unguided)" && git push -q origin route-a-v3-w0-diagnosis-20260902) >> "$LOG" 2>&1
(cd "$W" && git add phase_c_c3_20260908/ 2>/dev/null; git commit -q -m "analysis(route2): option 4b auto-gate + final comparison products" 2>/dev/null; git push -q origin route-a-v3-setflow-v5-base-fix-20260901) >> "$LOG" 2>&1
echo "{\"4b_launched\":true,\"harvest_rc\":$rc2,\"timestamp\":\"$(date -Iseconds)\"}" > "$DONE"
log "4b chain complete; exiting"
exit $rc2
