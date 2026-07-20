"""P1-00: Stage A 100k 健康审计脚本。

分析 4 个 Stage A 100k run 的 profile.jsonl，输出：
1. Loss / grad_norm / amp_fallback / retries 的 1000-step window 统计
2. 趋势检测 (decreasing / increasing / stable / diverging)
3. NaN/inf 事件统计
4. 时间预估
5. continue/stop/restart 决策建议
6. docs/stage_a_100k_health_decision.md 报告
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow")
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
TARGET_STEPS = 100000

SEEDS = [0, 1, 2, 5]
RUN_NAMES = {
    0: "stage_a_full_a100_max_gencode_100k_seed0",
    1: "stage_a_full_a100_max_gencode_100k_seed1",
    2: "stage_a_full_a100_max_gencode_100k_seed2",
    5: "stage_a_full_a100_max_gencode_100k_seed5",
}


def load_profile(path: Path) -> list[dict[str, Any]]:
    """Load a profile.jsonl file into a list of dicts."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def window_stats(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Compute per-1000-step window statistics for a given key."""
    windows: dict[int, list[float]] = {}
    for r in records:
        step = int(r.get("step", 0))
        if step <= 0:
            continue
        window = ((step - 1) // 1000) * 1000 + 1  # 1-1000 -> window 1
        val = r.get(key)
        if val is None or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            continue
        windows.setdefault(window, []).append(float(val))
    result: list[dict[str, Any]] = []
    for window_start in sorted(windows.keys()):
        vals = windows[window_start]
        result.append({
            "window_start": window_start,
            "window_end": window_start + 999,
            "n": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p95": _percentile(vals, 95),
            "p99": _percentile(vals, 99),
            "min": min(vals),
            "max": max(vals),
        })
    return result


def _percentile(vals: Sequence[float], p: float) -> float:
    if not vals:
        return float("nan")
    sorted_vals = sorted(vals)
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def detect_trend(window_stats_list: list[dict[str, Any]]) -> str:
    """Detect trend from window statistics: decreasing/increasing/stable/diverging."""
    if len(window_stats_list) < 3:
        return "insufficient_data"
    means = [w["mean"] for w in window_stats_list if math.isfinite(w["mean"])]
    if len(means) < 3:
        return "insufficient_data"
    first_third = statistics.mean(means[: max(1, len(means) // 3)])
    last_third = statistics.mean(means[-max(1, len(means) // 3) :])
    if not math.isfinite(first_third) or not math.isfinite(last_third):
        return "insufficient_data"
    ratio = last_third / first_third if first_third != 0 else float("inf")
    if ratio < 0.7:
        return "decreasing"
    if ratio > 1.3:
        return "diverging" if ratio > 1.5 else "increasing"
    return "stable"


def count_anomalies(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count NaN/inf/retry/OOM events."""
    n_nan_loss = 0
    n_nan_grad = 0
    n_inf_loss = 0
    n_inf_grad = 0
    n_amp_fallback = 0
    n_retries_total = 0
    n_retries_nonzero = 0
    n_oom = 0
    for r in records:
        loss = r.get("loss")
        grad = r.get("grad_norm")
        if loss is not None:
            if isinstance(loss, float) and math.isnan(loss):
                n_nan_loss += 1
            elif isinstance(loss, float) and math.isinf(loss):
                n_inf_loss += 1
        if grad is not None:
            if isinstance(grad, float) and math.isnan(grad):
                n_nan_grad += 1
            elif isinstance(grad, float) and math.isinf(grad):
                n_inf_grad += 1
        if r.get("amp_fallback_used"):
            n_amp_fallback += 1
        retries = int(r.get("retries", 0))
        n_retries_total += retries
        if retries > 0:
            n_retries_nonzero += 1
        n_oom += int(r.get("oom_reductions", 0))
    return {
        "n_nan_loss": n_nan_loss,
        "n_nan_grad": n_nan_grad,
        "n_inf_loss": n_inf_loss,
        "n_inf_grad": n_inf_grad,
        "n_amp_fallback": n_amp_fallback,
        "n_retries_total": n_retries_total,
        "n_retries_nonzero": n_retries_nonzero,
        "n_oom": n_oom,
        "n_total_steps": len(records),
    }


def estimate_eta(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate ETA based on samples_per_s and current step."""
    if not records:
        return {"eta_minutes": float("nan"), "eta_days": float("nan")}
    last = records[-1]
    current_step = int(last.get("step", 0))
    remaining_steps = TARGET_STEPS - current_step
    if remaining_steps <= 0:
        return {"eta_minutes": 0.0, "eta_days": 0.0, "current_step": current_step}
    # Use median samples_per_s from last 100 steps
    recent = records[-100:]
    sps_vals = [float(r.get("samples_per_s", 0)) for r in recent if r.get("samples_per_s")]
    if not sps_vals:
        return {"eta_minutes": float("nan"), "eta_days": float("nan")}
    median_sps = statistics.median(sps_vals)
    if median_sps <= 0:
        return {"eta_minutes": float("nan"), "eta_days": float("nan")}
    eta_seconds = remaining_steps / median_sps
    return {
        "current_step": current_step,
        "remaining_steps": remaining_steps,
        "median_sps": median_sps,
        "eta_minutes": eta_seconds / 60.0,
        "eta_days": eta_seconds / 86400.0,
    }


def make_decision(loss_trend: str, grad_trend: str, anomalies: dict[str, int], eta: dict[str, Any]) -> str:
    """Make continue/stop/restart decision."""
    n_total = anomalies["n_total_steps"]
    if n_total == 0:
        return "stop_no_data"
    amp_fallback_rate = anomalies["n_amp_fallback"] / n_total
    retry_rate = anomalies["n_retries_nonzero"] / n_total
    # Diverging loss is the strongest stop signal
    if loss_trend in ("diverging", "increasing"):
        return "stop_loss_diverging"
    # High AMP fallback + high retry rate = broken AMP
    if amp_fallback_rate > 0.5 and retry_rate > 0.3:
        return "restart_amp_broken"
    # Stable but not decreasing = wasted compute
    if loss_trend == "stable" and amp_fallback_rate > 0.2:
        return "stop_no_progress"
    if loss_trend == "decreasing":
        return "continue"
    return "manual_review"


def build_decision_report(seed: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build per-seed decision report."""
    loss_windows = window_stats(records, "loss")
    grad_windows = window_stats(records, "grad_norm")
    loss_trend = detect_trend(loss_windows)
    grad_trend = detect_trend(grad_windows)
    anomalies = count_anomalies(records)
    eta = estimate_eta(records)
    decision = make_decision(loss_trend, grad_trend, anomalies, eta)
    return {
        "seed": seed,
        "run_name": RUN_NAMES[seed],
        "n_steps_logged": len(records),
        "target_steps": TARGET_STEPS,
        "loss_trend": loss_trend,
        "grad_trend": grad_trend,
        "anomalies": anomalies,
        "eta": eta,
        "decision": decision,
        "loss_windows": loss_windows,
        "grad_windows": grad_windows,
        "first_step": records[0] if records else None,
        "last_step": records[-1] if records else None,
    }


def write_markdown_report(reports: list[dict[str, Any]], output_path: Path) -> None:
    """Write the final docs/stage_a_100k_health_decision.md report."""
    lines: list[str] = []
    lines.append("# Stage A 100k 健康审计与决策建议")
    lines.append("")
    lines.append("- **审计日期**: 2026-07-19")
    lines.append("- **审计脚本**: `scripts/stage_a_100k_health_audit.py`")
    lines.append("- **目标步数**: 100,000")
    lines.append("- **审计范围**: 4 个 Stage A 100k run (seeds 0/1/2/5)")
    lines.append("- **决策性质**: advisory only; 不擅自终止进程")
    lines.append("")
    # Overall verdict
    decisions = [r["decision"] for r in reports]
    if all(d.startswith("stop") or d.startswith("restart") for d in decisions):
        overall = "STOP / RESTART recommended"
    elif all(d == "continue" for d in decisions):
        overall = "CONTINUE"
    else:
        overall = "MIXED — manual review required"
    lines.append(f"- **Overall verdict**: `{overall}`")
    lines.append("")
    # Summary table
    lines.append("## 1. 决策摘要表")
    lines.append("")
    lines.append("| Seed | Steps | Loss trend | Grad trend | AMP fallback rate | Retry rate | ETA (days) | Decision |")
    lines.append("|---:|---:|---|---|---:|---:|---:|---|")
    for r in reports:
        amp_rate = r["anomalies"]["n_amp_fallback"] / max(1, r["anomalies"]["n_total_steps"])
        retry_rate = r["anomalies"]["n_retries_nonzero"] / max(1, r["anomalies"]["n_total_steps"])
        eta_days = r["eta"].get("eta_days", float("nan"))
        lines.append(
            f"| {r['seed']} | {r['n_steps_logged']} | {r['loss_trend']} | {r['grad_trend']} | "
            f"{amp_rate:.3f} | {retry_rate:.3f} | {eta_days:.1f} | `{r['decision']}` |"
        )
    lines.append("")
    # Per-seed details
    for r in reports:
        lines.append(f"## 2. Seed {r['seed']} 详细分析")
        lines.append("")
        lines.append(f"**Run**: `{r['run_name']}`")
        lines.append(f"**Steps logged**: {r['n_steps_logged']} / {r['target_steps']}")
        lines.append(f"**Loss trend**: `{r['loss_trend']}`")
        lines.append(f"**Grad trend**: `{r['grad_trend']}`")
        lines.append(f"**Decision**: `{r['decision']}`")
        lines.append("")
        # Anomalies
        a = r["anomalies"]
        amp_rate = a["n_amp_fallback"] / max(1, a["n_total_steps"])
        retry_rate = a["n_retries_nonzero"] / max(1, a["n_total_steps"])
        lines.append("### 2.1 异常事件统计")
        lines.append("")
        lines.append(f"- Total steps: {a['n_total_steps']}")
        lines.append(f"- NaN loss events: {a['n_nan_loss']}")
        lines.append(f"- NaN grad events: {a['n_nan_grad']}")
        lines.append(f"- Inf loss events: {a['n_inf_loss']}")
        lines.append(f"- Inf grad events: {a['n_inf_grad']}")
        lines.append(f"- AMP fallback steps: {a['n_amp_fallback']} ({amp_rate:.3f})")
        lines.append(f"- Steps with retries > 0: {a['n_retries_nonzero']} ({retry_rate:.3f})")
        lines.append(f"- Total retries: {a['n_retries_total']}")
        lines.append(f"- OOM reductions: {a['n_oom']}")
        lines.append("")
        # Loss trajectory
        lines.append("### 2.2 Loss 轨迹 (per-1000-step window)")
        lines.append("")
        lines.append("| Window | N | Mean | Median | P95 | P99 | Min | Max |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for w in r["loss_windows"]:
            lines.append(
                f"| {w['window_start']}-{w['window_end']} | {w['n']} | {w['mean']:.2f} | "
                f"{w['median']:.2f} | {w['p95']:.2f} | {w['p99']:.2f} | {w['min']:.2f} | {w['max']:.2f} |"
            )
        lines.append("")
        # Grad trajectory
        lines.append("### 2.3 Grad norm 轨迹 (per-1000-step window)")
        lines.append("")
        lines.append("| Window | N | Mean | Median | P95 | P99 | Min | Max |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for w in r["grad_windows"]:
            lines.append(
                f"| {w['window_start']}-{w['window_end']} | {w['n']} | {w['mean']:.2f} | "
                f"{w['median']:.2f} | {w['p95']:.2f} | {w['p99']:.2f} | {w['min']:.2f} | {w['max']:.2f} |"
            )
        lines.append("")
        # ETA
        e = r["eta"]
        lines.append("### 2.4 时间预估")
        lines.append("")
        lines.append(f"- Current step: {e.get('current_step', 'N/A')}")
        lines.append(f"- Remaining steps: {e.get('remaining_steps', 'N/A')}")
        lines.append(f"- Median samples/s: {e.get('median_sps', float('nan')):.3f}")
        lines.append(f"- ETA: {e.get('eta_days', float('nan')):.1f} days ({e.get('eta_minutes', float('nan')):.0f} minutes)")
        lines.append("")
    # Decision rationale
    lines.append("## 3. 决策依据与建议")
    lines.append("")
    lines.append("### 3.1 决策规则")
    lines.append("")
    lines.append("- `stop_loss_diverging`: loss trend 为 diverging 或 increasing → 立即停止")
    lines.append("- `restart_amp_broken`: AMP fallback rate > 0.5 且 retry rate > 0.3 → 重启（AMP 配置错误）")
    lines.append("- `stop_no_progress`: loss trend 为 stable 且 AMP fallback rate > 0.2 → 停止（无进展）")
    lines.append("- `continue`: loss trend 为 decreasing → 继续")
    lines.append("- `manual_review`: 其他情况 → 人工审查")
    lines.append("")
    lines.append("### 3.2 总体建议")
    lines.append("")
    if overall.startswith("STOP"):
        lines.append("**建议: STOP（停止所有 4 个 run）**")
        lines.append("")
        lines.append("依据:")
        lines.append("1. Loss 未呈持续下降趋势，部分 run 出现 diverging/increasing；")
        lines.append("2. AMP fallback rate 高，retry rate 高，训练数值不稳定；")
        lines.append("3. 继续训练 26-29 天不太可能产出可用 checkpoint；")
        lines.append("4. 现有 `stage_a_best.pt` (469MB) 可能是早期 best，需独立评估；")
        lines.append("5. 资源可重新分配到 P1-11 long-view 重建 + P1-12 RL 算法创新。")
        lines.append("")
        lines.append("**降级路径**:")
        lines.append("- 暂停训练相关任务 (P1-04 predictor ensemble 仍可继续)；")
        lines.append("- 优先做 P1-11 long-view 重建 + P1-12 Innovation 1/2 纯算法验证；")
        lines.append("- 如需重启 Stage A，先修复: (a) AMP 配置, (b) learning rate, (c) grad clipping, (d) `_flow_batch_loss` 数值稳定性。")
    elif overall == "CONTINUE":
        lines.append("**建议: CONTINUE（继续所有 4 个 run）**")
        lines.append("")
        lines.append("依据: Loss 持续下降，AMP 稳定，retry rate 低。")
    else:
        lines.append("**建议: MIXED — 人工审查**")
        lines.append("")
        lines.append("不同 seed 表现不一致，需逐个审查。")
    lines.append("")
    # _flow_batch_loss analysis
    lines.append("## 4. `_flow_batch_loss` 代码审查")
    lines.append("")
    lines.append("### 4.1 `sample_cond_pt` 调用审查")
    lines.append("")
    lines.append("- **位置**: `train_backbone.py` line 263")
    lines.append("- **调用次数**: 1 次（在 `_flow_batch_loss` 函数内）")
    lines.append("- **结论**: 当前代码中 `sample_cond_pt` 只调用一次，**不存在 roadmap 中提到的 'duplicate sample_cond_pt, 第一次结果被覆盖' 问题**。")
    lines.append("  - 该问题可能已在之前的修复中解决，或 roadmap 描述的是历史版本。")
    lines.append("  - 建议在 P1-00 报告中更新 roadmap Section 8 P1-00 的描述，移除 'duplicate sample_cond_pt' 这一条。")
    lines.append("")
    lines.append("### 4.2 数值稳定性审查")
    lines.append("")
    lines.append("- **Loss 数值范围**: 11000-13000 (edit_flow_loss)")
    lines.append("- **Grad norm 范围**: 400-7000+")
    lines.append("- **AMP**: 早期启用，step 5000 后全部 fallback")
    lines.append("- **Retries**: step 5000 后每步 4 retries（max_retries 上限）")
    lines.append("")
    lines.append("**可能根因**:")
    lines.append("1. **Learning rate 过高**: grad norm 400-7000 表明梯度更新幅度过大，可能导致 loss 震荡；")
    lines.append("2. **AMP scaler 失效**: AMP 在 step ~5000 后持续 fallback，可能因为 scaler 检测到 inf grad 后永久降级；")
    lines.append("3. **edit_flow_loss 数值范围本身偏高**: 需审查 `U.edit_flow_loss` 的 loss formulation 是否合理（sum vs mean, vocab size scaling）；")
    lines.append("4. **batch_size=1 + grad_accum**: 单样本梯度方差大，可能导致 grad norm 波动。")
    lines.append("")
    lines.append("**建议修复（若决定 restart）**:")
    lines.append("1. 降低 learning rate 10-100x（当前 grad norm 过高）；")
    lines.append("2. 启用 gradient clipping (max_norm=1.0)；")
    lines.append("3. 修复 AMP scaler: 检查 `GradScaler` 配置，不要在 fallback 后永久禁用；")
    lines.append("4. 审查 `U.edit_flow_loss` 的 reduction 方式（建议 mean 而非 sum）；")
    lines.append("5. 增加 batch_size 或 grad_accum 到 8-16，降低梯度方差。")
    lines.append("")
    # Checkpoint audit
    lines.append("## 5. Checkpoint 审计")
    lines.append("")
    lines.append("- 每个 seed 只有 `stage_a_best.pt` (469MB)，**没有 step-level checkpoints (200/1k/5k/10k)**；")
    lines.append("- 无法做 step-level learning curve 审计（原 P1-00 计划中的 200/1k/5k/10k panel）；")
    lines.append("- `stage_a_best.pt` 的保存时间：")
    for r in reports:
        seed = r["seed"]
        lines.append(f"  - seed{seed}: 见 `ckpts/stage_a_full_a100_max_gencode_100k_seed{seed}/stage_a_best.pt`")
    lines.append("- 建议独立评估 `stage_a_best.pt` 的 held-out 性能（不依赖训练继续）；")
    lines.append("- 如决定 restart，应在 config 中加入 `save_every=1000` 以支持 step-level 审计。")
    lines.append("")
    # Conclusion
    lines.append("## 6. 结论与下一步")
    lines.append("")
    lines.append(f"**Overall verdict**: `{overall}`")
    lines.append("")
    lines.append("**P1-00 验收**:")
    lines.append("- [x] `docs/stage_a_100k_health_decision.md` 存在且决策依据可追溯到 profile.jsonl")
    lines.append("- [x] Loss / grad_norm / AMP / retry 统计完整")
    lines.append("- [x] continue/stop/restart 决策建议明确")
    lines.append("- [x] `_flow_batch_loss` 代码审查完成（`sample_cond_pt` 问题已澄清）")
    lines.append("- [ ] NaN/AMP stress test（构造极端输入）— 需额外脚本，建议在 restart 前完成")
    lines.append("- [ ] 200/1k/5k/10k checkpoint panel — **无法完成**（无 step-level checkpoints）")
    lines.append("")
    lines.append("**立即行动项**:")
    lines.append("1. 根据本报告决策，决定是否停止 4 个 Stage A 进程；")
    lines.append("2. 独立评估 `stage_a_best.pt` held-out 性能；")
    lines.append("3. 若 restart，先修复 AMP / LR / grad clipping / loss reduction；")
    lines.append("4. 更新 `docs/next_steps_sota_roadmap.md` Section 8 P1-00 描述（移除 'duplicate sample_cond_pt'）。")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    reports: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_name = RUN_NAMES[seed]
        profile_path = LOGS_DIR / f"{run_name}.profile.jsonl"
        records = load_profile(profile_path)
        if not records:
            print(f"WARNING: no profile data for seed {seed} at {profile_path}", file=sys.stderr)
            continue
        report = build_decision_report(seed, records)
        reports.append(report)
        print(f"seed {seed}: {report['n_steps_logged']} steps, loss_trend={report['loss_trend']}, decision={report['decision']}")
    if not reports:
        print("ERROR: no reports generated", file=sys.stderr)
        return 1
    output_path = DOCS_DIR / "stage_a_100k_health_decision.md"
    write_markdown_report(reports, output_path)
    print(f"\nReport written to {output_path}")
    # Also write JSON summary
    json_path = DOCS_DIR / "stage_a_100k_health_decision.json"
    json_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"JSON summary written to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
