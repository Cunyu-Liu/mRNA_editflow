# A3 — critic 混合池判别力探针 (H2 直接测量)

日期 2026-09-07 · Protection: protected reads = 0（只读 DEVELOPMENT VALIDATION 结构信息 + measured 序列身份 + measured_direction_normalized_delta 定真值最好；未触碰任何 TEST/EVALUATION outcome 文件）。

## 1. 状态（务必先读）

- **GPU 全忙，无法在本会话执行 V5-critic BF16 推理**：6 张满血 A100 (GPU0-5) util 80–100%、显存 21–39GB 占用；GPU6/7 为 MIG 分区（4.75GB / 20GB 亦被占用或过小）。按纪律不抢占在途训练（load≈94）。
- **CPU 回退不可行**：`FrozenXEditCriticV5` 硬性 `_require(device.type=="cuda")` + `autocast(cuda,bf16)`，项目惯例 "CUDA BF16-only, cpu_fallback=false"。改代码违反"复用既有入口、不另造轮子"。
- **代码已就绪 + dry-stub 全 891 源管线冒烟通过**（plumbing/math 验证，非科学结论）。
- **全量 BF16 运行需由主会话在空闲某张满血 A100（或低峰窗）调度**，命令见 §6。

## 2. 探针设计 (A3.1)

对每个验证源，构造**混合池** = unguided base 生成 32 候选（去重保留 max generation_score）∪ 该源全部真实 measured 候选（未命中的从 VALIDATION measured_neighborhood 补全，即"added"序列）；去重后池大小 ~29.9±，区间 [4,36]。池内所有候选用 `FrozenXEditCriticV5.potentials(states, endpoint_id, region, source_row)` 打分（CUDA BF16，复用 TreeG 加载路径——critic 只读 state 的 source_sequence/current_sequence/assay_id/context_id，故 measured 序列可直接构 state，无需 trajectory 回放）。

**指标**（k=1 tie-aware，与 A1 hit@1 同语义）：
- `conditional_rank_acc@1` = 池内含 ≥1 个 measured 时（本设计全源皆含，因 measured 被补全），critic 把"真值最高 measured"准入 top-1 并列块：Σ_{seq∈top块, seq∈true_best} (1/|块|)。
- `measured_best_avg_rank` = 池内所有 measured 中排名最高者的块起始名次（1-based），逐源求均值。
- **base-self 基准线**：对同一混合池（池完全一致）用 B2 unguided `generation_score` 排序；measured added 序列 base 从未产出、赋 `min(generation_score)-1`（垫底）→ base-self 只能触达"它自己恰好生成的 measured"。
- 同时报告 natural-hit(32) 子集（A2 的 support 源）上的 critic vs base 对照，做同池 apples-to-apples。
- per-task 分解 MRL652/MPRAU108/HL111/polyA20；polyA 源池天然无 measured → 如实标 N/A（support=0）。

## 3. 结构级结论（model-independent，已实测，不依赖批评网络）

**“真值最好 measured 极大部分是 base 生成器从未发出的序列”——这是 H2 的关键前置事实：**

| task | n | trueBest ADDED | trueBest NATURAL | base_reachable% | natHit32% |
|---|---|---|---|---|---|
| MRL | 652 | 571 | 81 | **12.4** | 31.3 |
| MPRAU | 108 | 105 | 3 | **2.8** | 5.6 |
| HL | 111 | 105 | 6 | **5.4** | 5.4 |
| polyA | 20 | 20 | 0 | **0.0** | 0.0 |

- `base_reachable%` = 真值最高 measured 出现在 base 32 池内的源占比 → 这是 **base-self conditional_rank_acc@1 的理论上限**（即便 base 自身排序完美也到不了更高）。
- 即：MRL 只有 **12.4%** 的源，base 自身排序有"把真值最好 measured 排到 top-1"的可能；MPRAU 2.8%、HL 5.4%、polyA 0%。
- 混合池大小 min/mean/max = 4 / 29.87 / 36（4 出现在池内 32 生成去重后骤减 + measured 仅 2 的极少数源）。

**推论（对 H2）**：critic vs base-self 的对比**结构性注定向 critic 倾斜**——critic 能对 added measured 打分因此理论上触达 100% 源，而 base-self 被钉死在上表 reachable% 的天花板。**真正待测的 H2 是：critic 在"能触达"的 added 序列上，判别力是否真的把它排到 top-1——这必须靠 §6 的 BF16 运行给数值。**

## 4. dry-stub 冒烟（plumbing/math 验证，非科学）

`a3_critic_mixed_pool_probe.py --dry-stub` 跑通全部 891 源：数据加载、混合池构造、真值最好解析、tie-aware 指标、per-task 分桶、JSON 输出均正常；source_scoring_calls=891；polyA natural-hit=0、MRL natural-hit 0.3129 与 A2 逐位吻合。（stub 用确定性 hash 拟评分，数值无科学意义，仅证明管线正确。）

## 5. 产物

- `a3_critic_mixed_pool_probe.py`（脚本，/tmp/phaseA_lcy/ 与 本目录两处）
- `A3_dry_smoke_stub.json`（dry-stub 全 891 源输出，status=DRY_SMOKE_STUB）
- `A3_pool_structure.json`（§3 结构级数字）
- 报告：随主会话交付。

## 6. 全量 BF16 运行命令（交主会话调度，需空闲满血 A100）

```bash
cd /tmp/phaseA_lcy
# 空闲卡低峰时（例如某张 A100 util/显存回落）：
python3 a3_critic_mixed_pool_probe.py \
  --output /home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901/analysis_phaseA_20260907/A3_critic_mixed_pool_probe.json \
  --physical-gpu-index <空闲卡号> 
# 小规模冒烟：加 --cohort 60（non-terminal）；全量不加。
# 校验关键字段：status=TERMINAL, precision=BF16, overall.critic_conditional_rank_acc@1,
# overall.base_conditional_rank_acc@1, measured_best_avg_rank 对照。
```
预期解读锚点：A1 实测 base 隐式 hit@1 (unguided)=0.0367；base-self 天花板见 §3 table。on-manifold ρ（V5）：polyA 0.8219 / MRL 0.1354 / MPRAU 0.1025 / HL≈0——若混合池上 critic conditional_rank_acc 显著 > base_reachable%，则 critic 在"已触达"源上有真判别；若仍 ≈ 触达线上的 chance，则 H2 成立（critic 判别力不足）。
