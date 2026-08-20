# Route A V3.3 Route 2 下一阶段 Goal 与 To-Do 交接文档

**文档状态：** 当前可执行交接  
**快照时间：** 2026-08-20 17:30（Asia/Shanghai）  
**适用分支：** `route-a-v3-route2-method-repair-20260817`  
**本地工作树：** `/Users/liucunyu/Documents/Codex/2026-08-10/ssh-p-22-cunyuliu-36-137-2/work/route2_pretrained_scaleup_20260817`  
**A100 工作树：** `/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817`  
**运行产物根目录：** `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2`

本文件是执行交接，不是新的科学结果，也不修改当前资格、canonical 数量或论文 claim。普通工程任务按本文件直接推进；不恢复 successor、runtime ledger、one-read resource 或无实际用途的 checksum 约束。

---

## 1. 唯一总目标

在不读取冻结 Development TEST、GSE232572 或 E-MTAB-10902 outcomes 的前提下，完成 Development 上的独立 evaluator 裁决与同预算生成/搜索 baseline；同时前瞻性修复 Delta critic 的跨 seed 稳定性。只有新 critic 完整通过 controls 和三个 final seeds，才允许依次打开一次 Development TEST、在全部 126,165 条 Development records 上 refit、启动 frozen-critic guided XEditFlow，最后进入两个独立 Evaluation studies。

```text
独立 evaluator 正式裁决
          ↓
七种 matched-budget generation/search baselines
          │
          ├────────────────┐
          ↓                ↓
Development 生成比较     Critic V2 前瞻性修复
                           ↓
                  controls + three seeds
                           ↓ 仅当通过
                  一次冻结 Development TEST
                           ↓
                  all-126,165 final refit
                           ↓
                  frozen-critic guided XEditFlow
                           ↓
                  独立 Evaluation（最后）
```

---

## 2. 当前不可改写的事实

| 项目 | 当前事实 | 执行含义 |
|---|---|---|
| 正式数据状态 | `ordinary=1 / A1=1 / true-A2=0 / canonical=6,547` | Development-relaxed 数据不得写成 qualified |
| Development 总记录 | 126,165 | TEST 18,292 条仍冻结；不得提前读取 |
| mRNABERT Huber | 100 epochs 完成；best epoch 44；task-macro Spearman 0.149988 | 单 seed 结果，不等于 readiness |
| 三种 loss | Huber 胜出；fixed/learned variance 均更差 | uncertainty 只保留诊断，不进入 guidance reward |
| uncertainty 诊断 | learned uncertainty 与绝对残差相关，但 mean 性能下降 | 已观察到 uncertainty absorption；不再用 NLL 掩盖 mean 退化 |
| signal controls | candidate permutation 与 parameter-matched source-only 已运行 | source-only 同时承担只看 source anchor 的对照 |
| three final seeds | 仅 1/3 seed 优于 strongest baseline | critic readiness 失败；TEST/refit/guided 保持关闭 |
| Base Flow V2 | `FLOW_G0_READY` | 生成工程骨架可用；不代表 biological optimization 成功 |
| 外部 Evaluation | GSE232572 / E-MTAB-10902 outcomes read=0 | 必须保持到最后 |

### 2.1 Base Flow 已完成事实

- 891 个 source cohorts；
- 28,512 条 trajectories；
- hard legality 100%；
- edit/candidate budget violation 0；
- replay failure 0；
- numerical failure 0；
- small-graph DP 与独立枚举 TV distance 0；
- source-macro unique candidate rate 0.882891；
- critic 未参与 Base Flow 训练或 validation。

### 2.2 独立 evaluator 最新事实

独立 evaluator 的训练已经完成，但正式 adjudication 文件尚未生成。

| 字段 | 当前值 |
|---|---:|
| 状态 | `DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE` |
| 模型 | 独立 Siamese CNN，不使用 mRNABERT critic features |
| 可训练参数 | 509,845 |
| epochs / optimizer updates | 8 / 22,400 |
| TRAIN / VALIDATION | 89,580 / 18,293 |
| task-macro Spearman | 0.1025655357 |
| 预冻结 exclusive threshold | 0.1012475746 |
| margin | +0.0013179611 |
| positive tasks | 5/9，要求至少 5 |
| Development TEST | 未读取，18,292 条继续 withheld |
| Evaluation outcomes | 0 |
| GPU | A100，参数更新已验证 |

使用现有 adjudicator 在内存中只读计算，全部 checks 为 true，预期状态是 `INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED`。这只是 Development 方法选择工具的最低资格。margin 很小，不能写成独立 evaluator 已证明广泛预测能力，更不能写成论文科学结论成立。正式状态仍以 adjudication 文件实际落盘后的内容为准。

证据路径：

```text
/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/independent_generation_evaluator/neural_medium_siamese_task_scaled_seed20260816_frozen_development_validation_gpu2_v3/training_summary.json
/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/independent_generation_evaluator/neural_medium_siamese_task_scaled_seed20260816_frozen_development_validation_gpu2_v3/delta_predictor_checkpoint.pt
```

---

## 3. Goal A — 立即闭合独立 evaluator 裁决

### 目标

不重训 evaluator，只使用已经完成的 training summary 运行一次现有裁决器，生成正式 adjudication；随后让已经在等待的 matched-generation v2 调度器自然接棒。

### To-Do A

- [ ] A1. 确认 evaluator 训练进程已经结束，summary 和 checkpoint 存在。
- [ ] A2. 确认正式 adjudication 路径尚不存在，避免覆盖：

  ```text
  /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/comparisons/mrnabert_independent_evaluator_adjudication_v1.json
  ```

- [ ] A3. 在 A100 工作树运行一次裁决，不重新训练：

  ```bash
  cd /home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817
  /home/cunyuliu/miniconda3/envs/editflow/bin/python scripts/route_a_v3/adjudicate_route2_independent_generation_evaluator_v1.py --training-summary /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/independent_generation_evaluator/neural_medium_siamese_task_scaled_seed20260816_frozen_development_validation_gpu2_v3/training_summary.json --protocol configs/route_a_v3_route2_mrnabert_independent_evaluator_qualification_v1.json --output /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/comparisons/mrnabert_independent_evaluator_adjudication_v1.json
  ```

- [ ] A4. 读取一次 adjudication，确认：
  - `status=INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED`；
  - task-macro、threshold、margin 和 positive-task count 与本文件一致；
  - `development_test_outcomes_accessed=false`；
  - `evaluation_outcomes_accessed=false`；
  - 所有 checks 为 true。
- [ ] A5. 如果正式结果不是预期状态，不修改 threshold、不重跑 HPO、不挑 epoch；记录真实 NO-GO，并按 Goal B 的 NO-GO 分支继续候选生成但停止方法评分。
- [ ] A6. 确认 matched-generation v2 调度器仍在等待。adjudication 落盘后等待它在下一次轮询自然启动；不要同时手工再启动第二份 suite。
- [ ] A7. 只有调度器已经退出时，才允许重新启动一次：

  ```bash
  cd /home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817
  nohup scripts/route_a_v3/schedule_route2_matched_generation_suite_v2.sh >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/matched_generation_suite_v2.log 2>&1 </dev/null &
  ```

### Goal A 完成标准

- 正式 adjudication 文件存在且可解析；
- 没有重新训练 evaluator；
- TEST 和两个 Evaluation outcomes 的访问计数仍为 0；
- matched-generation v2 已启动或明确记录为何没有启动；
- 中央训练尝试表记录 evaluator terminal，而不是继续显示 RUNNING。

---

## 4. Goal B — 完成同预算生成/搜索 baseline

### 目标

在相同 source pool、编辑预算、候选数量和计算预算下运行七种基线，使用独立 evaluator 而不是 guiding critic 的 self-score 选择 Development strongest method。

### 必跑方法

1. random legal edits；
2. greedy；
3. beam search；
4. genetic algorithm；
5. local search；
6. generate-N-then-rerank；
7. unguided learned Base Flow。

### To-Do B

- [ ] B1. 确认 Flow summary 为 `FLOW_G0_READY`，evaluator adjudication 为 terminal，且 Evaluation 访问计数为 0。
- [ ] B2. 由 matched-generation v2 调度器在 GPU 0–5 中选择剩余显存足够的一张卡；不使用 utilization 阈值。
- [ ] B3. 所有方法使用相同 Development source pool、`SUB + STOP` action space、edit budget、candidate budget、generator NFE、critic/evaluator forward-equivalent、seeds 和 HPO 预算。
- [ ] B4. 每种方法记录 legality、budget violation、unique candidate rate、diversity、STOP、NFE、evaluator forwards、wall time 和 peak VRAM。
- [ ] B5. evaluator qualified 时，完成独立评分、source-paired comparison 和 strongest baseline 冻结。
- [ ] B6. evaluator NO-GO 时，仍完成七种候选和 compute ledger，但停止于 `MATCHED_GENERATION_CANDIDATES_COMPLETED_EVALUATOR_NO_GO`，不得使用 guiding critic self-score 选 strongest method。
- [ ] B7. generated candidates 不增加 canonical、ordinary、A1 或 true-A2 credit。
- [ ] B8. 完成后更新中央尝试表和项目进度文档，并提交、推送 GitHub。

### 预期主要产物

```text
/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/generation_search_baselines/matched_compute_position_progress_seed20260816_v2/
/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/generation_search_baselines/matched_compute_position_progress_seed20260816_v2/matched_generation_suite_summary_v2.json
```

### Goal B 完成标准

- 七种方法均出现明确 terminal，不丢失失败方法；
- 预算匹配、候选数量和 GPU 计算账本完整；
- strongest method 只在 independent evaluator qualified 时产生；
- 结果只写成 Development generation evidence，不写成 biological improvement；
- GSE232572 / E-MTAB-10902 outcomes read=0。

---

## 5. Goal C — 前瞻性修复 Critic V2 的跨 seed 稳定性

### 目标

解决当前 critic 单 seed 好看、三个 final seeds 不稳定的问题。目标不是再找一个最好 seed，而是让 candidate-specific signal 在同一 Validation 和三个预冻结 seed 上稳定优于 strongest same-information baseline。

三个 final seeds 相对 strongest baseline 的 margin 为：

```text
-0.015586
-0.014806
+0.005669
```

只有 1/3 为正。增加当前配置的 epoch、事后挑 seed 或直接增加参数量都不能修改这一事实。

### To-Do C

- [ ] C1. 只用已有 Development TRAIN/VALIDATION summaries 做 per-task、per-study 和 source-group 失败分析；不读 TEST。
- [ ] C2. 冻结一个 Critic V2 方法假设，优先解决训练目标与 task-macro selection 不一致：
  - task/study-balanced sampling；
  - 与 task-macro 目标一致的 loss weighting；
  - frozen mRNABERT encoder；
  - edit-centered pooling、source/candidate 差分、全局 source 背景和反对称约束；
  - Huber 主 loss；learned variance 只作诊断。
- [ ] C3. 冻结四种信息边界：
  1. full source+candidate+edit；
  2. candidate permutation；
  3. parameter-matched source-only；
  4. 如果单独保留 anchor-only 名称，则定义为 `source + edit metadata、无 candidate sequence`，不能与 source-only 同义重复计数。
- [ ] C4. 先运行 strongest same-information baseline 与 controls；controls 未证明 candidate-specific signal 时，不启动三个 final seeds。
- [ ] C5. controls 通过后运行三个预冻结 final seeds；不根据结果增加第四个 seed。
- [ ] C6. 三个 seed 统一报告 task-macro Spearman、standardized MAE、prediction spread、baseline margin、positive task count 和三种 control 差距。
- [ ] C7. 只有同时满足以下条件才标记 `CRITIC_READY_FOR_TEST`：
  - 三个 seed 的 margin 全部 `>0`；
  - candidate permutation 明显低于 full model；
  - source-only 和 source+edit-metadata 不能解释 full model 的主要收益；
  - 无 NaN、mean collapse 或异常 prediction spread；
  - TEST/Evaluation 访问仍为 0。
- [ ] C8. 如果再次失败，保留负结果，继续 benchmark/limits 论文路线；不事后降低 three-seed 条件。

### Goal C 完成标准

- 新假设、数据、split、loss、controls、seeds 和预算在运行前冻结；
- 至少形成一次可裁决的 three-seed terminal；
- PASS 或 NO-GO 都进入中央训练尝试表；
- 不读取 Development TEST 或外部 Evaluation。

---

## 6. Goal D — 条件式打开 TEST、全量 refit 与 guided XEditFlow

本阶段当前为 **LOCKED**。只有 Goal C 达到 `CRITIC_READY_FOR_TEST` 才执行。

### To-Do D

- [ ] D1. 只打开一次冻结 Development TEST；记录完整 TEST metrics，不再据此选择架构、loss、seed、epoch 或 threshold。
- [ ] D2. 固定结构、loss、训练预算和 seed policy，在全部 126,165 条 Development records 上做一次 final refit。
- [ ] D3. 冻结 final critic checkpoint、输入 schema、reward normalization 和 calibration policy。
- [ ] D4. generator 不得反向更新 critic；independent evaluator 不接收 generator 或 critic 的训练梯度。
- [ ] D5. 启动 frozen-critic guided XEditFlow，与 Goal B 的 strongest matched baseline 比较。
- [ ] D6. 检查 reward hacking、candidate collapse、OOD、diversity、legality、STOP 和预算使用。
- [ ] D7. 没有 measured outcome 的新候选只称为 `computationally prioritized candidates`。

### Goal D 完成标准

- TEST 只访问一次；
- all-126,165 refit 不再产生新的模型选择结论；
- guided、unguided 和 strongest baseline 预算匹配；
- independent evaluator 与 guiding critic 分离；
- generated candidates 不增加资格或 canonical credit。

---

## 7. Goal E — 最后执行独立 Evaluation

本阶段当前为 **LOCKED**，永远最后执行。

### To-Do E

- [ ] E1. 在外部 outcome 读取前冻结 predictor、generator、strongest baselines、metrics 和 adaptation policy。
- [ ] E2. GSE232572：A1 zero-shot Delta prediction。
- [ ] E3. E-MTAB-10902：true-A2 exploratory ranking/search。
- [ ] E4. 先永久记录 zero-shot，再允许 calibration/few-shot。
- [ ] E5. adaptation 不得覆盖 zero-shot headline。
- [ ] E6. 按证据选择论文结局：XEditFlow+Delta+Benchmark、Delta+Benchmark，或 Benchmark+Transfer/Generation Limits。

---

## 8. 并行执行与 GPU 规则

### 可以并行

- Goal B 的生成/搜索 baseline；
- Goal C 的 Critic V2 设计、实现和 Development-only runs；
- 训练记录、benchmark 文档和论文方法整理；
- 不读取 outcome 的数据与代码检查。

### 不可以提前

- Development TEST；
- all-126,165 final refit；
- frozen-critic guided XEditFlow；
- GSE232572 / E-MTAB-10902 outcomes；
- 根据外部 Evaluation 调整模型或 threshold。

### GPU 使用

- GPU 0–5 中只要有足够显存即可启动任务；不以 GPU utilization 为启动门槛。
- 每个任务只保留避免明显 OOM 所需的最低余量；这不是科学 gate。
- 不终止正常 GPU 作业来抢卡。
- 同一实验只启动一份；不能因为输出暂未出现就复制运行。
- 训练期间不频繁查看 epoch 日志；只在启动验收、terminal、异常退出或调度器停滞时检查。
- 等待 GPU 或长训练时，并行推进代码、controls、baseline 配置、记录和论文材料。

---

## 9. 每个实验必须记录的内容

中央表：

```text
/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiment_tracking/route2_training_attempts.csv
```

每次训练或生成尝试至少记录：

- run ID、目的和所属 Goal；
- Git commit；
- 数据集及 TRAIN/VALIDATION/TEST/Evaluation 记录数；
- split 与 withheld 状态；
- 模型结构、冻结/可训练参数量；
- loss、optimizer、learning rate、batch size、workers、precision；
- epochs、optimizer updates、seeds；
- physical GPU、peak VRAM、wall time；
- best/selected checkpoint；
- task-macro metrics 和 baseline margin；
- control 类型；
- terminal status；
- TEST/Evaluation 是否访问；
- 下一步裁决。

失败、停止和负结果必须保留，不删除，不改写成未发生。

---

## 10. 代码、数据和提交规则

- 代码、配置、测试和文档进入 Git；每个任务完成后及时 commit 并 push GitHub。
- 大数据、feature cache、checkpoint、generated candidates 和运行产物保存到 `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/`。
- 大文件不进入普通 Git 历史。
- 不覆盖已有 terminal run；新方法使用新目录。
- 不增加 hashes、资源只能读一次、successor 或多层 runtime 机制。
- 对真实问题直接修复；对正常结果明确写 PASS，不制造额外 finding。

---

## 11. 最短交接清单

1. [ ] 不重训，运行独立 evaluator adjudicator，生成正式 adjudication；
2. [ ] 确认 matched-generation v2 调度器自然启动；
3. [ ] 完成七种同预算 generation/search baselines；
4. [ ] 同时冻结 Critic V2 的 task-balanced 方法和清晰 controls；
5. [ ] controls 通过后跑三个 final seeds；
6. [ ] three seeds 未全正：继续关闭 TEST/refit/guided，记录负结果；
7. [ ] three seeds 全正：一次 TEST → all-126,165 refit → frozen-critic guided XEditFlow；
8. [ ] Development 方法全部冻结后，最后才打开 GSE232572 与 E-MTAB-10902；
9. [ ] 每项 terminal 后更新中央表、项目进度、Git commit 和 GitHub push。

---

## 12. 当前项目一句话状态

**Prediction critic 尚未通过跨 seed readiness；Base Flow 工程已 `FLOW_G0_READY`；独立 evaluator 训练已完成且按冻结规则预期 qualified，但正式 adjudication 尚待落盘；matched-generation v2 正在等待该文件；Development TEST、all-126,165 refit、guided XEditFlow 和两个外部 Evaluation 均保持未启动。**
