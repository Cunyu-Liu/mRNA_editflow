# mRNA EditFlow 当前执行索引

**文档角色：** 唯一当前执行入口  
**适用项目：** Route A V3 / XEditFlow V4.0.3  
**更新时间：** 2026-08-27（Asia/Shanghai）  
**执行原则：** 本文件只规定当前入口、阶段依赖和证据边界；科学阈值仍以冻结合同与协议为准。

> 除本文件外，`docs/execution/` 中带有旧日期、旧分支或旧工作树的执行文档均为历史快照。不得从历史快照复制命令启动当前实验。

## 1. 权威顺序

发生冲突时按以下顺序处理：

1. 冻结的主科学合同、数据角色和 protected-read 约束；
2. XEditFlow V4.0.3 冻结协议与配置；
3. 本文件；
4. `route_a_v3_route2_rapid_iteration_log_20260827.md` 与对应 HEAD 的 runner verification receipts；
5. 其他日期化记录和历史交接文档。

历史文件 `route_a_v3_route2_next_goal_todo_handoff_20260820.md` 已明确降级为只读历史快照，不是执行入口。

## 2. 当前代码与产物位置

| 项目 | 当前唯一位置 |
|---|---|
| GitHub 分支 | `route-a-v3-v403-no-vram-gate-20260827` |
| A100 工作树 | `/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v403_no_vram_gate_20260827` |
| 大型数据、权重与 artifacts 根目录 | `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2` |
| Python | `/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10` |
| 训练记录 | `docs/execution/route_a_v3_route2_rapid_iteration_log_20260827.md` |
| 尝试总表 | `docs/execution/route_a_v3_route2_training_attempt_table_20260817.md` |

当前执行 HEAD 定义为：**本文件所在提交的 clean HEAD，且该 HEAD 必须等于远端分支 HEAD**。不要在文档中长期手写一个会随下一次修复失效的提交哈希。实际启动时必须由调度器的 `--expected-head`、schedule、runtime 和对应 HEAD receipt 共同固定该值。

代码和提交留在 `/home` 工作树；大型数据、权重、日志和实验 artifacts 只写入 `/mnt/cunyuliu`。

## 3. 当前运行边界

本索引提交时已经存在以下两条在途实验；在它们各自形成合法终态前，其 canonical one-shot launcher 均不得再次调用：

1. SetFlow V4.0.3 validation recovery：
   `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v4/v403_validation_recovery_37c5901000cf6bef1606f05af242512f1342ceb6/runtime.json`
2. Critic V4.0.3 `v4_full`：
   `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v4/v403_rng_replay_fix_runner_f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea/runtime.json`

它们的实时状态不写入本文件，以 runtime、schedule 和终态 evidence 为准。合法终态后，只能由两小时 heartbeat 按第 4 节阶段图启动满足前置 gate 的后继；因此后继出现时，不应把上面的“本索引提交时”快照误读为永久限制。

监控只由 Codex 中名为“mRNA EditFlow 训练监控”的两小时 heartbeat 执行。人工执行阶段不得循环轮询、分钟级读取、tail 日志或重复解析同一 runtime。等待期间只做不会读取 protected outcomes、不会改变运行中 artifacts 的文档、静态审计、CPU-native 单元测试和数据契约核查。

## 4. 唯一阶段图

```text
SetFlow recovery ──> SetFlow screen PASS ──> 3-seed confirmation
                                            └─> 12 checkpoint Validation
                                                └─> XEDITSETFLOW_V4_G0_READY

Critic current full ──> 6 controls ──> cross-root screen PASS
                                      └─> 3-seed full + matched C0
                                          └─> confirmation Validation PASS
                                              └─> atomic frozen TEST
                                                  └─> 3 refits
                                                      └─> 42-job LOSO readiness

SetFlow G0_READY + Critic post-test readiness
                    └─> 18-cell guidance screen
                        └─> XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN
                            └─> 3-seed, 98-job final Development comparison
                                └─> XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL
                                    + XEDITFLOW_V4_PASS
```

每一箭头都要求前一阶段产物完整、唯一、精确终态、protected reads 合法且冻结 gate 达标。不得以 smoke、proxy、训练集指标、单 seed、单臂、screen、原子 TEST、G0 或 guidance screen 代替后续 gate。

## 5. 后继启动入口

所有后继脚本只从第 2 节工作树的 clean、已推送 HEAD 启动；所有 `--expected-head` 类参数均使用启动时的当前 HEAD。脚本名如下，具体参数由两小时 heartbeat 按 frozen protocol 和当前 receipt 物化，不从历史文档复制：

### SetFlow

- `launch_route2_xeditsetflow_v403_recovered_confirmation.py`
- `launch_route2_xeditsetflow_v403_recovered_confirmation_posttraining.py`

SetFlow confirmation 训练 schedule 与 posttraining schedule 是两个不同文件。Guidance 只能消费 posttraining schedule。SetFlow dual-readiness 的精确终态是 `XEDITSETFLOW_V4_G0_READY`，不是 `G0_PASS`。

### Critic

- `launch_route2_xeditcritic_v403_controls_after_full.py`
- `transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py`
- `launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py`
- `launch_route2_xedit_v4_confirmation_posttraining_after_terminal.py`
- `launch_route2_xeditcritic_v4_atomic_frozen_test_after_confirmation.py`
- `launch_route2_xeditcritic_v4_refit_after_atomic_test.py`
- `launch_route2_xeditcritic_v4_loso_after_refits.py`

冻结 TEST 只允许在 confirmation 科学 gate PASS 后原子读取一次。LOSO 不得在 refit 或 readiness 前置条件未满足时提前启动。

### Guidance 与 Final

- `launch_route2_xeditflow_v403_guidance_after_dual_readiness.py`
- `launch_route2_xeditflow_v4_final_after_guidance_screen.py`

Guidance 必须同时消费 SetFlow posttraining readiness 与 Critic post-test readiness。Final 只能消费 bridge launch receipt 中的 `final_successor` 字段，不能裸调用或回落到 legacy 默认路径。Guidance 的精确终态是 `XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN`；Final 每个 seed 的 evidence 成功标记是 `final_evidence/seed_manifest_row.json`。

## 6. GPU 与失败处理

- 训练和 GPU 验证必须真实使用 CUDA；神经网络任务按冻结配置使用 CUDA/BF16。
- CUDA 不可用、设备证据不一致、CPU 静默降级、OOM、启动失败、双终态或无精确终态，均是技术失败：停止受影响 package 的后续排队，保存命令、日志、设备和首失败证据。
- 任一长队列出现首技术失败后，只允许已在飞任务自然收尾；不得再启动 pending job。Pending 必须记录 `NOT_RUN_AFTER_TERMINAL_FAILURE`，并跳过技术上不完整 package 的 adjudication。
- 科学 `NO_GO` 是成功完成的裁决结果，不得伪装成技术失败，也不得改写成 PASS。
- 禁止按剩余显存、预计显存或预留显存设置启动 gate。GPU inventory 和显存只作诊断；任何有可用显存的配置 GPU 都可按调度器安排使用。
- Final 启动前 GPU inventory 若执行、解析或设备枚举失败，必须在 runtime root 创建前写 sibling `*.failed.json`；同一 family 不覆盖、不自动重试。真正需要重试时建立新 retry/run family。

## 7. 包级一致性与提交纪律

Confirmation training、confirmation posttraining、guidance screen 和 final comparison 均采用包级首失败语义。每个 job 在真正启动前，都必须在同一启动锁内确认工作树仍等于 schedule 固定的 clean HEAD；终态分类和首失败记录也必须在线性化锁内完成。

任何代码或当前执行文档修改后，下一条新实验启动前必须完成：

1. 提交并推送到当前 GitHub 分支；
2. 以六个隔离 Python 进程运行 successor focused tests；
3. 运行 96 项 V3.3.2 回归；
4. 在 `/mnt/cunyuliu` 生成与新 HEAD 对应的 shared runner receipt 和 SetFlow receipt；
5. 用两类实际消费者分别接受 receipt；
6. 把两小时 heartbeat 的 exact HEAD 与 receipt 路径原子更新到新提交。

Receipt 路径模板：

```text
/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/xedit_v4/v403_successor_runner_verification_<CURRENT_HEAD>.json
/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/xeditsetflow_v4/confirmation_v403_recovered_runner_verification_<CURRENT_HEAD>.json
```

不要为不改变决策的重复检查、checksum 或额外自审制造工作。只核验会阻止错误启动、保护冻结数据或改变下一步动作的条件。

## 8. “优秀结果”的唯一口径

只有同时满足以下两个冻结条件，才可称为“优秀的 Development 模型结果”：

```text
final_adjudication.json.status == XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL
final_adjudication.json.gate.status == XEDITFLOW_V4_PASS
```

在此之前只能报告工程阶段、技术终态和 frozen gate 的实际状态，不能提前写最终科学结论。即使 Development PASS，也不等于外部 Evaluation 成功；在新的 outcome-unexposed Evaluation 合法可用前，必须保持 `submission_ready=false`，不得宣称外部最终科学结论。

## 9. 每轮训练记录的最低内容

每个 run/retry family 必须在 rapid iteration log 与 attempt table 中记录：配置/协议、Git commit、GPU 与 CUDA 证据、seed、开始和结束时间、终态、关键冻结 gate、失败原因、是否允许后继以及本轮结论。技术重试新建独立 family，不覆盖旧 artifacts、不降低冻结阈值。

本索引不记录正在运行实验的中间性能数值；中间数值只由低频 heartbeat 在必要时读取并写入对应实验证据。
