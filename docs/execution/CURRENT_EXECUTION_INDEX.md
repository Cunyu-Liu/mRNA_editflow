# mRNA EditFlow 当前执行索引

**文档角色：** 唯一当前执行入口  
**适用项目：** Route A V3 / XEditFlow V4.0.3 / SetFlow V4-S1
**更新时间：** 2026-08-28（Asia/Shanghai）
**执行原则：** 本文件只规定当前入口、阶段依赖和证据边界；科学阈值仍以冻结合同与协议为准。

> 除本文件外，`docs/execution/` 中带有旧日期、旧分支或旧工作树的执行文档均为历史快照。不得从历史快照复制命令启动当前实验。

## 1. 权威顺序

发生冲突时按以下顺序处理：

1. 主科学合同（外部 canonical locator，不在本文件复制内容）：`/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 数据gate转向后的合同.md`；
2. repo 内 V4 总协议：`docs/paper/route2_xedit_v4_prospective_experiments_protocol_v1.md`；
3. frozen guidance protocol config：`configs/route_a_v3_route2_xeditflow_v4_guidance_protocol_v1.json`；
4. 本文件；
5. `route_a_v3_route2_rapid_iteration_log_20260827.md` 与对应 HEAD 的 runner verification receipts；
6. 其他日期化记录和历史交接文档。

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

原两条活跃游标均已结案，不再轮询：

1. SetFlow V4.0.3 recovered Validation 已精确 8/8 SUMMARY terminal；冻结 gate 为
   `XEDITSETFLOW_V4_SCREEN_NO_GO`，`confirmation_authorized=false`，protected、Development
   TEST 与 new Evaluation reads 均为 0。该科学 NO-GO 和全部旧路径永久只读；禁止重启 recovery、启动旧
   recovered confirmation/posttraining、降低阈值或追加 seed。
2. Critic V4.0.3 `v4_full` 已发布唯一 terminal SUMMARY：seed 20260907、pass 8、22,416 updates、
   physical/effective batch 32、A100 GPU5、CUDA/BF16、无 CPU fallback、f34 authorization，protected、
   Development TEST 与 new Evaluation reads 均为 0。六个 controls 没有启动，因为它们无法恢复已经失败的
   SetFlow dual-readiness；该单臂 summary 不是 screen PASS 或最终科学结论。

对应 repo 事实审计为：

- `audits/route_a_v3_route2_xeditsetflow_v403_recovered_screen_terminal_nogo_v1.json`
- `audits/route_a_v3_route2_xeditcritic_v403_full_terminal_v1.json`

当前新入口是独立的 SetFlow V4-S1 mechanics family。它是主合同下的前瞻性从属修订，不是旧 V4.0.3
重试，也不覆盖旧 NO-GO。配置为
`configs/route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_v1.json`；任何真实启动仍必须等待包含实现的
clean、已推送 exact HEAD 完成全量准入与双 receipt。

监控只由 Codex 中名为“mRNA EditFlow 训练监控”的两小时 heartbeat 执行。人工执行阶段不得循环轮询、分钟级读取、tail 日志或重复解析同一 runtime。等待期间只做不会读取 protected outcomes、不会改变运行中 artifacts 的文档、静态审计、CPU-native 单元测试和数据契约核查。

## 4. 唯一阶段图

```text
旧 SetFlow V4.0.3 recovered screen ──> XEDITSETFLOW_V4_SCREEN_NO_GO（冻结、只读）
旧 Critic V4.0.3 full ──> terminal SUMMARY（controls 暂停、非 screen PASS）

SetFlow V4-S1 prospective freeze
    └─> v4_s1_full + v4_s1_single_mode（seed 20260911）
        └─> checkpoints 4/6/8/10 的冻结 891×32 outcome-free Validation
            ├─> XEDITSETFLOW_V4_S1_SCREEN_NO_GO：该新方法 family 终止
            └─> XEDITSETFLOW_V4_S1_SCREEN_PASS
                └─> 仅允许新 S1-bound 三 seed confirmation；旧 V4.0.3 launcher 禁用
                    └─> 后续仍须逐级重获 SetFlow G0_READY、Critic readiness、
                        guidance frozen 与 98-job Final terminal
                            └─> Final gate PASS：优秀 Development 结果
```

每一箭头都要求前一阶段产物完整、唯一、精确终态、protected reads 合法且冻结 gate 达标。不得以 smoke、proxy、训练集指标、单 seed、单臂、screen、原子 TEST、G0 或 guidance screen 代替后续 gate。

## 5. 后继启动入口

所有后继脚本只从第 2 节工作树的 clean、已推送 HEAD 启动；所有 `--expected-head` 类参数均使用启动时的当前 HEAD。脚本名如下，具体参数由两小时 heartbeat 按 frozen protocol 和当前 receipt 物化，不从历史文档复制：

### SetFlow

- `launch_route2_xeditsetflow_s1_screen_after_v403_terminal.py`
- `run_route2_xeditsetflow_s1_screen_scheduler.py`

S1 只新增 fixed-weight 0.05 的跨状态候选 mode-responsibility forward KL；根 `EMPTY` slot 0 posterior
停梯度，约束同一 occurrence/canonical candidate 的 compatible nonroot、nonstructural states，归约严格为
state→candidate→occurrence。原 V4 架构、sampler、seed 20260911、十 passes、batch 32、checkpoints
4/6/8/10、891×32 Validation 与全部绝对/相对 gate 不变；不存在 weight sweep 或额外 screen seed。

- `launch_route2_xeditsetflow_v403_recovered_confirmation.py`
- `launch_route2_xeditsetflow_v403_recovered_confirmation_posttraining.py`

上面两个 V4.0.3 recovered launcher 现在仅保留为历史实现，已被冻结 NO-GO 禁用。未来只有 S1 screen
精确 PASS 才可实现并启动新 S1-bound confirmation；其训练 schedule 与 posttraining schedule 仍必须分离。
Guidance 只能消费真正达到 `XEDITSETFLOW_V4_G0_READY` 的 posttraining schedule，而不是 screen PASS。

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

### Final terminal 后的交接收尾

- `reproduce_route2_base_flow_v2_handover_validation.py`
- `export_route2_xeditflow_v4_terminal_training_ledger.py`

两者只在 Final runtime 与 adjudication 均精确为 `XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL` 后执行；科学 gate 为 PASS 或 NO_GO 都应保存交接证据，技术失败或在途状态均不得触发。Base Flow V2 R3 只从 terminal candidate、Development source manifest 与 measured-neighborhood rows 独立重算冻结指标，不重训、不重跑七方法、不读取 TEST/new Evaluation，也不做模型 forward；如果未来另行要求重新生成候选，则该 forward 必须真实使用 CUDA，CUDA 不可用或 CPU fallback 时立即停止留证。Terminal ledger 只导出 Final 实际依赖的 72 个参数更新尝试，不递归扫描历史、不读取日志/checkpoint/private payload、不改中央 ledger，也不改变任何训练、阈值、gate 或科学结论。

上述两个直接测试模块属于 successor focused receipt 第 8 组的强制覆盖。工具可以在 Final 前完成代码和合成测试，但真实 `/mnt` 交接产物只能在 terminal gate 后一次性物化。

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
2. 以八个隔离 Python 进程运行 successor focused tests；
3. 运行恰好 96 项 V3.3.2 回归，命令必须保留 `*v332*.py` glob；
4. 在 `/mnt/cunyuliu` 生成与新 HEAD 对应的 shared runner receipt 和 SetFlow receipt，并写入八组各自的实际通过数及其总和；
5. 用两类实际消费者分别接受 receipt；
6. 把两小时 heartbeat 的 exact HEAD 与 receipt 路径原子更新到新提交。

`c5db9a3617f1798742566ffe23b8e9faa750e7a5` 已验证覆盖给出的 focused 准入下限是 203 项；203 是覆盖下限，不是永久锁死的精确总数。后续新增测试时，八组实际通过数和更高的实际总数必须写入对应 exact-HEAD receipt，且分组之和必须等于总数。

第 8 组必须包含 Base Flow V2 handover reproduction 与 terminal training ledger exporter 的直接测试；不得因它们只在 Final terminal 后运行而从工程准入回执中省略。

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
