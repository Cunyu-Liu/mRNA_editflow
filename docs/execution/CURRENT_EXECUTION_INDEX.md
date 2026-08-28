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
   Development TEST 与 new Evaluation reads 均为 0。该单臂 summary 不是 screen PASS 或最终科学结论。
   2026-08-28 的执行决策已把 Critic readiness 提升为 Guidance 的第一关键路径：历史暂停只说明当时没有
   启动 controls，不再阻止 Critic 独立完成自己的 controls、confirmation、atomic TEST、refit 与 LOSO。
   已存在的 full=f34 和 C0=937 只作为精确历史 v1 producer；六个尚未启动的 controls 必须使用新的许可
   clean exact HEAD、当前 trainer 和 v2 terminal evidence，不能恢复或冒充旧 f34 controls。

Critic 与基础 SetFlow mechanics 在数据依赖上独立，可以并行使用多张 GPU；Critic 未达到自己的冻结
readiness 时仍严禁进入 value target 或 Guidance。SetFlow 的旧科学 NO-GO 不得被改写，但也不再被用作
停止 Critic 自身验证链的理由。

对应 repo 事实审计为：

- `audits/route_a_v3_route2_xeditsetflow_v403_recovered_screen_terminal_nogo_v1.json`
- `audits/route_a_v3_route2_xeditcritic_v403_full_terminal_v1.json`

旧 SetFlow V4-S1 930 family 已按用户明确指令停止，并精确收敛为
`XEDITSETFLOW_V4_S1_SCREEN_TECHNICAL_FAILURE`。其 canonical invalidation receipt 为
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/xeditsetflow_v4/s1_screen_seed_20260911_runner_930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8_terminal_invalidation.json`；
`successor_authorized=false`、`same_family_retry_authorized=false`、protected reads=0。旧 930 runtime、
launcher 与 transition 永久结案，不再读取或重用。

许可主分支已 clean、push 到 exact HEAD
`ebf99ebf8a253ad27e311e555121d328df8fae10`。该 HEAD 的正式准入为八组 focused
158/17/149/27/22/10/110/92，共 585/585 PASS，V3.3.2 为 96/96 PASS，shared 与 SetFlow 双 receipt 已由
真实 consumers 接受。它从两个独立 one-shot launcher 启动了当前两条活跃执行：

1. corrected SetFlow S1 runtime：
   `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v4/s1_screen_seed_20260911_runner_ebf99ebf8a253ad27e311e555121d328df8fae10/runtime.json`。
   full/single 固定 GPU0/1 并发，后续八项 Validation 覆盖 GPU0–5；seed-before-model、matched
   initialization、objective、weight 0.05、十 passes、batch 32、checkpoints 4/6/8/10、891×32 cohort 与
   全部科学阈值保持冻结。
2. Critic controls runtime：
   `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v4/v403_control_recovery_runner_ebf99ebf8a253ad27e311e555121d328df8fae10/runtime.json`。
   六臂曾固定 GPU0–5 同时启动；GPU0/1/2 的 source-only、edit-metadata-only 与
   no-candidate-sequence 已产生不可覆盖 CUDA OOM failure，GPU3/4/5 三臂按包级语义自然收尾。该 package
   已失去科学完整性，最终只能是技术失败；严禁运行 cross-root adjudication 或 confirmation，也不得把 OOM
   写成科学 NO-GO。显存数值只作失败诊断，不得成为 gate、排序或筛卡依据。

Critic 技术重试代码只在独立 prep 分支
`route-a-v3-v403-controls-oom-retry-prep-20260828`、worktree
`/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v403_controls_oom_retry_prep_20260828` 开发；该分支不是 GPU
执行入口。新实现先由唯一 transition 将旧 package 的精确技术终态冻结成不可覆盖 receipt，再允许新 HEAD、
新根目录、retry index 1 的六臂完整重跑。调度固定为 GPU0–2 第一波、GPU3–5 第二波；第一波全部精确成功才
启动第二波，仍是一臂一卡、多 GPU 并行。子进程使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
减少 allocator 碎片；不查询 `memory.free`、不等待/排序/筛卡、不设显存阈值。模型、trainer、数据、seed、
batch、passes、更新预算和科学 gate 均不得改变。

监控只由 Codex 中名为“mRNA EditFlow 训练监控”的两小时 heartbeat 执行。每个活跃 runtime 每窗口最多读取
一次；不做分钟级轮询或 tail。旧 Critic package 精确终态后只运行一次新的 OOM terminal transition，receipt
或 `.partial` 已存在时禁止重读。等待期间只做不读取 protected outcomes、不改变活跃 artifacts 的文档、
静态审计、CPU-native 单元测试和数据契约核查。

## 4. 唯一阶段图

```text
旧 SetFlow V4.0.3 recovered screen ──> XEDITSETFLOW_V4_SCREEN_NO_GO（冻结、只读）
旧 Critic C0=937 + repaired full=f34 ──> 历史 v1 terminal producer（只读、非 screen PASS）

SetFlow V4-S1 旧 930 family ──> 技术终态 invalidated（冻结、只读）

┌─ Critic 第一关键路径（与基础 SetFlow 并行）
│  └─> Y3 六 controls：GPU0/1/2 OOM，GPU3/4/5 自然收尾
│      └─> 精确技术终态 + immutable OOM receipt；禁止 adjudication
│          └─> 新 HEAD retry1：GPU0–2 / GPU3–5 两个固定三卡 waves，六臂全重跑
│              └─> historical C0/full + retry1 controls 的 mixed 八臂 cross-root gate
│                  ├─> 科学 NO_GO：停止 Critic 科学分支
│                  └─> PASS
│                      └─> 三 seed full/matched-C0，六 job 六卡并发 confirmation
│                          └─> 一次 atomic frozen Development TEST
│                              └─> 三 seed refit（三卡并发）
│                                  └─> 42-job LOSO（多卡队列）
│                                      └─> Critic frozen readiness
│
└─ SetFlow V4-S1 corrected independent retry
   └─> seed-before-model、canonical-full 投影的 matched full/single initialization
       └─> checkpoints 4/6/8/10 的冻结 891×32 outcome-free Validation
           ├─> XEDITSETFLOW_V4_S1_SCREEN_NO_GO：该 corrected family 终止
           └─> XEDITSETFLOW_V4_S1_SCREEN_PASS
               └─> 新 S1-bound 三 seed confirmation 与 12-job Validation
                   └─> SetFlow G0_READY

Critic frozen readiness + SetFlow G0_READY
    └─> 冻结 SetFlow rollout + 冻结 Critic score target
        └─> value training / Guidance screen frozen
            └─> 三 seed、98-job Final Development + 独立 evaluator
                └─> Final gate PASS：优秀 Development 结果
```

每一箭头都要求前一阶段产物完整、唯一、精确终态、protected reads 合法且冻结 gate 达标。不得以 smoke、proxy、训练集指标、单 seed、单臂、screen、原子 TEST、G0 或 guidance screen 代替后续 gate。

## 5. 后继启动入口

所有后继脚本只从第 2 节工作树的 clean、已推送 HEAD 启动；所有 `--expected-head` 类参数均使用启动时的当前 HEAD。脚本名如下，具体参数由两小时 heartbeat 按 frozen protocol 和当前 receipt 物化，不从历史文档复制：

### 终态转接与正式准入

- `transition_record_route2_xeditsetflow_s1_930_terminal_invalidation.py`
- `transition_record_route2_xeditcritic_v403_controls_oom_terminal.py`
- `verify_and_materialize_route2_xedit_v403_successor_runner_receipts.py`

第一项是 heartbeat 对旧 930 S1 runtime 的唯一 reader。它只在旧 family 为精确科学终态或精确技术终态、
2+8 job inventory 与 adjudication/first-failure 闭合、无 partial/双终态、scheduler 已退出、protected reads=0
时，写 canonical immutable invalidation receipt；该 receipt 永久保持
`successor_authorized=false`、`same_family_retry_authorized=false`，只证明旧 family 已结束且不能作为科学
后继依据。

第二项是旧 Critic controls runtime 的唯一 terminal reader。它只接受包级精确技术终态、六臂全部非
RUNNING/PENDING、scheduler 已退出、唯一 terminal artifact、first failure 不变、无 cross-root gate、protected
reads=0；receipt 永久保持 `successor_authorized=false`、`same_family_retry_authorized=false`，只把新的独立
retry family 标记为 eligible。它不读取旧 failure/summary payload，也不运行 GPU 或模型。

第三项只在主许可分支 fast-forward、push 到新的 clean exact HEAD 后运行一次。它从严格 Critic 与 SetFlow
consumer 的同一八组 marker 构造八个隔离 pytest 进程并发执行，使用实际 PASS 数而非旧固定计数；全部通过后
才运行字面量 `tests/route_a_v3/*v332*.py` 并要求精确 96/0。随后它在 canonical `/mnt` audit 路径原子物化
shared 与 SetFlow 两份不同 schema/status、相同实际证据的 receipt，并调用 Critic controls 与 corrected S1
的真实 pre-GPU consumer 验收。任一 final 或 `.partial` 已存在时不覆盖；consumer 外部条件修复后只使用
`--validate-receipts-only` 复验，不重跑八组与 V3.3.2。该工具不读取实验 runtime/protected outcome、不做
GPU inventory 或模型执行。

### SetFlow

- `launch_route2_xeditsetflow_s1_screen_after_v403_terminal.py`
- `run_route2_xeditsetflow_s1_screen_scheduler.py`

S1 只新增 fixed-weight 0.05 的跨状态候选 mode-responsibility forward KL；根 `EMPTY` slot 0 posterior
停梯度，约束同一 occurrence/canonical candidate 的 compatible nonroot、nonstructural states，归约严格为
state→candidate→occurrence。原 V4 架构、sampler、seed 20260911、十 passes、batch 32、checkpoints
4/6/8/10、891×32 Validation 与全部绝对/相对 gate 不变；不存在 weight sweep 或额外 screen seed。

旧 930 family 的上述 launcher 已消费完毕，禁止再次调用。corrected retry 只能在新 exact HEAD、完整准入和
新 family 路径下消费一次。修复要求 CPU/CUDA seed 在任何模型构造之前应用，并把初始化 seed、应用顺序、
CUDA/A100/BF16 与 no-CPU-fallback 证据贯穿 training summary、checkpoint、Validation 和 gate。canonical
launcher 还必须显式拒绝 930 HEAD，并在 current-HEAD runner receipts、任何 GPU inventory/probe 和 family
创建之前消费上一小节的 terminal-invalidation receipt；launcher 本身不得重读旧 runtime。

- `launch_route2_xeditsetflow_s1_confirmation_after_screen_pass.py`
- `launch_route2_xeditsetflow_s1_confirmation_posttraining.py`
- `adjudicate_route2_xeditsetflow_s1_confirmation.py`

这三个 S1-bound confirmation 入口已在准备分支实现，但当前保持硬锁定：旧 930 family 被显式拒绝，只有
corrected retry 的 exact screen HEAD/path 在未来 commit 中前瞻冻结、并产生精确 PASS 后才能启用。训练固定
三个 `v4_s1_full` seeds 20260912/20260913/20260914；posttraining 固定 3×4 个 Validation，不重训
single-mode，不追加 seed，不预选 confirmation checkpoint。

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

当前 Y3 controls family 已因三项 CUDA OOM 进入技术失败语义，禁止重入。新 HEAD 的 canonical controls launcher
必须先消费 immutable OOM terminal receipt，再消费仓库冻结的 full-terminal audit、clean exact HEAD 与该 HEAD
的 shared runner receipt；它不重读旧 runtime/failure/summary payload。retry1 使用全新 output/runtime/
authorization/log/gate roots与 attempt IDs，六个 controls 固定分配到物理 GPU0–5，但按 GPU0–2、GPU3–5 两个
三卡 waves 运行。第一波任一技术失败时第二波全部 `NOT_RUN_AFTER_TERMINAL_FAILURE`；已在飞臂自然收尾。
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 仅改变 allocator 行为，不改变模型数学或科学协议。
launcher/scheduler 禁止读取、排序、筛选或等待空闲显存，`free_memory_gate_applied=false` 必须贯穿全部证据。

该重试的独立技术语义基线为 `793eedfb4b84e8c0dbd5a30bdf79c8923ddf8110`，对应审计是
`audits/route_a_v3_route2_xeditcritic_v403_controls_oom_retry_training_semantics_793eedfb4b84e8c0dbd5a30bdf79c8923ddf8110.json`。
它保留并继续验证 f1a v2 科学/训练语义审计；f1a→该基线在冻结 pathspec 下只允许 controls
launcher 与 scheduler 两个路径，之后到真实 confirmation runner HEAD 的同 pathspec diff 必须为空。
这是 protected reads=0 的 CPU-native 技术执行证据，不授权同 family 重试、不授权科学后继，也不是模型结果。

六 controls 精确终态后，transition 才一次消费 C0、full 与六 controls 的八份 summary 并执行 mixed-provenance
gate；历史 v1 例外严格只有 C0=937 和 full=f34。screen PASS 后，confirmation 的三个 seed
20260908/20260909/20260910 各训练 full 与 matched C0，共六 job 六卡并发。后续 atomic TEST、三卡 refit 和
42-job 多卡 LOSO 仍逐 gate 串行依赖；Critic readiness 本身不授权 Guidance，必须等待 SetFlow readiness。

SetFlow corrected S1 retry 可与 Critic controls/confirmation 并行推进。两条 family 可按各自冻结物理 GPU
队列同时运行；配置 GPU 存在即可使用，显存只作诊断。任何 CUDA/设备/CPU-fallback/OOM 技术失败均由各
自 package fail closed，不得借另一条分支的状态覆盖或重试同一 family。

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
- Critic controls、confirmation、confirmation posttraining、Atomic TEST wrapper、refit 与 LOSO 的 one-shot
  launcher 在 schedule/authorization/attempt 已物化但 scheduler/wrapper 进程无法启动时，必须在该 family
  写不可覆盖 `scheduler_launch.failed.json`；不得生成 PID、`LAUNCHED` 状态或虚假 launch receipt，同一
  family 不得再次调用。
- Cross-root transition 在读取八臂 terminal payload 前必须验证当前工作树仍为 control schedule 固定的
  clean exact HEAD。任一 identity、payload validation、evaluation 或 gate-write 技术异常必须写 gate sibling
  `.failed.json`，并在下一次调用读取任何 terminal payload 前拒绝同 family 重入；科学 PASS/NO-GO 仍写正常 gate。
- Atomic frozen TEST wrapper 必须在真正 TEST runner 启动和任何 TEST access 之前验证 job-fixed clean exact
  HEAD。identity 或 process-spawn 失败必须以 `development_test_access_started=false`、access count 0、
  outcome reads 0 的 FAILURE + terminal runtime 结束，不得消耗唯一 TEST 授权。
- Refit/LOSO 不仅训练 job，adjudication 与 readiness 的每个 barrier 进程启动前也必须重新验证 schedule-fixed
  clean HEAD。任何 barrier identity/spawn 失败均为包级技术失败，readiness 不运行且
  `guidance_authorized=false`。

## 7. 包级一致性与提交纪律

Critic controls、confirmation training、confirmation posttraining、refit、LOSO、guidance screen 和 final
comparison 均采用包级首失败语义。每个 job 或 barrier 在真正启动前，都必须确认工作树仍等于 schedule
固定的 clean HEAD；并发长队列的终态分类和首失败记录必须在线性化锁内完成。

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
