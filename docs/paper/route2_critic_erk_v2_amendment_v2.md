# Route 2 Critic：ERK v2 训练臂 Amendment v2（草案 · 待用户拍板）

**起草：** v1 2026-09-16 深夜（批次九十九后）→ v2 修订 2026-09-18（批次一百〇一，用户拍板指令落盘）
**性质：** 预注册草案——**任何 G/判定门在第 7 节拍板前不生效**；拍板后冻结，再修订需 v3。
**前置依据（全部冻结产物，零新计算）：**

1. **D16-C 四门终判闭合**（amendment v1 §4.2，2026-09-16）：G1 FAIL（gap 0.8235 vs V5 0.7943）/ G2 PASS（polyA 0.8134）/ G3 registered（数据侧独立无效）/ G4 FAIL（结构池 graded sc-hit@1 = 0.0）——within-source 密度增强（4.9→32）三证据链（gap 反升 + train rho 反升 0.956 + 结构池零信号）闭合「容量性记忆偏好」反假说。
2. **E7 v2 探针**（erk_e7_v2_20260912.json，CPU 闭式拟合零训练）：一阶上下文能量表 E7-a v2 全任务 PASS——MRL 0.207（gate 0.0677）/ polyA 0.504（gate 0.411）/ MPRAU 0.063（gate 0.0513）；二阶增量 E7-b（立项门 Δρ ≥ +0.02）：**polyA +0.0248 PASS**、MRL +0.0044 / MPRAU 0.0 不显著；**E7-c v2 混合池 cond_acc@1 = 0.040 FAIL**（gate 0.10，vs V5 0.0614）——上下文修复 on-manifold，不修复 off-manifold。
3. **判定链**：数据侧唯一干预（D16-C）已证伪 → ERK 参数侧成为唯一剩余路径（G3 条款原话）。

## 0. v1 → v2 修订记录（2026-09-18 用户拍板指令）

1. **判定门**：ERK-G1~G4 阈值与口径值**不变**；仅修订晋级口径——单 seed 20260913 先行（v1 原文三 seed 表述废止）。
2. **架构范围**：polyA 参照臂**启用二阶 Kernel 交互核**（E7-b 依据：polyA Δρ=+0.0248 过立项门；MRL +0.0044 一阶已饱和、MPRAU 0.0 不带）。MRL 主臂保持一阶；不设 MPRAU 臂。
3. **主臂范围**：确认按 v1 原案——MRL P0 + polyA 参照，原始 Development 数据，合成增强仅作可选对照臂。
4. **seed 系列**：确认先单 seed 20260913（扩展 seed 系列待 G1 过后定）。

## 1. 科学假说（H-ERK-v2，与 v1 一致）

> 在 frozen mRNABERT 表征之上，以**显式一阶结构先验**（per-position × base-change 上下文能量表）替换 170M 自由 critic 头，能（a）在 MRL on-manifold 排序上把 train-backtest gap 从 V5 的 0.794 压到 ≤0.30 且 VALIDATION ρ ≥0.135；（b）不破坏 polyA（≥0.80）；（c）使表征在结构池内产生非零排序信号（graded sc-hit@1 > 0）——即 D16-C 在数据侧未能达成、而留待参数侧达成的同一靶标。

**与 D16-C 的关系**：同一靶标（G1/G4 门）的两个干预臂。D16-C 改数据分布（已 FAIL）；本 amendment 改函数类（参数先验）。主臂用**原始 Development 数据**（不用合成增强——其独立无效性已被 G1/G4 判定）；合成增强仅作为可选对照臂。

## 2. 架构范围（继承 D14 ERK 三层分解，spec 增补·八冻结定义；v2 修订 Kernel 启用条件）

- **Energy 模块（显式一阶）**：上下文能量表 E_task(p, a→b | context_k)——D14 模块 1 冻结形态（L × 3 × 4^k 显式表，非深度网络），MRL 与 polyA **各一任务表**，band 1-3M/表。E7-a v2 已实证的 PASS 形态。
- **Residual 模块**：源级残差/背景头（轻量，≤1M；只承载来源锚点信息，不从合成行学习）。
- **Kernel 模块（v2 修订·启用）**：低秩双线性交互核 I(p,q) = <u_p, v_q>（D14 模块 2 冻结定义，嵌入由能量梯度初始化，反对称性由构造保证）。**启用范围：仅 polyA 参照臂**（E7-b 立项门 Δρ≥+0.02：polyA +0.0248 PASS；MRL +0.0044 一阶饱和不启用；MPRAU 0.0 不设臂）。低秩 rank r ≤ 8 预注册；参数量计入总 band 上限。
- frozen mRNABERT encoder 不动（113.4M 冻结）；**总可训练参数预注册 band：1M–12M（v1 不变，Kernel 计入上限）**。

## 3. 预注册判定门（阈值/口径继承 D16-C，v1 冻结值；v2 仅修订晋级口径）

| 门 | 判定 | 阈值 | 口径 |
|---|---|---|---|
| ERK-G1 泛化 | train-backtest gap ≤ 0.30 **且** MRL VAL ρ ≥ 0.135 | 同 D16-C G1（继承连锁，便于对位） | run_v5_train_backtest 镜像协议对 FINAL-EPOCH ckpt |
| ERK-G2 非破坏 | polyA VAL ρ ≥ 0.80 | 同 D16-C G2 | FINAL-EPOCH 行 |
| ERK-G3 反假说登记 | gap > 0.50 时如实入档「参数侧独立无效」 | — | 零事后调整 |
| ERK-G4 结构池 | graded sc-hit@1 > 0 | 同 D16-C G4（D15-2 calibre） | 结构池探针对 FINAL-EPOCH ckpt |

**晋级口径（v2 修订）**：主门 = G1 + G2（G4 为表征级直接判据，G3 为反假说兜底）。
- **单 seed 20260913 先行发射**；G1+G2 过 → 扩 3-seed 扩展臂（扩展 seed 系列届时预注册）；
- G1 FAIL 且 gap > 0.50 → G3 触发：参数侧宣告独立无效，路线转「外部高密度数据获取」（真实新库，非合成）；
- G1 FAIL 且 0.30 < gap ≤ 0.50 → **amendment 未覆盖地带**：暂停收割、如实报告、用户拍板；
- G1 过但 G2 FAIL → 非破坏条款触发，按诚实条款 4 如实报告，不回退 polyA 主行。

## 4. 诚实条款（冻结，不得事后弱化）

1. **E7-c 已 FAIL**（0.040 < 0.10）：off-manifold（混合池）迁移优势无先验证据——本训练臂的 off-manifold 效果**不作预注册主张**，仅作探索性报告口径（cond_acc@1 对位 V5 0.0614）。
2. **E7 探针数字 ≠ 训练结果**：探针是闭式拟合的 analytic 上界形状证据，不是梯度训练的预测值；不得把 0.207 写成预期训练 VAL。
3. **D16-C 双 FAIL 继承**：结构池零信号（G4）与数据侧证伪（G3）是既有事实；ERK 一旦在 G4 也 FAIL，须与 D16-C G4 并列呈现为「数据+参数两侧均未使表征学结构」的完整现象学。
4. polyA 是家族最强资产（V5 0.8219）：任何非破坏门外的波动如实报告，不回退主行。
5. protected reads = 0；FINAL-EPOCH-FIXED 禁 peak-picking；CUDA BF16-only（cpu_fallback=false 存证）；产物 /mnt、代码 /home worktree + push。

## 5. 执行清单（拍板后，零自由裁量顺序）

1. [ ] **前置检查**：COMB tier-2 已终态收割（GPU 资源释放确认——与 D16-C 相同的串行纪律；当前状态 2026-09-18：main/seed16 已终态、seed17 在途）
2. [ ] **实现**：ERK 头（Energy 表 ×2 + polyA 臂 Kernel 低秩双线性 + Residual 轻头）接入 FrozenXEditCriticV5 骨架（encoder 冻结复用）；单测（参数量 band 断言 / 能量表查表语义 / **Kernel 仅作用于 polyA 头不回流 MRL** / 无合成行依赖）
3. [ ] **冒烟**：MRL 单任务 1 pass（~1 GPU·h）全链出数
4. [ ] **主臂发射**：MRL P0 + polyA 参照（单 seed 20260913，GPU 预算 ≤8 GPU·h，整卡）
5. [ ] **收割**：ERK-G1~G4 全口径（gap backtest 镜像协议 + polyA 非破坏 + 结构池探针复用 `run_d16c_g4_structure_probe_v1.py` calibre 改指向 ERK ckpt）
6. [ ] **分叉**：按 §3 晋级口径执行（3-seed 扩展 / G3 入档 / 未覆盖地带上报 / 非破坏条款）
7. [ ] **纪律自查**：八条（见 §4/§5）

## 6. 边界声明

- 不改 V5 架构、不动 V5 榜单主行；不触 TEST（R11 保持）；不把 ERK 训练臂写成 SetFlow 引导 critic 的替换（引导部署决策 = ERK 过门后另议）。
- 合成增强数据（D16-C 产物）本 amendment 主臂不使用；对照臂（如需）另行预注册。
- COMB tier-2 / 论文主线 Task 10.4 的优先级高于本训练臂——GPU 空闲窗口插入，不与在途实验争卡。

## 7. 拍板记录

- [x] 主臂范围（MRL P0 + polyA 参照，不用合成增强）——确认按 v1 原案（2026-09-18）
- [x] 判定门——确认仅改晋级口径（单 seed 20260913 先行，阈值/口径值不动）（2026-09-18）
- [x] 架构范围——确认 polyA 臂带二阶 Kernel，MRL 一阶，MPRAU 不带（2026-09-18）
- [x] seed 系列——确认先单 seed 20260913；扩展 seed 待 G1 过后定（2026-09-18）
- [ ] 批准状态：___________（2026-__-__）