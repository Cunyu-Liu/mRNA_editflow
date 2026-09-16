# Route 2 Critic：ERK v2 训练臂 Amendment v1（草案，待用户拍板）

**起草：** 2026-09-16 深夜（批次九十九后）
**性质：** 预注册草案——**任何 G/判定门在第 7 节拍板前不生效**；拍板后冻结，修订需 v2。
**前置依据（全部冻结产物，零新计算）：**

1. **D16-C 四门终判闭合**（amendment v1 §4.2，2026-09-16）：G1 FAIL（gap 0.8235 vs V5 0.7943）/ G2 PASS（polyA 0.8134）/ G3 registered（数据侧独立无效）/ G4 FAIL（结构池 graded sc-hit@1 = 0.0）——within-source 密度增强（4.9→32）三证据链（gap 反升 + train rho 反升 0.956 + 结构池零信号）闭合「容量性记忆偏好」反假说。
2. **E7 v2 探针**（erk_e7_v2_20260912.json，CPU 闭式拟合零训练）：一阶上下文能量表 E7-a v2 全任务 PASS——MRL 0.207（gate 0.0677）/ polyA 0.504（gate 0.411）/ MPRAU 0.063（gate 0.0513）；二阶增量 E7-b：polyA +0.0248 PASS、MRL +0.0044 / MPRAU 0.0 不显著；**E7-c v2 混合池 cond_acc@1 = 0.040 FAIL**（gate 0.10，vs V5 0.0614）——上下文修复 on-manifold，不修复 off-manifold。
3. **判定链**：数据侧唯一干预（D16-C）已证伪 → ERK 参数侧成为唯一剩余路径（G3 条款原话）。

## 1. 科学假说（H-ERK-v2）

> 在 frozen mRNABERT 表征之上，以**显式一阶结构先验**（per-position × base-change 上下文能量表，1-3M 参数）替换 170M 自由 critic 头，能（a）在 MRL on-manifold 排序上把 train-backtest gap 从 V5 的 0.794 压到 ≤0.30 且 VALIDATION ρ ≥0.135；（b）不破坏 polyA（≥0.80）；（c）使表征在结构池内产生非零排序信号（graded sc-hit@1 > 0）——即 D16-C 在数据侧未能达成、而留待参数侧达成的同一靶标。

**与 D16-C 的关系**：同一靶标（G1/G4 门）的两个干预臂。D16-C 改数据分布（已 FAIL）；本 amendment 改函数类（参数先验）。主臂用**原始 Development 数据**（不用合成增强——其独立无效性已被 G1/G4 判定）；合成增强仅作为可选对照臂。

## 2. 架构范围（继承 D14 ERK 三层分解，spec 增补·八冻结定义）

- **Energy 模块（显式一阶）**：上下文能量表 E(context, position, base_change)——E7-a v2 已实证的已 PASS 形态，参数量 1-3M（band 预注册）。
- **Residual 模块**：源级残差背景头（轻量，只承载来源锚点信息，不从合成行学习）。
- **Kernel 模块**：编辑间交互核——**本训练臂预注册不启用**（E7-b 只有 polyA 显著；MRL 主目标一阶为主，二阶作为 v2 扩展候选，启动条件 = 一阶臂过门后另立）。
- frozen mRNABERT encoder 不动（113.4M 冻结）；总可训练参数预注册 band：**1M–12M**。

## 3. 预注册判定门（草案，拍板后冻结）

| 门 | 判定 | 阈值 | 口径 |
|---|---|---|---|
| ERK-G1 泛化 | train-backtest gap ≤ 0.30 **且** MRL VAL ρ ≥ 0.135 | 同 D16-C G1（继承连锁，便于对位） | run_v5_train_backtest 镜像协议对 FINAL-EPOCH ckpt |
| ERK-G2 非破坏 | polyA VAL ρ ≥ 0.80 | 同 D16-C G2 | FINAL-EPOCH 行 |
| ERK-G3 反假说登记 | gap > 0.50 时如实入档「参数侧独立无效」 | — | 零事后调整 |
| ERK-G4 结构池 | graded sc-hit@1 > 0 | 同 D16-C G4（D15-2 calibre） | 结构池探针对 FINAL-EPOCH ckpt |

**主门 = G1 + G2**（G4 为表征级直接判据，G3 为反假说兜底）。三 seed（seed 2026091x 系列预注册）全过才晋级；单臂即 G1 过 → 3-seed 扩展臂；G1 FAIL 且 G3 触发 → 参数侧宣告独立无效，路线转「外部高密度数据获取」（真实新库，非合成）。

## 4. 诚实条款（冻结，不得事后弱化）

1. **E7-c 已 FAIL**（0.040 < 0.10）：off-manifold（混合池）迁移优势无先验证据——本训练臂的 off-manifold 效果**不作预注册主张**，仅作探索性报告口径（cond_acc@1 对位 V5 0.0614）。
2. **E7 探针数字 ≠ 训练结果**：探针是闭式拟合的 analytic 上界形状证据，不是梯度训练的预测值；不得把 0.207 写成预期训练 VAL。
3. **D16-C 双 FAIL 继承**：结构池零信号（G4）与数据侧证伪（G3）是既有事实；ERK 一旦在 G4 也 FAIL，须与 D16-C G4 并列呈现为「数据+参数两侧均未使表征学结构」的完整现象学。
4. polyA 是家族最强资产（V5 0.8219）：任何非破坏门外的波动如实报告，不回退主行。
5. protected reads = 0；FINAL-EPOCH-FIXED 禁 peak-picking；CUDA BF16-only（cpu_fallback=false 存证）；产物 /mnt、代码 /home worktree + push。

## 5. 执行清单（拍板后，零自由裁量顺序）

1. [ ] **前置检查**：COMB tier-2 已终态收割（GPU 资源释放确认——与 D16-C 相同的串行纪律）
2. [ ] **实现**：ERK 头（Energy 上下文表 + Residual 轻头）接入 FrozenXEditCriticV5 骨架（encoder 冻结复用）；单测（参数量 band 断言 / 能量表查表语义 / 无合成行依赖）
3. [ ] **冒烟**：MRL 单任务 1 pass（~1 GPU·h）全链出数
4. [ ] **主臂发射**：MRL P0 单臂（GPU 预算 ≤8 GPU·h，整卡，seed 预注册）
5. [ ] **收割**：ERK-G1~G4 全口径（gap backtest 镜像协议 + polyA 非破坏 + 结构池探针复用 `run_d16c_g4_structure_probe_v1.py` calibre 改指向 ERK ckpt）
6. [ ] **分叉**：G1 过 → 3-seed；G1 FAIL → G3 判定入档 + ERK 失败原因归因（对比 D16-C 失败归因写两侧合流结论）
7. [ ] **纪律自查**：八条（见 §4/§5）

## 6. 边界声明

- 不改 V5 架构、不动 V5 榜单主行；不触 TEST（R11 保持）；不把 ERK 训练臂写成 SetFlow 引导 critic 的替换（引导部署决策 = ERK 过门后另议）。
- 合成增强数据（D16-C 产物）本 amendment 主臂不使用；对照臂（如需）另行预注册。
- COMB tier-2 / 论文主线 Task 10.4 的优先级高于本训练臂——GPU 空闲窗口插入，不与在途实验争卡。

## 7. 拍板记录（用户填入）

- [ ] 主臂范围（MRL P0 + polyA 参照）确认
- [ ] 判定门 ERK-G1~G4 确认（或修订 → v2）
- [ ] seed 系列数字确认
- [ ] 批准状态：___________（2026-__-__）
