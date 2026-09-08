# V9-1a Adapter-Zoo 预注册（route2_v9_adapter_zoo v1）

**状态**：FROZEN（2026-09-08；SPECS_CRITIC_V6 spec「2026-09-08 增补」N.4 双轨执行案——批准即立项，本文件为 Task 12 执行前置预注册）
**依据**：V9-0 裁决（批次五十三：M1 合并 7/7 FAIL → adapter-zoo 路线确认；M2 任务向量正交；M2b 共享头梯度冲突 −0.92~−0.96 → per-task 头结构性必要）

## 1. 架构（固定条款，spec N.4.4）

- **底座**：V8-S Stage 1 联合先验 checkpoint（`v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt`），整体冻结（base.requires_grad_(False)）。
- **任务专属 LoRA**：每个 benchmark domain（9 个）一组 LoRA，注入全部 12 层的 QKV/attention.dense/gated_layers/wo（复用 V8 wrap_lora 目标集），rank=16 α=32 dropout=0.05——梯度只来自该任务（PLE"任务专属 expert"极致版）。
- **共享 LoRA**：一组 rank=32 的跨任务 LoRA（同一注入位置），梯度来自全部任务（PLE"共享 expert"）。有效更新 = 共享 LoRA + 当前任务专属 LoRA 之和。
- **per-task 线性头**：每 domain 一个 768→1 头（M2b 证据：共享头是梯度冲突主战场 all-params cos −0.92~−0.96 vs backbone-only −0.12~−0.28；per-task 头结构性消解）。
- **polyA CNN stem（固定条款）**：polya domain（domain_id=1）forward 时注入 CNN motif stem（权重从 `stage1_h_epoch2.pt` 的 `stem.*` 拷贝初始化，随适配器可训练；S trunk + H stem 嫁接的分布失配风险登记，门② polyA ≥0.80 暴露即处置）。
- **路由**：task_id 确定性路由 = 训练/评估的 domain_ids 直接索引 LoRA 组与头（不学路由器——V5 路由塌缩教训，spec 固定条款）。batch 保持 task-homogeneous（V8 逐 study 子 batch 模式不变）。
- **domain/cell embeddings**：保留 V8 几何（9 域/6 细胞），Stage 1 前 3 行 init，随适配器可训练（共享组件）。

## 2. 训练协议

- 数据：benchmark TRAIN pool（`load_benchmark_domains`，与 V8 Stage 2 均衡适配臂同源同采样器 DomainBalancedSampler）；MPRAU 行保留原始每细胞标签 + cell conditioning（V6 λ 教训条款）。
- 损失：pair-delta MSE（z-scored per-study target，与 V8 Stage 2 完全同口径：source/candidate 各 forward 取差对 target 回归）。
- 优化：AdamW lr 2e-5 wd 1e-4（LoRA 参数），cosine to 10% + 5% warmup（V8 Stage 2 同款），batch 128（domain-balanced），epochs 6，seed ∈ {20260907, 20260911, 20260915}（V5/V8 系 seed 惯例），FINAL-EPOCH-6-FIXED（禁 peak-picking）。
- CUDA BF16-only（cpu_fallback_used=false 留证）；protected reads = 0；产物 /mnt、代码 /home worktree + push。
- 预算：3 seeds × ~8-12h（manager 排程，GPU 空闲即发）；A/B 共享方向消融（shared-A / shared-B）与 symmetric 主配置的关系：**v1 只训 symmetric × 3 seeds**，消融臂待主配置门①②结果后另案增补（报告级，不设门——预注册留痕）。

## 3. 判定门（预注册，不改；spec N.4.3 全文引用生效）

- **门①（混合池探针硬门）**：`evaluate_route2_mixed_pool_probe_v1.py` per-task cond_acc@1 ≥ 该任务历史最强单模型 − 0.005——参照行（`mixed_pool_probe_reference_v1.json`）：MPRAU 0.1497 / MRL 0.0422 / polyA 0.1750（n=20）/ HL 0.1280；V5 总体 0.0614。
- **门②（on-manifold 9 任务表）**：MRL ≥ 0.28 / polyA ≥ 0.80 / MPRAU pair-mean > 0.1351 且 paired bootstrap CI 不跨零（D5 近期带）/ TE 族 ≥ 内靶 0.1317 / task-macro ≥ 0.167（V8 Stage 2 门全量沿用）。
- **门③（V9-2 guided 终判）**：最优配置接入 SetFlow guided runner（potentials 非常数断言前置），891 源全量 vs unguided 0.12046 / V5-critic guided 0.12626；B2 原门（Δrecovery ≥ +0.05 且 CI 不跨零且 hit@1 不劣化）+ per-task 分解。
- **失败回退梯（spec N.4.4 顺序）**：门①② FAIL → CPI 式核心参数移植（M2 热图指导）→ 数据全量扩充重训（轨道 B）→ 数据体制边界结论。

## 4. 3-seed 消费

- 3 seeds 终态后：主判据 = 3-seed mean ± std（per-task）+ 最优单 seed 不作选择依据（FINAL-EPOCH-FIXED + seed 全报）；MPRAU 门②用 3-seed ensemble pair-mean（预测 z-mean，V8 5-seed 先例口径）+ paired bootstrap vs V5 0.1025 与 s_mprau_in 0.1351。

## 5. R3 泄漏边界

- 训练数据 = 冻结 benchmark TRAIN split（既有管线已隔离）；无新外部数据入训练（轨道 B 数据审计另行，未过审不入）。底座 = Stage 1 联合先验（280K/APA 库均已过鸽笼审计 flagged=0）。本臂无新增泄漏面。

## 6. 预注册时点状态（防事后翻案）

- V9-0 三件套终态已知（M1 FAIL / M2 正交 / M2b 头冲突）；本预注册据此设计，不依赖任何 V9-1 训练期信息。
- 写死条款：polyA stem、per-task head、shared LoRA 必须存在（三者是 V9-a 与"逐任务独立微调"的身份区别）；不学路由器；LoRA r16α32/共享 r32。
