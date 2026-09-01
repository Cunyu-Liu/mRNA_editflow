# V7 验收 Checklist

## 预注册
- [x] 设计冻结：梯度范数缩放（任务间、EMA、几何均值中心化）落 spec.md
- [ ] 验收门槛冻结：主判据 ρ>0.103 且 CI 不跨零 + ≥V6 0.051 有统计证据；辅助偏移头区分度；工程五要素
- [ ] 同 seed 20260907、同 V6 数据/架构/其余字段对照声明

## 实现与单测
- [ ] 开关关 = V6 loss bit 级一致（单测）
- [ ] 开关开 = 梯度范数均衡（单测）
- [ ] EMA/防护/有限性（单测）
- [ ] 既有测试全绿（≥36 项）

## 工程前置
- [ ] 参数 165–175M 预flight PASS；CUDA/BF16、cpu_fallback=false
- [ ] protected reads=0（dev/eval 零访问）
- [ ] exact-HEAD 授权链 + A100 sync audit
- [ ] 独立 family，不动 V4.0.3/S1/V6 gate

## 训练与判定
- [ ] 8/8 pass、22,416 updates、per-pass checkpoint
- [ ] 主判据核算落盘（pair 级 bootstrap CI 口径一致）
- [ ] 过门 → W2-b；未过门 → V8 协议修订
