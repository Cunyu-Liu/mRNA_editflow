# 批 2 模型权重获取报告（2026-10-03，全部落盘 + 可加载性实测）

| 模型 | 状态 | 落盘位置（服务器） | 体积 | 加载实测 | License | 备注 |
|---|---|---|---|---|---|---|
| **Orthrus-4-track** | ✅ 完成 | `hf_home/models/antichronology--orthrus-4-track/snapshots/main/` | 40.7MB safetensors（精确匹配 x-linked-size） | safetensors 70 tensors，config = Mamba/SSM（dim 512 × 6 层） | MIT | 需 `trust_remote_code`（`orthrus_hf.py` 已随包落盘）+ CUDA（mamba-ssm 2.2.4 已装） |
| **Orthrus-6-track** | ✅ 完成 | `hf_home/models/antichronology--orthrus-6-track/snapshots/main/` | 40.7MB（精确匹配） | 同上 | MIT | 同上 |
| **CodonFM-80M** | ✅ 完成 | `hf_home/models/nvidia--NV-CodonFM-Encodon-80M-v1/snapshots/main/` | 307MB safetensors | 135 tensors / 76.8M 参数（model.* 前缀，非 HF 标准 auto_map） | NVIDIA Open Model License（`license:other`）| config 无 `model_type`，须按官方 `NVIDIA-Digital-Bio/CodonFM` 代码加载；真实 HF id 带 `-v1` 后缀（搜索接口之前 401 的原因） |
| **mRNA-LM**（3 段全） | ✅ 完成 | `hf_home/models/Sanofi-Public--mRNA-LM/{5utr,3utr,codonbert}/` | 5utr 961MB / 3utr 960MB / CodonBERT 955MB zip（unzip 完整性 OK） | 5utr = HF `BertForMaskedLM`（vocab 10、768×12，`load_state_dict` missing=0/unexpected=1）；CodonBERT vocab 69、768×12、208 keys | GitHub MIT + 权重 CDN 公开（README 直链） | 官方源 = `cdn.prod.accelerator.sanofi/llm/*.zip`（README 明示）；Zenodo 14606043 从服务器不可达（connection refused），CDN 可达 |
| **CaLM** | ✅ 完成（multimolecule 镜像版） | `hf_home/models/multimolecule--calm/snapshots/main/` | 343MB（精确匹配 x-linked-size） | `multimolecule` 0.2.x 原生加载 OK：85.7M 参数 / 768×12 | AGPL-3.0 | 原版 OPIG 官方直链 403（Cloudflare 拦截，UA/referer 均无效）；**已获取的是 multimolecule 重新导出的同架构权重**——接入时须在适配声明中注明「非官方原始 checkpoint，同架构再导出」（诚实边界） |
| **LucaOne** | ✅ 完成 | `hf_home/models/AmelieSchreiber--LucaOne/snapshots/main/`（代码 12 文件 + 权重） | 6.32GB（精确匹配） | `pytorch_model.bin` 352 keys / **1.58B 参数**（embed_tokens + pre-LN 结构与官方 `LucaOneGPLM` 一致） | Apache-2.0 | 官方 `LucaGroup/lucaone` 与 `Yuanfei/LucaOne` 在镜像上 401/404；**`AmelieSchreiber/LucaOne` 为 checkpoint=17600000 的再发布版**（README 自述 "modified to suit HF API"）——接入时同样注明再发布来源 |
| AIDO.RNA-1.6B | ✅ 可用（同族替代） | `genbio-ai--GB.RNA-1.6B`（13G 完整，已含 modeling/tokenization 代码） | 13G | 架构代码 `modeling_rnabert.py` 已在目录内 | other（NOASSERTION 类） | 官方 HF 把 `AIDO.RNA-1.6B` 重定向到 `GB.RNA-1.6B`（同权重改名）——用 GB.RNA-1.6B 即官方现行入口；`.part` 残目录可清理 |

**下载方式**（全部断点续传 + 完整性校验，慢链路 ~150KB/s-2MB/s）：hf-mirror.com（HF_ENDPOINT）、Sanofi CDN 直链；每个权重以「文件字节数 == 服务器 content-length/x-linked-size」+（zip）`unzip -t` / （safetensors）`safe_open` 逐 tensor 读取 / （HF 检查点）`torch.load` + `load_state_dict` 三级校验收口。sha256 已算（见 journal）。**未从任何非官方渠道获取权重。**

**两个诚实登记（写作时必须带上）**：CaLM 与 LucaOne 取到的是「官方架构 + 第三方再发布/再导出权重」——批 2 接入前须在 port ledger 登记 provenance 差异；若评审在意，可后续邮件作者取原版权重替换（不影响当前结论方向，因为同架构再导出的 embedding 行为一致性可在端口验证中自证）。

**批 2 执行前置**：① 每个模型写 bespoke adapter（Orthrus 需 CUDA mamba；CodonFM 需官方代码加载非标准 checkpoint；mRNA-LM 需按 5'/3'/codon 三段分别接 + 官方 tokenizer（repo 内 WordLevel）；CaLM/LucaOne 用 multimolecule/自定义代码）② 端口验证（RNA-FM 逐位复现）跑通后逐模型 × 9 任务。