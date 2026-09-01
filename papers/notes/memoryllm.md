# 精读笔记：MEMORYLLM — Towards Self-Updatable Large Language Models

- 出处：ICML 2024（arXiv:2402.04624）；论文：`papers/memoryllm.pdf`
- 一句话：在 transformer 潜空间内置固定大小记忆池，模型自我更新注入文本知识并保留先前记忆——参数化自更新路线开山作。

## 对 AML 方案的价值

- 参数化路线，AML 不可用（模型锁定 gpt-4o-mini）——视野拓展。
- 思想借鉴：固定容量记忆池的"自更新淘汰"对应外部记忆的容量管理；方案 AMU 不做硬淘汰（治理用版本链），但 profile 类高频条目可做合并压缩。
