# 精读笔记：MemoryBank — Enhancing LLMs with Long-Term Memory

- 出处：AAAI 2024（arXiv:2305.10250）；论文：`papers/memorybank.pdf`
- 一句话：拟人记忆机制——艾宾浩斯遗忘曲线驱动的记忆衰减 + 基于过往交互的用户人格画像综合。

## 对 AML 方案的可复用点

- **遗忘曲线衰减**：记忆强度随时间与召回次数变化——可用于 Search 排序的 recency/frequency 加权（Generative Agents 三因子打分的同源思想）。AML 场景慎用：评测事实不应"遗忘"，建议只用作排序微调、不做淘汰。
- **用户画像综合**：从历史交互持续合成 personality 摘要——方案 profile 类 AMU 的参考；画像作为独立直通召回路。
- 个性化维度（AML 七维度之一）的早期经典参照。

## 备注

- 遗忘机制与 AML"事实召回"目标有冲突风险，衰减权重应小、且永不为零淘汰；治理仍以版本链为准。
