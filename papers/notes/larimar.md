# 精读笔记：Larimar — LLMs with Episodic Memory Control

- 出处：ICML 2024（arXiv:2403.11901，IBM Research）；论文：`papers/larimar.pdf`
- 一句话：脑启发分布式情景记忆架构——知识一次性写入、无需重训，事实编辑速度比基线快 8-10 倍，架构简单且 LLM 无关。

## 对 AML 方案的价值

- 参数化/架构级路线（修改模型本身），AML 规则下不可用（Add/Search 必须用 gpt-4o-mini 而非自训模型）——**仅作视野拓展**。
- 可借鉴点：其"一次性写入、即时可检索"的编辑语义与 AML Add 的同步屏障要求一致；方案用数据库事务保证同样的语义。
