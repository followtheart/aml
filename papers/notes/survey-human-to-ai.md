# 精读笔记：From Human Memory to AI Memory — A Survey on Memory Mechanisms in the Era of LLMs

- 出处：arXiv:2504.15965（华为诺亚方舟实验室, 2025）；论文：`papers/survey-human-to-ai.pdf`
- 一句话：以人类记忆（编码-存储-检索；感觉/短期/长期；情景/语义/程序记忆）为对照系，系统映射 LLM 记忆机制。

## 对 AML 方案的价值

- **AMU type 分类的理论依据**：情景记忆（episode）/ 语义记忆（fact）/ 程序记忆（rule/workflow）三分正对应方案的类型字段——综述为此提供心理学背书。
- 编码-存储-检索三阶段 ↔ LongMemEval 的 indexing/retrieval/reading ↔ 方案 Add/Search 管线——叙事一致性。
- 巩固（consolidation：短期→长期转化）概念 → 方案的 Add 侧治理即"计算式巩固"。
