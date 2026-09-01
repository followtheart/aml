# 精读笔记：A Survey on the Memory Mechanism of LLM-based Agents

- 出处：arXiv:2404.13501（人大高瓴 + 华为诺亚，2024；高被引综述）；论文：`papers/survey-memory-mechanism.pdf`
- 一句话：系统梳理 agent 记忆的读写/反思机制，抽象通用设计模式。

## 对 AML 方案的价值

- **方案文档的综述引用基座**：记忆操作的"写-读-反思"三分法可组织方案章节。
- 记忆来源分类（trial 内/trial 间/外部知识）对应 AML 的 session 内/跨 session/图谱派生。
- 写作动机："缺乏系统性回顾来抽象共同有效的设计模式"——方案设计文档可引用其分类框架自证完备性。
