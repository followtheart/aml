# 精读笔记：MIRIX — Multi-Agent Memory System for LLM-Based Agents

- 出处：arXiv:2507.07957（MIRIX AI, 2025）；论文：`papers/mirix.pdf`
- 一句话：六型记忆 + 多智能体协同——Core / Episodic / Semantic / Procedural / Resource / Knowledge Vault 六种记忆类型，由多 agent 框架动态协调更新与检索；多模态（视觉截图）记忆是主打差异。

## 对 AML 方案的可复用点

- **记忆类型分类学**是目前最细的：方案的 AMU type 字段可对照扩展——core（画像常量）/ episodic（事件）/ semantic（事实）/ procedural（规则流程）/ resource（文档资源）/ vault（凭证类敏感信息）。
- **Knowledge Vault 对敏感信息单独成库**——对应 AML 安全与隐私维度：敏感记忆标记后可在 Search 中加过滤策略。
- 多 agent 协调（meta manager + 各记忆 manager）对 AML 过重（延迟与成本），但其"按记忆类型路由处理"的思想可用轻量规则实现。
- AML 设有多模态 Track（ATM-Bench/Mem-Gallery）——若未来扩展，MIRIX 是主要参照。
