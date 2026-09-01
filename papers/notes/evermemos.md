# 精读笔记：EverMemOS — A Self-Organizing Memory OS for Structured Long-Horizon Reasoning

- 出处：arXiv:2601.02163（EverMind/盛大集团, 2026）；论文：`papers/evermemos.pdf`
- 一句话：engram（记忆印迹）启发的记忆生命周期——Episodic Trace Formation（对话流 → MemCell：情景轨迹+原子事实+时限预见）→ Semantic Consolidation（MemCell 组织为主题 MemScene，蒸馏稳定语义、更新用户画像）→ Reconstructive Recollection（重构式召回）。

## 核心机制

- **MemCell ≈ 方案的 AMU**：原子事实 + 情景轨迹 + 时间边界三合一——当前报告 SOTA（LoCoMo 93.05%、LongMemEval 83.00%、HaluMem 90.04% recall 的自报告数字）。
- **MemScene 主题聚合**：记忆按主题场景组织，而非扁平列表——对应 AML persona/长程整合维度。
- **冲突解决与巩固**是一等公民：consolidation 阶段处理演化经验的一致性。
- **重构式召回**：召回时按场景重构上下文而非返回孤立碎片。

## 对 AML 方案的可复用点

- "MemCell 三合一"验证方案 AMU 字段设计（事实 + episode 引用 + 时间区间）是 SOTA 形态。
- MemScene → 方案可在二期加"主题场景"层（类似 Zep community/Zettelkasten box 的折中）。
- 重构式召回 → Search 返回 content 时附带同场景的最小上下文（方案 5.3 已设计）。
- EverMind 为 AML 相关活跃团队（其博客追踪 AML），其路线代表当前冲榜前沿。
