# 精读笔记：G-Memory — Tracing Hierarchical Memory for Multi-Agent Systems

- 出处：NeurIPS 2025（arXiv:2506.07398，NUS/NTU 等）；论文：`papers/g-memory.pdf`
- 一句话：组织记忆理论启发的三层图层级——insight graph / query graph / interaction graph，新查询到达时做双向记忆遍历检索。

## 核心机制

- **三层图**：交互层（原始轨迹）→ 查询层（任务级）→ 洞察层（跨任务抽象）。
- **双向遍历**：自顶向下（从洞察到证据）+ 自底向上（从证据到洞察）。
- 针对多智能体系统（MAS）的协作轨迹记忆。

## 对 AML 方案的可复用点

- 单 agent 场景下 MAS 协作层用不上，但**"洞察层"（跨会话抽象记忆）**对 AML 有价值：可从多个 episode 中归纳跨会话规律（如"用户每周三健身"），作为 AMU 的一种派生类型——对应 AML 多跳整合与个性化维度。
- 双向遍历思想 → Search 中"抽象记忆命中后向下取证据条目"（洞察 + 原始证据一起返回，证据合规）。
