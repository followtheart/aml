# 精读笔记：ReasoningBank — Scaling Agent Self-Evolving with Reasoning Memory

- 出处：ICLR 2026（arXiv:2509.25140，UIUC + Google Cloud AI）；论文：`papers/reasoningbank.pdf`
- 一句话：从 agent **自我评判**的成功与失败经验中蒸馏可泛化的"推理策略"记忆；配套 MaTTS（memory-aware test-time scaling）用更多交互经验加速学习。

## 核心机制

- **策略蒸馏**：不存原始轨迹，存"为什么成功/失败"的推理策略（标题+描述+适用条件）。
- **自我评判**：无需金标——agent 自评成败后蒸馏，闭环自进化。
- **记忆感知测试时扩展**：同一任务多次尝试产生多样经验，再蒸馏。

## 对 AML 方案的可复用点

- "检索想法而非原始数据"的方向（与 Thought-Retriever 同属一派）：Search 返回的 content 可以是**蒸馏后的策略陈述**而非原文——但注意 AML 要求"记忆证据"，蒸馏必须保留可追溯性（raw_refs）。
- 自我评判环路在 AML 静态评测中没有反馈信号（平台不返回对错），故只作 Coding Track 自测框架参考，不进参赛主链路。
- 记忆条目结构设计（标题/描述/适用条件三段式）可借鉴为 AMU 的 type=rule 模板。
