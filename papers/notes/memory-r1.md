# 精读笔记：Memory-R1 — Enhancing LLM Agents to Manage and Utilize Memories via RL

- 出处：arXiv:2508.19828（LMU Munich 等, 2025）；论文：`papers/memory-r1.pdf`
- 一句话：用 RL 训练两个专门 agent——Memory Manager（学会 ADD/UPDATE/DELETE/NOOP 结构化操作）与 Answer Agent（预筛选记忆后推理），奖励来自下游 QA 表现。

## 核心机制

- **Memory Manager**：把 Mem0 式"LLM 直选操作"升级为 RL 训练的策略模型——写入侧控制从提示工程走向学习化。
- **Answer Agent**：先从检索结果中预筛选相关记忆再作答——即"检索后过滤"也是可学习/可提示的模块。
- 用 PPO/GRPO 训练，下游 QA 准确率作奖励。

## 对 AML 方案的可复用点

- **Answer Agent 的预筛选思想**可直接落到 Search 管线：RRF 融合后、重排前加一道 gpt-4o-mini"相关性过滤"（逐条 yes/no 或批量打分），把明显无关的候选在进入 top-100 前剔除——成本低、防噪声证据。
- Memory Manager 证明 ADD/UPDATE/DELETE/NOOP 四操作是治理的正确抽象（与 Mem0 互相印证）。
- RL 训练在 AML 规则下受限（模型必须 gpt-4o-mini，不能换自训模型；评测数据禁止训练）——故 RL 路线不可用，但其操作集与奖励设计可用于本地消融实验。
