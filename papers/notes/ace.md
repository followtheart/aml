# 精读笔记：ACE — Agentic Context Engineering: Evolving Contexts for Self-Improving LMs

- 出处：ICLR 2026（arXiv:2510.04618，Stanford/SambaNova）；论文：`papers/ace.pdf`
- 一句话：把上下文当作"可演化 playbook"——生成、反思、策展三段式模块化更新，防止"简洁偏差（brevity bias）"与"上下文坍塌（context collapse）"。

## 核心机制与警示

- **Context collapse 警示**：迭代式让 LLM 整体改写记忆/摘要会逐步丢失细节——直接警示方案：AMU 治理时**不要整体重写记忆库**，只做增量式、条目级更新（方案采用条目级版本链，规避此风险）。
- **结构化增量更新**：条目化 playbook，每条独立增删改，而非整篇重生成。
- 生成（新经验条目）→ 反思（条目质量/相关性）→ 策展（去重、合并、淘汰）三段式，对应方案治理管线的可扩展形态。

## 对 AML 方案的可复用点

- 治理模块设计原则：**条目级增量更新 + 保留细节**，禁止全局压缩重写。
- profile/画像类记忆可组织为 playbook 式条目列表（每条偏好独立成条目，可增删），优于整段画像描述——利于更新题答对。
