# 精读笔记：MemAgent — Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent

- 出处：ICLR 2026（arXiv:2507.02259，清华 AIR + ByteDance Seed）；论文：`papers/memagent.pdf`
- 一句话：分段处理长文 + **覆盖式（overwrite）记忆更新**的 agent 工作流；扩展 DAPO 算法端到端优化记忆能力，多会话独立上下文生成训练。

## 对 AML 方案的价值

- 训练路线不可用；但**分段处理 + 记忆覆盖更新**与 AML 平台分块（20 消息/2000 词切段）形态天然匹配：每个 Add chunk 到达时更新该 session 的滚动摘要（Mem0 式 S）——这是可直接落地的模块。
- 提醒：覆盖式更新有丢信息风险（ACE 的 context collapse），滚动摘要只作抽取上下文、不作检索证据。
