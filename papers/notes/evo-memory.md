# 精读笔记：Evo-Memory — Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory

- 出处：arXiv:2511.20857（UIUC + Google DeepMind, 2026-05 版）；论文：`papers/evo-memory.pdf`
- 一句话：流式（streaming）自进化记忆评测框架——把数据集组织成连续任务流，要求 agent 在每次交互后检索、整合、更新记忆；统一实现了 10+ 代表性记忆模块做横向对比。

## 核心观点

- 现有评测（LoCoMo/LongMemEval）是**静态对话式**的：记忆被动检索答题。Evo-Memory 考察的是**任务流中的经验积累与复用**——更接近 AML Coding Track 的形态（从历史工程任务复用调试经验）。
- 统一对比了 10+ 记忆模块（可作选型参考表）。

## 对 AML 方案的价值

- Coding Track（SWEContextBench）本地代理评测的设计参照：把历史任务组织成流，考察经验复用率。
- 提醒：AML 文本 Track 是静态形态（先全量 Add 再 Search），Evo-Memory 的流式更新能力属于"加分项"，不必优先实现。

## 可复用要点

- 任务流式评测协议（检索→执行→更新→下一任务）可用于 Coding Track 自测脚本。
- 其统一记忆模块接口（add/search/update）与 AML Add/Search 契约形态一致，本地 harness 可参考其模块抽象。
