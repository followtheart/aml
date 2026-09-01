# 精读笔记：MEM1 — Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents

- 出处：ICLR 2026（arXiv:2506.15841，SMART/NUS/MIT）；论文：`papers/mem1.pdf`
- 一句话：端到端 RL 让 agent 以**恒定内存**运行长程多轮任务——每轮更新一个紧凑共享内部状态，联合支持记忆巩固与推理。

## 对 AML 方案的价值

- RL 训练路线，AML 不可用；但问题定义值得引用：full-context prompting 导致内存无界增长、成本上升、OOD 长度退化——方案"抽取式外部记忆"正是对该问题的工程解。
- "记忆巩固与推理共享状态"的思想 → 方案中查询理解与重排可共享一次 LLM 调用的中间表示（工程优化，可选）。
