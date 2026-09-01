# 精读笔记：M+ — Extending MemoryLLM with Scalable Long-Term Memory

- 出处：arXiv:2502.00592（2025）；论文：`papers/m-plus.pdf`
- 一句话：MemoryLLM 的可扩展续作——潜空间记忆池（1B 参数，16k tokens 内有效）之外挂一个**共训练检索器**的长期记忆，突破 20k tokens 后的遗忘瓶颈。

## 对 AML 方案的价值

- 混合架构启示：**快速内部状态 + 外部检索式长期记忆**的双层组合是参数化路线也在收敛的方向——与方案"内存级 episode 缓存 + 持久 AMU 库"同构。
- AML 同样不可用其训练方法；佐证外部检索式记忆在超长程上不可替代（参数化方法 20k 后衰减，外部记忆无此限）。
