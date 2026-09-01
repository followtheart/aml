# 精读笔记：On the Structural Memory of LLM Agents

- 出处：arXiv:2412.15266（Glasgow/Aberdeen, 2024）；论文：`papers/structural-memory.pdf`
- 一句话：受控实证对比四种记忆结构（chunks / 知识三元组 / 原子事实 / 摘要）与三种检索方法（单步 / 重排序 / 迭代检索）。

## 核心结论（直接指导 AML 方案选型）

1. **不同结构各有优势**，应按任务选择——支持方案的多层设计（AMU 原子事实 + episode chunk + 三元组图谱并存）。
2. **混合记忆结构在噪声环境下韧性最强**——方案的"AMU + episode 兜底"双存储由此获得实证支持。
3. **迭代检索在各场景一致优于其他方法**——方案 Search 的多跳子问题分解（多轮召回）对应此结论；但 AML Search 有并发与延迟约束，折中方案：查询理解阶段一次性分解出子查询并行召回（一轮内完成"伪迭代"），不做多轮 LLM 循环。

## 可复用要点

- 该文是方案"为什么 AMU + episode + 图谱三索引并存"的实证引用来源。
- 重排序（reranking）对单步检索有稳定增益——支持方案 5.3 的 gpt-4o-mini 轻量重排。
-  noisy 环境下混合结构韧性最强 → AML 全集数据集多样（locomo/persona/script 等），混合结构是正确赌注。
