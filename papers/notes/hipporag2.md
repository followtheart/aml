# 精读笔记：HippoRAG 2 — From RAG to Memory: Non-Parametric Continual Learning for LLMs

- 出处：ICML 2025（arXiv:2502.14802，OSU）；论文：`papers/hipporag2.pdf`
- 一句话：HippoRAG 的续作——发现纯 KG 方法在**基础事实记忆任务上显著弱于标准 RAG**，通过更深的 passage 集成与关系消歧修复，成为事实/关联/意义建构三类任务全面超 RAG 的框架。

## 核心发现（对 AML 极其重要的警示）

- **HippoRAG 1 的教训：图检索会牺牲简单事实召回**——dense vector 对"直接事实查询"仍是最强基线。方案把向量检索作为第一路主力召回、图谱只作扩展路，正是规避此坑。
- 修复手段：passage 节点直接入图（query 可同时匹配 passage 与实体）、关系消歧（同义边合并）、PPR 权重调优。

## 可复用模块

- **passage 节点入图**：AML 图谱中除实体节点外保留 AMU 节点，PPR 种子同时覆盖两者。
- **关系消歧**：同义关系标签归一（"lives_in" ≈ "resides_in"），Add 时做轻量归一化。
- 关联记忆任务 +7% 的实测增益，作为方案图谱路的升级路径（先跑通 HippoRAG 1 式轻量版，再升级到 2）。
