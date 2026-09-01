# 精读笔记：HippoRAG — Neurobiologically Inspired Long-Term Memory for LLMs

- 出处：NeurIPS 2024（arXiv:2405.14831）；代码：github.com/OSU-NLP-Group/HippoRAG
- 论文：`papers/hipporag.pdf`
- 一句话：模仿海马体索引理论——LLM 把语料转成无模式知识图谱，检索时以查询实体为种子跑 Personalized PageRank，**单步检索完成多跳推理**，MuSiQue/2Wiki 上超 SOTA 最高 20%，比 IRCoT 便宜 10-30 倍、快 6-13 倍。

## 核心机制

1. **离线索引（≈ Add 侧）**：LLM 对每个段落做 OpenIE 式三元组抽取（主语-关系-宾语），构建无模式 KG；名词短语节点 + 关系边；同义词边通过向量相似度连接。
2. **在线检索（≈ Search 侧）**：
   - 查询 → LLM 抽取命名实体/关键概念（query NER）；
   - 实体链接到 KG 节点（向量匹配）；
   - 以这些节点为种子跑 **Personalized PageRank**——多跳关联在一次图遍历中完成，无需迭代检索；
   - 按 PPR 分数聚合到段落级，输出 top 段落。
3. **path-finding 多跳**：传统 RAG 无法解的"找同时具备 A 和 B 属性的人"类问题（没有任何段落同时提到两者），靠图的关联结构解决——这正是 AML 多跳整合维度的题型本质。

## 对 AML 方案的可复用模块

| HippoRAG 模块 | 复用到方案 | 收益维度 |
|---|---|---|
| 三元组抽取构建无模式 KG | AMU 的 (entity, relation, entity) 三元组表 | 多跳整合 |
| PPR 单步多跳 | Search 图谱路召回：query 实体种子 → PPR → 关联 AMU | 多跳整合（核心武器） |
| 查询实体抽取 + 链接 | 查询理解模块的实体识别子任务 | 召回精度 |
| 同义词/相似节点边 | AMU 间向量相似度 ≥ 阈值建边 | 链接质量 |

## 提示词设计

- 查询侧：命名实体抽取提示词（从问句中提取概念节点用于 PPR 种子）。可并入方案 5.1 的查询理解单次调用中，避免额外 LLM 开销。
- 索引侧：OpenIE 三元组抽取提示词（句子 → (subject, relation, object) 列表）。AML 中可与 AMU 事实抽取合并为一次调用，输出 facts + triples 双份。

## 成本与工程注意

- PPR 是纯图算法、零 LLM 调用、毫秒级——非常适合 AML Search 256 并发场景；LLM 成本全部前置到 Add 侧，与方案"Add 重 Search 轻"原则吻合。
- 无模式 KG 无需 Neo4j：邻接表 + SQLite/Postgres 即可实现 PPR（论文官方实现即如此），符合方案"轻量图谱"决策。
- 续作 HippoRAG 2（From RAG to Memory, ICML 2025）增加 passage 节点与关系消歧，关联记忆 +7%，可作二期升级参考。
