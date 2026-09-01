# 精读笔记：Zep — A Temporal Knowledge Graph Architecture for Agent Memory

- 出处：arXiv:2501.13956（Zep AI 公司技术报告，2025-01）
- 论文：`papers/zep.pdf`；开源引擎：Graphiti
- 一句话：双时间轴（bi-temporal）时序知识图谱作为 agent 记忆层，LongMemEval 上准确率最高 +18.5%、延迟降 90%。

## 核心机制

1. **三层子图**：episode 子图（原始消息，非有损存储）→ semantic entity 子图（实体+关系）→ community 子图（实体聚类的高层摘要，label propagation 动态扩展）。
   - **原始数据与派生事实双存储**——直接印证方案中"AMU + episode 兜底"的设计。
2. **双时间轴（最关键的可复用点）**：T = 事件时间线（t_valid / t_invalid，事实在真实世界生效的区间）；T′ = 事务时间线（t′_created / t′_expired，系统何时写入/废止）。
   - 方案 AMU 的 event_time / valid_from / valid_to 即源于此。
3. **Edge Invalidation（边失效）**：新边入库时用 LLM 与"同实体对的语义相关旧边"比对；发现时间上重叠的矛盾时，把旧边的 t_invalid 设为失效边的 t_valid。**不删除，只关闭有效期**；新信息优先。
   - 限定"同实体对"内比对，既防误伤又把去重计算复杂度从 O(N) 降到子集——此技巧必须复用。
4. **实体解析**：embedding 召回 + 全文检索双路候选 → LLM 判定是否同一实体 → 合并并更新 name/summary。用**预定义 Cypher 查询**而非 LLM 生成数据库语句（防幻觉）。
5. **检索三段式**：f = χ(ρ(ϕ(α)))：Search（cosine + BM25 + BFS 图扩展三路）→ Reranker → Constructor（把 fact 带 t_valid/t_invalid 格式化成交给回答模型的文本）。
   - 上下文模板：`FACT (Date range: from - to)` + `ENTITY_NAME: entity summary`——**时间区间直接写进证据文本**，让 Answer 模型自己完成时序推理。

## 提示词设计（附录，复用点）

- **实体抽取**：处理当前消息 + 前 n=4 条消息（两个完整轮次）作为指代消解上下文；再用 reflexion 式反思降低幻觉。
- **相对时间归一化**：以消息的 reference timestamp（t_ref）为锚，把 "next Thursday" / "two weeks ago" 解析为绝对时间——方案 4.2 的"时间表达归一化"直接采用。
- **Constructor 模板**（上引）——AML Search 返回的 content 可内嵌 `[valid: 2023-05 ~ 2024-01]` 前缀，帮助平台 Answer 模型答时序题。

## 对 AML 方案的可复用模块

| Zep 模块 | 复用到方案 | 收益维度 |
|---|---|---|
| 双时间轴 + 边失效（不删除） | AMU valid_from/to + supersedes 版本链 | 时序理解、记忆治理 |
| 同实体对限定比对 | 治理模块只在 top-5 近邻/同实体内判定 | 成本、并发安全 |
| episode/semantic 双层 | AMU + episode 原始切片兜底 | 事实召回下限 |
| 三路混合检索 | 向量 + BM25 + 图谱 BFS | 多跳整合 |
| 证据内嵌时间区间 | content 前附加 `[时间区间]` | 时序理解（白捡的分） |
| 预定义查询代替 LLM 生成 SQL/Cypher | 索引写入用固定 SQL | 防注入、可审计 |

## 注意

- 公司技术报告，非同行评审；但 Graphiti 开源可核验。
- community 子图（高层摘要）对 AML 的 persona/profile 维度有参考价值，但构建成本高，可作为二期优化。
