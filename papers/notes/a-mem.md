# 精读笔记：A-MEM — Agentic Memory for LLM Agents

- 出处：NeurIPS 2025（arXiv:2502.12110v11）
- 论文：`papers/a-mem.pdf`；代码：github.com/WujiangXu/AgenticMemory
- 一句话：Zettelkasten 卡片盒启发的自组织记忆——每条记忆是带多维属性的"原子笔记"，新记忆入库时自动建立链接并触发旧记忆演化。

## 核心机制

1. **Note Construction**：每条记忆 m = {content, timestamp, Keywords, Tags, conteXt（一句话上下文描述）, embedding, Links}。embedding 由 content+keywords+tags+context 拼接后编码（ei = f_enc[concat(ci, Ki, Gi, Xi)]）——**多字段联合编码**而非只编码原文。
2. **Link Generation**：新笔记先向量召回 top-k 近邻，再用 LLM（Ps2）判断是否应建立链接——相似度做粗筛，LLM 做精判，兼顾扩展性与语义细微关系。
3. **Memory Evolution**：LLM（Ps3）决定新记忆是否触发演化，可执行动作：strengthen（强化链接）、update_neighbor（更新邻居的 context/tags）、merge、prune。**旧记忆的属性可被新经验改写**——这是与 Mem0 的关键差异。
4. **Retrieval**：查询编码 → top-k 命中 → **同 box 内链接的记忆自动级联带出**（一跳扩展），再交给 LLM。

## 提示词设计（附录 B，可直接复用）

- **Ps1 笔记构建**：要求输出 JSON {keywords（≥3 个，按重要性排序，排除说话人名和时间）, context（一句话概括主题/论点/受众）, tags（≥3 个，含领域/格式/类型）}。
  - 复用点：AML 的 AMU 抽取提示词可直接借鉴此 JSON 结构；"排除说话人名和时间"这条约束很实用（避免关键词被噪声占据）。
- **Ps3 记忆演化**：返回 JSON {should_evolve, actions: [strengthen/merge/prune], suggested_connections, tags_to_update, new_context_neighborhood, new_tags_neighborhood}。
  - 复用点：AML 治理模块的 duplicate/update/contradiction 判定可扩展为此结构化决策格式。

## 对 AML 方案的可复用模块

| A-MEM 模块 | 复用到方案 | 注意 |
|---|---|---|
| 多字段联合 embedding | AMU embedding 用 content+entities+type 拼接编码 | 提升专有名词召回 |
| 链接级联召回 | Search 的图谱一跳扩展 | 命中记忆时带出 linked AMU |
| LLM 精判链接/演化 | 治理模块 duplicate/update/contradiction 判定 | A-MEM 全 LLM 判定成本高，AML 需压缩到 top-5 近邻 |
| 原子笔记原则 | AMU 原子事实抽取 | 一条一事实，利于治理与时序 |

## 成本警示

- 论文实验中 GPT-4o-mini 配置 k=40~50 的近邻规模；AML 256 并发 Search 下，演化/链接的 LLM 调用只应发生在 Add 侧。
- 每条记忆的演化可能改写邻居（级联更新），需要事务保护避免并发写冲突（AML Add 并发 16-64）。
