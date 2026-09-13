# AML 记忆系统 Review — 逻辑漏洞 / 性能瓶颈 / 实现与设计差异

> 日期：2026-09-08 · 范围：`agent-memory-system-design.md` v0.2.1、`memory_system/app/*`、`prompts/*`
> 目标函数：提高召回率（返回的 100 条证据含答题所需事实且靠前）与召回性能（Add/Search 延迟、并发容量）

## 一、逻辑漏洞（按召回损害排序）

### P0-1 Embedding 被暴力截断到 256 维
- 位置：`memory_system/app/embeddings.py:43-51`（`_fit_dim`）、`config.py:79`（`EMBED_DIM = 256`）
- 问题：`text-embedding-3-small` 输出 1536 维，代码直接切前 256 维再归一化。这不是 Matryoshka 截断，相似度分布被破坏。向量路是设计中的第一主力召回路（设计 §5.2-1），其排序质量因此大幅下降；治理近邻阈值 `_score > 0.55`（`add_pipeline.py:141`）同样基于该失真空间。
- 修法：调用时传 `dimensions=256`（3-small 原生支持 Matryoshka 短向量），或存全量 1536 维。

### P0-2 嵌入失败静默降级为哈希向量，污染向量空间
- 位置：`embeddings.py:34-40`
- 问题：真实 embedding 调用失败后 fallback 到 `_hash_embed`，与既有真实向量不在同一空间，余弦比较无意义；仅一行 warning，Search 无感知。一次 Add 期间的网络抖动会让该用户向量路从此报废。
- 修法：写入侧硬失败让 Add 重试；或给 AMU 打 `embed_space` 标签，Search 只在同空间内比较。

### P0-3 查询理解的时间锚点用错（时序题集体翻车）
- 位置：`search_pipeline.py:32`（`current_time=datetime.now(...)`）
- 问题：评测对话发生在过去，"last week" 应锚定该用户记忆的最近 event_time / 会话末条消息 timestamp；Add 侧抽取正确地用了消息时间（`add_pipeline.py:20-24`），Search 侧却锚到服务运行当天。写入与查询对同一相对时间解出不同区间，时序证据匹配不上。
- 修法：`current_time` 取 `max(event_time)` / 最近消息 timestamp，无则 now()。

### P0-4 Rerank 全量批打分：调用数失控 + 未评分惩罚不可比
- 位置：`search_pipeline.py:119-167`
- 问题a：对 fused 全量按 `RERANK_CANDIDATES=40` 一批逐批调 LLM，fused 去重后常 200–400 条 → Search 侧 LLM 调用 6–11 次，违反设计 §1.2"仅 2 次"与 §5.3"重排只对 top-30"。
- 问题b：`search_pipeline.py:155` 未评分候选 `_final = _fused * 0.1`（≈0.003）与已评分 relevance（0–1）混排；某批失败或漏评时，高融合分证据被压到榜底，top_k 截断直接丢失。
- 修法：RRF 后先截 top-40，一次性批打分；未评分候选保留 `_fused` 原值参与混排（或单独按融合序追加）。

### P1-5 治理 UPDATE 分支可能清空元数据
- 位置：`add_pipeline.py:176-192`
- 问题：UPDATE 时把 merged_content 当一条新"消息"重跑 `_extract`；若重抽取 facts 为空（merged 文本非对话体，grounded check 易全灭），`entities`/`keywords` 置空、`retrieval_key` 变空串——一次 UPDATE 洗掉一条记忆的全部检索键。且每次 UPDATE 多付 1 次抽取 + 1 次 embed。
- 修法：重抽取为空时保留旧元数据；或直接用 merged_content 单次 embed，元数据取新旧并集。

### P1-6 写入侧 grounded 过滤误杀合法事实
- 位置：`add_pipeline.py:34-47`（`_is_grounded_fact`）
- 问题：要求 fact 至少一个 ≥3 字符词命中本 batch 原文。代词消解/上下文补全后的事实，关键名词可能只存在于 prior 上下文或 session 摘要，整条事实被丢弃进 episode 兜底，原子事实退化为整段对话，精度与召回双输。
- 修法：把 source 扩大到 batch + prior + summary；或对失败样本只做降置信而非丢弃。

### P1-7 CJK 全文检索近乎失效
- 位置：`store.py:295`（`fts_search` 分词）
- 问题：`re.sub(r"[^0-9A-Za-z一-鿿]", "", t)` 把整段中文并成一个 token，FTS5 unicode61 又把整段中文当一个词 → OR 匹配要求文档含整串原文，中文查询召回趋零。`summary_search`（`store.py:373`）同病。
- 修法：FTS5 表改 `tokenize='trigram'`，或写入前对 CJK 分词。

### P2-8 实体未归一化，PPR 种子匹配脆弱
- 位置：`graph.py:17`（仅小写化）
- 问题：查询理解产出的实体与抽取期实体在单复数/别名上不对齐时 PPR 种子落空，多跳路静默失效。
- 修法：抽取与查询两侧共用一个实体归一化函数（去空格、单复数、别名表）。

### P2-9 幂等并发缺陷
- 位置：`main.py:28-33` + `store.py:138-144`
- 问题：两个并发相同 request_id 都通过 `request_seen` 检查 → 都进入 staged → 后者乐观锁失败返回 500。平台会重试并成功，但消耗重试预算。修法：按 request_id 加进程内锁/去重 in-flight map。

## 二、实现与设计差异（design v0.2.1 对照）

| # | 设计 | 实现 | 影响 |
|---|---|---|---|
| D1 | §5.2-5 时序切片召回路（temporal 意图按 time_scope 拉区间） | `_recall`（`search_pipeline.py:82-116`）没有此路；time_scope 只进 rerank prompt | 时序维度缺整条召回路 |
| D2 | §4.2 滚动摘要只作抽取上下文，不作检索证据（防 ACE 坍塌） | `search_pipeline.py:114-115` 把 session_summary 加为召回路并可被 top_k 返回 | 违反设计红线；摘要稀释证据质量 |
| D3 | §5.3 重排只对 top-30；Search 侧共 2 次 LLM 调用 | 全量 fused 批打分（见 P0-4） | Search 延迟/成本 ≥5× |
| D4 | §5.4 低置信弃权返回空数组 | 未实现 | 弃权/安全维度失分，噪声外溢 |
| D5 | §2/§4.3 SeCom 式话题段切片 | 固定 6 条消息硬切（`config.py:77-78`） | 跨段事实割裂 |
| D6 | §4.4 治理比对 = top-5 向量近邻 + 同实体对 AMU | `_govern_one`（`add_pipeline.py:135-160`）只有向量近邻 | 同实体矛盾漏判，治理失分 |
| D7 | §7 不记请求正文日志；评测数据 30 天删除 | `MEMORY_DEBUG_LOG`/`SEARCH_DEBUG_LOG` 默认开启且全量落盘正文（`config.py:86-92`、`search_pipeline.py:194-200`） | 合规风险 |
| D8 | §7 存储 Postgres+pgvector HNSW | SQLite 暴力扫描（`store.py` docstring 已声明为 swap target） | 规模上限低，见 B1-3 |

## 三、性能瓶颈

### B0-1 Add 串行化 + 每次 Add 全库备份（最严重）
- 位置：`store.py:123-152`（`staged()`）
- 问题：全局锁内 `conn.backup()` 整库拷到内存，再乐观锁（data_version+total_changes）重放 SQL。所有 Add 互斥；库涨后每次 Add 拷全库；并发 Add 必然冲突抛 500（`store.py:144`），消耗平台重试预算，极端情况写不落库——直接卡死召回率上限。
- 修法：WAL 模式 + 单写者事务内直写（SQLite 足够），取消内存副本方案；迁移 pgvector 后自然解决。

### B0-2 PPR 每次 Search 重建稠密全图
- 位置：`search_pipeline.py:103-108`（拉全量 triples + 全量 `get_amus_by_ids`）、`graph.py:31-49`（n×n float32 稠密矩阵，20 次幂迭代）
- 问题：5000 三元组 → 节点上万 → 矩阵 ~400MB 级，每次 Search 一次。
- 修法：稀疏邻接表 + 种子 k 跳子图幂迭代；或按 user 缓存图结构增量更新。

### B1-3 向量路暴力全扫 × 6 查询
- 位置：`store.py:267-289`、`search_pipeline.py:98-100`
- 问题：每 query 全量读该用户 AMU 做 matmul，最多 6 次/Search。
- 修法：合并为一次 matmul（查询矩阵×记忆矩阵）；中期换 pgvector HNSW（设计既定目标）。

### B1-4 Search 响应体积失控
- 位置：`search_pipeline.py:186-198`、`store.py:388-392`
- 问题：每条记忆拼全部 source_messages 原文 JSON；session_summary 命中时回传整个 session 消息。top_k=100 响应可达数 MB，挤占平台 Answer 上下文、稀释证据密度。
- 修法：sources 只回 `raw_refs`（request_id+message_index），原文只在 debug 接口提供。

### B2-5 治理逐条串行 LLM 调用
- 位置：`add_pipeline.py:250-258`
- 问题：每条 fact 一次近邻 + 一次治理判定，串行 await。20 事实的 Add = 20 次串行调用。
- 修法：同 request 的 facts 批量一次判定，或与 embedding 并行。

### B2-6 `_recall` 内的全量 ID 反查
- 位置：`search_pipeline.py:106`：为过滤 closed 记忆，对**全部** triple 的 amu_id 做 `get_amus_by_ids`，用户大时 O(全库)。
- 修法：SQL 层 join 过滤（`triples JOIN amu ON ... AND valid_to IS NULL`）。

## 四、修复优先级路线

**第一梯队（召回率，小改动大收益）**
1. embedding 原生 `dimensions` 或全维存储；隔离/删除哈希降级（P0-1、P0-2）
2. `current_time` 锚定用户最近 event_time（P0-3）
3. 补时序切片召回路（D1）；下掉 session_summary 证据路（D2）
4. rerank 前 RRF 截 top-40 单批打分；未评分保留 `_fused`（P0-4）

**第二梯队（召回性能/容量）**
5. `staged()` 改 WAL+事务直写，消除并发 500（B0-1、P2-9）
6. PPR 稀疏子图化；向量扫描合并单 matmul（B0-2、B1-3）
7. sources 只回索引（B1-4）

**第三梯队（治理与合规）**
8. UPDATE 元数据 bug（P1-5）；治理补同实体候选（D6）+ 批量化（B2-5）
9. CJK 分词 / trigram FTS（P1-7）；实体归一化（P2-8）
10. debug 日志默认关（D7）；实现 abstention（D4）；话题段切片（D5）
