# AML 参赛记忆系统设计方案（v0.4）

> 目标平台：Agent Memory Leaderboard（agentmemoryleaderboard.ai）
> 目标评测：Textual Track（学术方法榜），兼顾 Coding Track 扩展
> v0.1：2026-09-01 初版（基于官方文档/API 指南）
> v0.2：2026-09-01 依据 33 篇前沿论文精读（`papers/notes/`）修订：
> ① 图谱召回降级为扩展路（HippoRAG 2 教训）；② 治理改为条目级增量更新（ACE 教训）；
> ③ DELETE 改为软废止（Zep 双时间轴）；④ 提示词资产独立成 `prompts/` 模板文件。
> v0.2.1：2026-09-01 LLM 抽象改为 LiteLLM（`AML_LLM_MODEL` 可配置，默认 gpt-4o-mini）；
> M1–M3 已实现并通过验证，代码在 `memory_system/`（契约自测 11/11，LoCoMo 管线冒烟通过）。
> v0.3：2026-09-15 按 `llm-memory-survey/memory-system-design.md`（ULM）§11 落地七项增量：
> ① 语义边界切分（`segment.py`）；② MemCell = episode + facts，MemScene 巩固 + 场景→情景两阶段召回（`scenes.py`）；
> ③ `plan` 类型前瞻信号 + 时效状态/过滤；④ MemoryOS 热度晋升/驱逐 + 画像 stable/transient + rules 常驻；
> ⑤ 充分性验证器（`prompts/08`）驱动 ≤2 轮迭代召回与低置信弃权（补齐 REVIEW D4）；
> ⑥ 惊奇度/新颖度写入门控（重复与全新事实不调 LLM）；⑦ 艾宾浩斯冷热分层（默认关闭，不删除）。
> 同时修复 REVIEW P0-1（Matryoshka 短向量）、P0-3（时间锚点取用户最新记忆）、P0-4（重排仅头部 40、未评分不惩罚）、
> P1-5（UPDATE 不重抽取，元数据取并集）、D2（会话摘要默认不作检索证据）。细节见 `memory_system/STORAGE.md`。
> v0.4：补齐敏感记忆双重授权、点时态版本折叠、画像状态/TTL、Memory-Only/Memory-Doc、
> 冷 Scene 回温、显式任务反馈经验路和合规 purge；RRF 按通道计票，热度改为饱和衰减公式。

---

## 1. 平台约束与需求提炼

### 1.1 接口契约（硬性）

- **Add（同步）**：`POST {request_id, messages[{role, content, timestamp?}], user_id, session_id}`
  - 返回 HTTP 200 + `{success: true, request_id, user_id, session_id}`，三个 ID **逐字节回显**。
  - **必须等写入完成且可被 Search 立即检索后才能返回 200**（内部可异步，对外同步）。
  - 不得返回 202 / task_id / 轮询地址；响应不需要 memory_ids。
- **Search**：`POST {query, options?, user_id, top_k}`
  - 返回 `{"data": [{id, content, score?, created_at?}]}`，按相关性降序；不超过 `top_k`（正式评测固定 100）。
  - **只返回记忆证据**：不得生成最终答案、不得把答案伪装成记忆；无结果返回空数组。
- **Health**：无鉴权 GET，任意 2xx；默认检查 Add 同源 `/health`。
- **鉴权**：Token / Bearer / X-Api-Key 三选一（正式评测必须启用）。
- **user_id 是唯一检索隔离边界**：存储层强制行级隔离，不依赖查询层自觉。

### 1.2 评测行为约束

- **Add/Search 内部 LLM 必须满足平台模型要求**（第一期 Full 前置清单第 3 条为 gpt-4o-mini；平台复现，分数差异过大作废）。实现上**统一走 LiteLLM 抽象**（`AML_LLM_MODEL` 环境变量，默认 `gpt-4o-mini`，可切换任意 litellm 支持的模型），温度 0；第二期开放后先复核该条款再定默认值。Mem0 论文（ECAI 2025）即为全链路 gpt-4o-mini 的合规范本。
- 平台分块：每个来源会话默认一次 Add；**超过 20 条消息或 2000 词按消息/句子边界分段**。
- 并发目标：Add 16–64，Search 16–256。当前 SQLite 参考实现支持单进程并发准备、短事务发布和 revision 冲突内部重试，但不宣称水平扩展；生产需切换 Postgres/pgvector 与数据库级 request claim。
- Search 首轮最多执行查询理解、一次小 R 重排和一次充分性验证；追加轮最多再执行一次重排和验证。调用量由轮数和候选硬上限约束，不再假设固定两次。
- 错误处理：Add 遇 408/409/425/429/5xx 平台有限重试 → Add 以 request_id 幂等去重；400/422 不重试 → 响应格式必须一次正确。
- 托管接口公网可达、稳定 ≥30 天；评测数据 30 天内删除、禁止训练/外泄。
- **规则复核待办**：第二期预计 2026-09-20 开放；开放后第一件事重新核对评测页 Full 前置清单（含 gpt-4o-mini 条款）是否有变化。本方案所有模型相关设计均为配置项，规则变动时改动成本极低。

### 1.3 评分维度（设计目标函数）

文本 Track 七维度：事实召回、多跳整合、时序理解、记忆治理、个性化、规则执行、认知安全与隐私。第一期榜首 58.02（MemoraX），**多跳、时序、治理是主要失分区**，为本方案差异化重点。维度-模块映射见 §9。

---

## 2. 总体架构

```
                ┌────────────────────── Add Pipeline ──────────────────────┐
 Add 请求 ────► │ 1. 幂等去重(request_id)                                   │
                │ 2. 消息规范化 + 滚动 session 摘要 S 更新(Mem0/MemAgent 式) │
                │ 3. 话题段切片(SeCom 式 episode segmentation)              │
                │ 4. gpt-4o-mini 原子事实抽取 → AMU（一次调用出 facts+三元组)│
                │ 5. 条目级治理：ADD/UPDATE/SUPERSEDE/NOOP（软废止，不删除） │
                │ 6. 多索引写入：向量 + BM25 + 时序 + 轻量图谱               │
                │ 7. 全部落库成功 → 返回 200（同步屏障）                     │
                └──────────────────────────────────────────────────────────┘
                                       │ 持久层（user_id 行级隔离）
                ┌────────────────────── Search Pipeline ───────────────────┐
 Search 请求 ──► │ 1. 查询理解(gpt-4o-mini 单次)：意图/时间锚定/实体/子问题  │
                │ 2. 多路召回：向量(主力) + BM25 + 图谱PPR扩展 + 时序切片    │
                │    + 画像/规则直通                                        │
                │ 3. RRF 融合去重 → 相关性过滤 → 治理感知排序(版本链取最新)  │
                │ 4. gpt-4o-mini 重排 top-30，其余按融合分                  │
                │ 5. content 内嵌时间区间 → 截断 top_k=100                  │
                └──────────────────────────────────────────────────────────┘
```

核心原则：**Add 侧做重（写入时结构化），Search 侧做快而准**。平台 Answer 模型固定，分数完全取决于"返回的 100 条证据是否包含且靠前包含答题所需事实"。

---

## 3. 记忆数据模型：原子记忆单元（AMU）

```json
{
  "id": "amu_<uuid>",
  "user_id": "...",
  "session_id": "...",
  "content": "记忆文本（面向检索的陈述句事实，禁止指令式文本）",
  "retrieval_key": "问题形式改写（When did Dave start photography?）",
  "raw_refs": ["session:0:msg:3", "session:0:msg:7"],
  "type": "fact | preference | rule | workflow | event | profile | episode",
  "entities": ["Caroline", "adoption agency"],
  "keywords": ["photography", "scenery", "hobby"],
  "event_time": 1704067200000,
  "valid_from": 1704067200000,
  "valid_to": null,
  "supersedes": "amu_older_id",
  "confidence": 0.9,
  "sensitivity": "normal | sensitive",
  "embedding": "[content + retrieval_key + keywords + entities 拼接编码]",
  "created_at": "2026-07-01T12:00:00Z"
}
```

v0.2 修订（来源见 `papers/notes/README.md` 对照表）：
- **双时间轴**（event_time vs valid_from/to）：Zep 双时间模型的直接应用，时序题与治理题基础。
- **软废止版本链**：知识更新不删除旧记忆，`valid_to` 关闭 + `supersedes` 链接（修正 Mem0 硬 DELETE 的缺陷）。
- **retrieval_key 字段**（新增）：LongMemEval 的 fact-augmented key expansion——检索键与存储值解耦，问句形式改写提升问句-事实匹配率。
- **keywords/entities 联合编码**（新增）：A-MEM 多字段联合 embedding。
- **sensitivity 字段**（新增）：敏感记忆默认从向量、FTS、图、时序、Scene 和画像路全部排除；只有服务策略与请求双重 opt-in 才可召回。
- **episode 类型保留原始切片**：Zep episode/semantic 双层 + structural-memory 混合结构韧性——抽取失败时兜底召回。
- **raw_refs 溯源**：平台审计 + 本地调试。
- **content 禁止指令式文本**：记忆投毒防御（survey-security WRITE 阶段）。

---

## 4. Add 管线详细设计

### 4.1 幂等与同步屏障
- `request_id` 主键去重；重复请求直接回显成功。
- 内部队列并行，但响应等待该 request_id 全部子任务完成（写入即可检索）。
- 降级：LLM 抽取超时则**至少把原始 chunk 作为 episode AMU 落库**——保证召回下限。

### 4.2 滚动会话摘要（v0.2 新增）
- 每个 session_id 维护滚动摘要 S（Mem0 式异步刷新；MemAgent 式分块更新）。
- S 作为下一次抽取的全局上下文；**S 只作抽取上下文，不作检索证据**（防 ACE 式上下文坍塌）。

### 4.3 原子事实抽取（gpt-4o-mini，一次调用）
提示词模板：`prompts/01_extract_amu.txt`。单 chunk 单次调用，输出 JSON：
- **facts**：原子事实陈述句（一条一事实），含 entities、keywords、type、event_time（相对时间以消息 timestamp 为锚归一化，Zep t_ref 法）、retrieval_key；
- **triples**：(entity, relation, entity) 三元组（HippoRAG OpenIE 式，与 facts 同次调用产出，零额外成本）；
- **偏好/规则/流程**单独标记 type=preference/rule/workflow（个性化与规则执行维度直通召回）。

### 4.4 条目级记忆治理（写入时消解）
提示词模板：`prompts/02_governance_decision.txt`。
- 新 AMU 与**同 user_id 下 top-5 向量近邻 + 同实体对 AMU**（Zep 限定比对范围技巧，控成本防误伤）做关系判定；
- 输出四操作（Mem0 框架，DELETE 替换为 SUPERSEDE）：
  - `ADD`：无等价记忆 → 直接入库；
  - `UPDATE`：互补信息 → 合并进现有 AMU，confidence 提升；
  - `SUPERSEDE`：矛盾/更新 → 旧 AMU 关闭 valid_to，新 AMU 链接 supersedes（**软废止，历史保留**）；
  - `NOOP`：无需变更。
- **只做条目级增量更新，禁止整体重写**（ACE 的 context collapse 教训）。

### 4.5 多索引
- **向量索引**（主力召回路）：AMU 的 content+retrieval_key+keywords+entities 拼接编码；HNSW + user_id 强制过滤。
- **BM25/全文**：专有名词、数字、代码标识符。
- **时序索引**：event_time + valid_from/to，支持时间切片。
- **轻量图谱**：实体节点 + AMU 节点 + 关系边（HippoRAG 2 式 passage 节点入图）；关系标签归一化（同义合并）；邻接表存 Postgres，PPR 纯算法实现，不依赖 Neo4j。

---

## 5. Search 管线详细设计

### 5.1 查询理解（gpt-4o-mini，单次调用）
提示词模板：`prompts/03_query_understanding.txt`。输出 JSON：
- **intent**：fact / multi-hop / temporal / preference / rule / abstention-check；
- **time_scope**：相对时间展开为绝对区间（LongMemEval time-aware query expansion，论文实测有效）；
- **entities**：命名实体列表（HippoRAG query NER，作 PPR 种子）；
- **sub_queries**：多跳题拆 2–3 个子查询（structural-memory：迭代检索一致更优，AML 延迟约束下用"单轮并行多子查询"折中）；
- **options 利用**：选择题选项并入辅助召回查询（仅召回，不生成答案）。

### 5.2 多路召回（并行）
1. **向量 top-100（主力路）**——HippoRAG 2 教训：纯图检索牺牲事实召回，向量永远第一路；
2. BM25 top-100；
3. 子查询向量召回（多跳）；
4. **图谱 PPR 扩展**：query 实体 + AMU 节点种子 → PPR 一跳邻域 AMU（HippoRAG 单步多跳）；
5. 时序切片（temporal 意图）；
6. **画像/规则直通**（preference/rule 意图时拉取该 user 全量画像与规则——体量小，Dynamic Cheatsheet 式整份直通）。
7. **经验路**（procedural 意图按任务签名召回 strategy/workflow/skill/playbook）。

### 5.3 融合、过滤与重排
- RRF 融合多路结果并去重；
- **相关性过滤**（v0.2 新增，Memory-R1 Answer Agent 思想）：gpt-4o-mini 对融合后 top-40 批量打分，剔除明显无关项——低成本防噪声；
- **治理感知排序**：同版本链只保留最新有效版本；temporal 意图明确问历史时返回指定时间切片版本；
- 重排只对 top-30 打分（256 并发下的成本控制），其余按融合分；
- **content 内嵌时间区间**（Zep Constructor 法）：返回时前缀 `[valid: 2023-05 ~ 至今]`——平台 Answer 模型直接拿到时序证据，时序题的"白捡分"手段；
- 截断 top_k=100。

### 5.4 认知安全 / 弃权
- 低置信且无高相关候选 → 返回空数组（LongMemEval abstention 维度）；
- sensitivity=sensitive 默认不参与任何召回；服务端 `AML_SENSITIVE_RECALL_ENABLED=1` 且请求 `include_sensitive=true` 时才允许；
- user_id 隔离在存储层强制。
- `DELETE /memory/{user_id}` 硬删数据库和显式开启的本地调试日志，只保留无正文删除凭证。

---

## 6. Coding Track 扩展（SWEContextBench）

复用同一内核，扩展 AMU 类型与抽取器：
- type 增加 `debug_experience | fix_pattern | workflow | repo_context`（ExpeL 成败对比提炼 + AWM 工作流归纳 + Voyager 技能库双层结构）；
- 抽取器提示词针对工程历史调优（issue/patch/test 日志）；
- BM25 与符号级检索权重上调（函数名、错误码、路径精确匹配）；
- 图谱节点扩展为文件/函数/错误三元组；
- 本地自测参考 Evo-Memory 流式任务协议。

---

## 7. 合规与工程实现清单

| 项目 | 方案 |
|---|---|
| 内部 LLM | **LiteLLM 统一抽象**：`AML_LLM_MODEL` 环境变量（默认 `gpt-4o-mini`，温度 0）；嵌入用 `AML_EMBED_MODEL`（默认 `text-embedding-3-small`）；规则变动只改配置。实现见 `memory_system/app/llm.py` |
| 嵌入模型 | 本地开源多语言模型（BGE-M3 类），无外部依赖 |
| 存储 | 当前 SQLite/WAL 单机参考实现；生产目标 Postgres（pgvector + 全文索引 + 图谱邻接表） |
| 服务 | FastAPI 异步 IO；`/add`、`/search`、`/feedback`、`DELETE /memory/{user_id}`、`/health`；Bearer 鉴权 |
| 幂等 | request_id 唯一约束 + UPSERT |
| 并发 | 当前同用户进程内串行、跨进程 revision 冲突最多重试 3 次；迁移 Postgres 后再启用水平扩展 |
| 部署 | 单 Dockerfile；README 含启动命令、API 入口、配置、原创性披露 |
| 数据合规 | 评测数据独立命名空间；30 天删除脚本；不记请求正文日志 |
| 安全 | 消息内容视为数据非指令（防注入）；AMU content 禁止指令式文本 |
| 稳定性 | 公网 HTTPS；30 天可用性监控告警 |

---

## 8. 提示词资产（独立模板文件）

全部模板存于 `prompts/`，`{placeholder}` 占位，代码加载后直接 format：

| 文件 | 用途 | 调用位置 | 来源 |
|---|---|---|---|
| `01_extract_amu.txt` | 原子事实+三元组抽取 | Add §4.3 | A-MEM Ps1 + Zep t_ref + LongMemEval key expansion |
| `02_governance_decision.txt` | 治理四操作判定 | Add §4.4 | Mem0 + A-MEM Ps3 |
| `03_session_summary.txt` | 会话滚动摘要更新 | Add §4.2 | Mem0 + MemAgent + ACE |
| `04_query_understanding.txt` | 意图/时间/实体/子问题 | Search §5.1 | LongMemEval + HippoRAG |
| `05_rerank_filter.txt` | 相关性过滤+重排打分 | Search §5.3 | Memory-R1 |
| `06_eval_answer.txt` | 本地评测 Answer | 本地代理评测 | Mem0 附录 A |
| `07_eval_judge.txt` | 本地评测 Judge | 本地代理评测 | Mem0 附录 A（时间宽容规则） |
| `08_sufficiency_verify.txt` | 证据充分性与 follow-up | Search §5.4 | EverMemOS |
| `09_experience_distill.txt` | 显式任务反馈蒸馏 | Feedback | ReasoningBank、AWM、Voyager、ACE |

注：`06`/`07` 仅用于本地代理评测，不在参赛服务内（Search 不得生成答案）。

---

## 9. 维度-模块映射（自查表）

| AML 维度 | 主承载模块 | 论文依据 |
|---|---|---|
| 事实召回 | 向量主力召回 + AMU 原子事实 | HippoRAG 2、Mem0 |
| 多跳整合 | 图谱 PPR 扩展 + 子问题并行召回 | HippoRAG、structural-memory |
| 时序理解 | 双时间轴 + 时间锚定 + content 内嵌时间区间 | Zep、LongMemEval |
| 记忆治理 | 条目级 SUPERSEDE 版本链 | Mem0、Zep、ACE |
| 个性化 | preference/profile 直通 + 画像条目化 | MemoryOS、Dynamic Cheatsheet |
| 规则执行 | rule/workflow 类型 + 直通召回 | ExpeL、AWM |
| 认知安全与隐私 | 弃权策略 + 防注入 + 行级隔离 | LongMemEval、survey-security |

---

## 10. 验证计划（提交前）

1. **本地代理评测**：LoCoMo（第一基准，AML request_id 直接出现 locomo）+ LongMemEval-S + PersonaMem；Answer/Judge 用 `prompts/05-06`（gpt-4o-mini）；按七维度出分项分数，目标 Overall ≥ 55。
2. **粒度与结构消融**：轮级/段级/事实级（SeCom 法）；向量单路 vs +图谱 vs +治理，确认各模块贡献。
3. **契约自测**：模拟平台分块、409/429 重试、错误格式注入；验证幂等与响应格式。
4. **平台 smoke**：接口变更后必跑（每小时 1 次配额）；smoke 通过后再提交 full（每 3 个月 1 次）。
5. **弃权专项**：构造记忆外问题验证空返回行为，避免噪声证据。

## 11. 里程碑

| 阶段 | 内容 | 产出 |
|---|---|---|
| M1（1 周） | AMU 模型 + Add/Search 骨架 + 幂等/鉴权/health | 可过 smoke 的服务 |
| M2（2 周） | 抽取 + 治理 + 多索引 + 多路召回 + 重排（接入 prompts/ 模板） | 完整管线，LoCoMo 跑分 |
| M3（1 周） | 时序/多跳/个性化专项 + 消融 | 分项分数报告 |
| M4 | 第二期开放（预计 2026-09-20）后：规则复核 → 申请 → smoke → full | 公榜候选结果 |

## 12. 关键风险与对策

- **gpt-4o-mini 复现一致性**：温度 0 + 固定版本号 + 记录每次调用；若第二期规则变动，改 `AML_LLM_MODEL` 一处即可。
- **full 每 3 个月 1 次**：本地代理评测充分后再提交，禁止用 full 调试。
- **Search 延迟**：重排只对 top-40；首轮含理解、重排、验证，追加轮有硬上限并跳过重复 follow-up。
- **证据伪装答案红线**：AMU content 一律陈述性事实原句；宁可少召回不可违规。
- **记忆投毒**：评测对话可能含注入文本；抽取提示词明确"内容即数据"，content 字段拒绝指令式句子。
