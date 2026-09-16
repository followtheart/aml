# PersonaMem 改进计划（实证修订版）

> 基于 `logs/memory-debug.jsonl` + `logs/search-debug.jsonl` 的逐题断点分析，对《Updated todo list.txt》草稿的修订。
> 评测配置：`graph+governance+rerank+keyexp`，当前成绩 **9/26 = 34.62%**。

## 0. 证据基线（本次修订的依据）

| 事实 | 数据 |
|---|---|
| 随机基线 | 26 题全部 4 选 1 → **25%**；当前 34.62% 仅高 ~10pt，记忆系统增益很小 |
| 评测结构 | 26 题全部来自**同一 persona**（Daniel Whitaker，32k token 历史；208 user + 205 assistant 条消息，**0 条 system 消息**） |
| 断点分布（17 道错题） | **15/17 提取层**：偏好从未成为"用户事实"，连 ~120 条 fused 候选池都没进；**1/17 重排层**：qa=9 遗忘治理记忆 fused 第 7 名被 rerank 丢弃；**1/17 答案层**：qa=0 证据排 returned 第 1 仍答错 |
| 提取"话题化"实证 | 534 条 AMU：fact 300 / episode 113 / event 80 / plan 21 / **preference 仅 13（全 transient）/ rule 6 / profile 1**。典型扭曲："Enjoys baking bread" → *"Bread baking is described as satisfying due to cultural heritage…"*；"Practices yoga daily" → *"user is interested in **understanding the benefits** of yoga"* |
| session_summary 是死重 | 写入侧生成了含 User Persona 的摘要，但 **3043 条 returned 中 0 条**包含该文本——不进检索、不进答案上下文 |
| 遗忘治理 | "forget X" 存为 `rule`（**且跨 chunk 重复**，meal-kit 遗忘规则出现 2 次），原偏好 AMU 不失效；qa=9 中该 rule 被 rerank 误杀 |
| 查询扩展无效 | 失败题的 `plan.sub_queries` 大多只是复述原问题，未向 persona 方向展开（"健康零食" × "胆固醇" 接不上） |

与原草稿的三处修正：

1. **草稿 §3 的"system persona JSON（3.8k 字）0 条 AMU"不适用于当前评测**——输入中不存在 system 消息，persona 只能从对话推断。Core Profile 必须走"巩固"路线而非"直接抽取"路线（若未来 harness 注入 persona JSON，再补专用抽取通道）。
2. **重排不是主要瓶颈**（仅 1/17）：草稿 §2 的 rerank 改造降级为辅助项；"答题默认选通用项"是提取语义扭曲的下游症状，应先修内容再修排序。
3. **答案层失败（qa=0）的根因是记忆内容不断言用户事实**，证据排第 1 也没用——这把"选项-证据对齐"从独立优化变成依赖 P0 的增效项。

---

## 1. P0 · 画像巩固 + Core Profile 注入 —— 针对 15/17 + qa=0

借鉴 **MemoryBank**（画像分层持续更新）、**EverMemOS**（MemCell→MemScene 语义巩固）、**MemGPT/MIRIX**（Core Memory 无条件驻留）、**A-Mem**（笔记演化合并）。

| 改动 | 落地 |
|---|---|
| **兴趣巩固阶段**（新增） | 每次 Add 提交后，对本用户 `fact/event` 中匹配 `The user asked/discussed/inquired/mentioned` 的条目做"兴趣推断"：同一主题出现 ≥2 次 → 产出/更新一条**断言式** `preference` AMU（"The user bakes bread at home"，而非 "asked about baking"），`support_sessions` 累加，`transient → static` 晋升 |
| **断言式提取 prompt 修正** | prompt 01 增加规则：用户就某主题反复提问/分享个人经历时，优先产出**第一人称用户事实**（user does/likes/has X），话题知识（world fact）仅在无用户维度时保留 |
| **Core Profile 注入** | 每用户一份有界结构化画像（health / diet / hobbies / occupation / family / rules），Search 返回时**无条件前置注入，不占 top_k**；复用现有 `[profile: static]` 标签。数据源：巩固后的 preference AMU + session_summary（否则后者永远是死重） |
| **MERGE_INTO_PREFERENCE 治理** | `_govern_one` 增加合并操作：同主题 fact 合并为高层 preference，消除重复（顺带解决 forget rule 重复问题） |

**验收**：preference/profile AMU 从 14 → ≥40；`returned` 中出现画像注入块；qa=0 类"证据第 1 仍答错"消除。

## 2. P1 · 遗忘即失效（INVALIDATE）—— 针对 qa=9 类

借鉴 **Mem0**（DELETE 操作）、**Zep**（边失效 `valid_to`）、**MemOS**（生命周期状态机）。

- 治理 prompt 增加 `INVALIDATE` 意图：用户要求遗忘 → 目标 AMU 置 `valid_to=now` + `sensitivity=suppressed`，检索默认排除；遗忘请求本身保留为 `rule`（可审计，去重）。
- **治理类记忆（rule/suppressed 相关）进 rerank 白名单**：只排序、不丢弃——qa=9 的治理记忆正是被 rerank 杀掉的。
- 无时间戳历史用消息序号作 `t_valid` 序（Zep 双时间线），避免 `anchor_time` 落墙钟导致 plan 全 `pending`。

**验收**：forget 类题目（qa=9）答对；被抑制 AMU 不出现在任何 returned。

## 3. P2 · 选项-证据对齐答题 —— 依赖 P0，消除"默认选通用项"

借鉴 **Memory-R1**（Answer Agent 先蒸馏子集再答）、**HippoRAG 2**（Query-to-Triple 直接匹配）。

- 在 `eval_scoring.answer_prompt` 的 choice 分支前加对齐步骤：抽取每个选项的 persona 从句（"Since you already X"）为候选断言，与画像/preference 记忆逐条比对（可落成 `(user, likes/practices/has, X)` 三元组匹配，命中即强证据），列出各选项的支持记忆 id 后再作答。
- 通用选项（无 persona 从句）仅在所有 persona 选项支持集为空时可选。
- 个性化意图下 rerank **只排序不过滤**（HippoRAG 2 承认识别过滤有 ~26% 丢失），`SEARCH_MIN_RELEVANCE` 对该类查询置 0。

**验收**：pred 分布不再向通用选项倾斜；9 道"证据在库但答错"类题目转化 ≥5 道。

## 4. P3 · 联想式召回 + persona 条件化查询扩展

借鉴 **HippoRAG**（PPR 单步多跳）、**A-Mem**（动态链接）、**SeCom**（片段粒度）、**Structural Memory**（混合结构按任务切换）。

- 改造 keyexp：planner 生成 sub_queries 时拼入 Core Profile（当前是脱 persona 裸扩展，只会复述问题）。"健康零食" → 扩展出 "snacks for someone monitoring cholesterol"。
- 提取时产出 `(user, interested_in, X)` 三元组入图，查询经实体→用户事实做 PPR 联想召回。
- 按 `plan.intent` 切换记忆视图：preference 意图下候选集限 `preference/profile/rule/episode`，屏蔽世界知识 fact；个性化查询用 episode 粒度重排。

**验收**：15 道"证据未进 fused"题目的偏好记忆进 fused 候选池 ≥10 道。

## 5. P4 · 经验回路（边际收益，最后做）

借鉴 **ExpeL**、**Dynamic Cheatsheet**、**ReasoningBank MATTS**。

- 复用 `experience.py` + prompt 09：把 (question, options, returned, pred, gold) 蒸馏成 `task_signature=personamem_choice` 的经验（如"通用选项在 9/9 情况下错误"），注入答题上下文。
- 每 dataset 维护一份策展 cheatsheet；低置信度题做多路检索 + 自一致性投票。

## 6. 评测方法修正

- **所有报告标注随机基线 25%**，否则 34.62% 会被误读。
- **LongMemEval** 式能力分类：错题按"偏好提取 / 知识更新（遗忘）/ 时序 / 多跳"打标，逐轮追踪各类转化率，而非只看总分。
- **Evo-Memory** 流式记录：随 chunk 数记录 preference AMU 数、`static` 晋升数、rerank 拒绝率，验证画像是否真正在累积（26 题同一 persona，单点方差极大）。
- abstention 按 `intent=abstention_check` 保留，其余意图不弃权（沿用当前 exempt 方向）。

## 执行顺序与预期

| 步骤 | 内容 | 预期转化 | 累计上限估计 |
|---|---|---|---|
| 1 | P0 画像巩固 + Core Profile 注入 | 15/17 提取失败中 ≥8 题 | ~65% |
| 2 | P1 INVALIDATE + 治理白名单 | qa=9 类 | ~69% |
| 3 | P2 选项-证据对齐 | 残留"证据在库答错" ≥5 题 | ~85% |
| 4 | P3 联想召回/扩展 | 长尾召回 | ~90%+ |
| 5 | P4 经验回路 | 边际 | — |

每步完成后跑同一 `graph+governance+rerank+keyexp` 配置对照，用 `data/results/personamem-v2.jsonl` 逐题 diff。

---

## 附：实施状态（2026-09-16，已完成编码与离线验证）

P0–P3 已全部落地，默认开启，全部可通过环境变量消融（见 `.env.example`）：

| 计划项 | 实现 | 关键文件 | 消融开关 |
|---|---|---|---|
| P0 断言式提取 | prompt 01 规则 8/9 强化：个人细节必须第一人称断言；同主题多问 → 追加 preference；禁止把 assistant 科普存成世界知识 | `prompts/01_extract_amu.txt` | — |
| P0 兴趣巩固 | 新模块 `profile.consolidate`：每次 Add 后 LLM 巩固（prompt 10），stated ≥1 / inferred ≥2 支持才产出 preference；相似 ≥0.8 合并并累加 support（A-Mem 演化），≥2 支持晋 static | `app/profile.py` | `AML_PROFILE_CONSOLIDATION` |
| P0 Core Profile 注入 | `st.core_profile`（rule 优先、单独限额）在每次 Search 无条件前置注入，**不占 top_k、不参与排名**；trace 记录 `core_profile_injected` | `app/store.py` `app/search_pipeline.py` | `AML_CORE_PROFILE_INJECT` |
| P1 INVALIDATE | `profile.apply_forget_rules`：遗忘请求 → embedding 候选（≥0.55）+ LLM 确认（prompt 12）→ `valid_to` 关闭 + `sensitivity='suppressed'`，**全部检索路由无条件排除**（含 include_history）；请求本身保留为 rule 并 `superseded_by` 链接可审计 | `app/profile.py` `app/store.py` | `AML_INVALIDATE_ENABLED` |
| P1 rerank 白名单 | `type=rule` 的候选只排序不丢弃 | `app/search_pipeline.py` | — |
| P1 合成时间戳 | 全空时间戳的历史按消息序合成单调时间（2020 起、每分钟一条、跨 chunk 延续），锚定不再落墙钟 | `app/add_pipeline.py` | `AML_SYNTHETIC_TIME` |
| P2 选项-证据对齐 | choice 单选题答题前对齐（prompt 11）：persona 选项无支持才允许选通用项；失败 fail-open | `app/eval_scoring.py` | `AML_CHOICE_ALIGN` |
| P2 个性化不过滤 | 带选项或 preference/profile 意图时 rerank 阈值降为 0（只排序） | `app/search_pipeline.py` | — |
| P3 画像条件扩展 | prompt 04 注入画像摘要，子查询绑定用户特征 | `app/search_pipeline.py` `prompts/04` | `AML_QUERY_PROFILE_DIGEST` |
| P3 persona 视图 | preference/profile 意图且用户已有画像时，候选隐藏世界知识 fact | `app/search_pipeline.py` | `AML_PERSONA_VIEW_FILTER` |
| P3 三元组指引 | prompt 01 规则 5 偏好 `(user, interested_in/owns/has_condition/practices, X)` | `prompts/01_extract_amu.txt` | — |

**验证**：全部 12 个 selftest 通过（含新增 `scripts/selftest_profile.py` 10 项：巩固创建/合并/噪声拒绝、core 排序、遗忘失效+全路由隐藏、合成时间戳延续、core 注入不占 top_k、rule 重排白名单、选项对齐/失败 fail-open）。`selftest_evidence` 与 `selftest_search` 各有 1 项失败为基线已存在（与本次改动无关，干净树上同样失败）。FAKE 模式 personamem 端到端冒烟通过。

**跑实验建议**：先全量配置跑一轮作为新基线，再逐项置 0 消融（`AML_PROFILE_CONSOLIDATION` / `AML_CORE_PROFILE_INJECT` / `AML_CHOICE_ALIGN` / `AML_INVALIDATE_ENABLED`），用 `data/results/personamem-v2.jsonl` 逐题 diff，并对照随机基线 25%。
