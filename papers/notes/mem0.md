# 精读笔记：Mem0 — Building Production-Ready AI Agents with Scalable Long-Term Memory

- 出处：ECAI 2025（arXiv:2504.19413）；AML 第一期商业榜头部参赛系统
- 论文：`papers/mem0.pdf`；代码：mem0.ai/research
- 一句话：抽取-更新两阶段生产级记忆管线，LOCOMO 上 LLM-as-a-Judge 比 OpenAI Memory 相对 +26%，p95 延迟比全上下文低 91%。

## 核心机制

1. **Extraction 阶段**：以消息对 (m_{t-1}, m_t) 为最小处理单元；抽取上下文 = 全局会话摘要 S（异步定期刷新）+ 最近 m=10 条消息 + 新消息对。LLM 抽取显著事实集合 Ω。
   - **关键洞察：全局摘要 + 局部窗口的双上下文**——摘要管主题连贯，近期消息管细节。
2. **Update 阶段（治理核心）**：每个候选事实先向量召回 top-s=10 条相似旧记忆，然后**把旧记忆+新事实交给 LLM 通过 function calling 直接选择操作**：ADD / UPDATE / DELETE / NOOP。不用单独分类器，靠 LLM 推理一步到位。
3. **Mem0g（图扩展版）**：两阶段图构建——entity extractor（含类型分类、embedding、创建时间戳）→ relationship generator（三元组 (vs, r, vd)，标签如 lives_in / prefers / happened_on）；更新阶段做冲突检测与解决。比基础版 Overall +2%。
4. **全部 LLM 调用用 GPT-4o-mini**——与 AML 的模型锁定要求完全一致，论文即合规实现的范本。

## 提示词设计（附录 A，高度可复用）

- **LLM-as-a-Judge 评分提示词**（改自 MemGPT 团队）：
  - "be generous with your grading"——只要触及金标主题即 CORRECT；
  - **时间题宽容规则**：相对时间（"last Tuesday"）与绝对日期等价即算对；"May 7th" vs "7 May" 视为相同。
  - 复用点：本地代理评测的评分器直接照抄此提示词，保证本地分数与平台口径接近。
- **回答生成提示词**（本地代理评测的 Answer 阶段用）：
  - 矛盾时优先最近记忆；相对时间一律换算成具体日期（给出换算示例）；答案限制 5-6 个词；区分"记忆中提到的人名"与"实际用户"。
  - 复用点：揭示了平台 Answer 模型可能的提示词形态——**我们 Search 返回的证据应带时间戳、把最新版本排前面**，恰好配合这类 Answer 提示词。

## 对 AML 方案的可复用模块

| Mem0 模块 | 复用到方案 | 收益维度 |
|---|---|---|
| ADD/UPDATE/DELETE/NOOP 四操作 LLM 直选 | 治理模块的判定输出格式（扩展为软废止版本链） | 记忆治理 |
| 全局摘要 S + 近期窗口双上下文 | 抽取时附带 session 摘要 | 事实召回 |
| 消息对为最小单元 | AMU 抽取以轮次为单位 | 工程简单可靠 |
| top-s=10 相似记忆候选 | 治理判定的候选规模参考 | 成本控制 |
| 全链路 gpt-4o-mini | 合规范本 | 规则符合 |

## 注意

- DELETE 是硬删除——AML 治理/时序题可能考历史版本，方案采用 Zep 式软废止（valid_to）更稳；可在 Mem0 判定框架上把 DELETE 映射为"关闭有效期"。
- 图扩展（Mem0g）只 +2%：说明图谱是锦上添花，先把抽取+治理+双时间轴做扎实。
