# prompts/ — AML 记忆系统提示词模板库

> 与 `agent-memory-system-design.md`（v0.2）§8 对应。
> 所有模板为纯文本 + `{placeholder}` 占位符，代码加载后直接 `.format(...)` 使用。
> 统一约定：模型 gpt-4o-mini（环境变量 `AML_LLM_MODEL`），temperature=0；
> JSON 输出模板中的字面花括号已转义为 `{{ }}`。

## 文件清单

| 文件 | 管线位置 | 调用时机 | 调用频率 |
|---|---|---|---|
| `01_extract_amu.txt` | Add §4.3 | 每个 chunk 抽取原子事实+三元组 | 每 Add 请求 1 次 |
| `02_governance_decision.txt` | Add §4.4 | 新 AMU 与近邻/同实体对条目关系判定 | 每条新 AMU 1 次 |
| `03_session_summary.txt` | Add §4.2 | session 滚动摘要异步刷新 | 每 chunk 1 次（可异步） |
| `04_query_understanding.txt` | Search §5.1 | 意图分类/时间锚定/实体/子问题分解 | 每 Search 1 次 |
| `05_rerank_filter.txt` | Search §5.3 | 融合候选头部（默认 40 条）相关性打分 | 每 Search 每轮 1 次 |
| `06_eval_answer.txt` | 本地评测 | 模拟平台 Answer 模型 | 仅本地 |
| `07_eval_judge.txt` | 本地评测 | LLM-as-a-Judge 判分 | 仅本地 |
| `08_sufficiency_verify.txt` | Search（ULM §5.5） | 证据充分性验证 + 缺口查询改写 + 弃权置信 | 每 Search 每轮 1 次（默认 ≤2 轮） |
| `09_experience_distill.txt` | 经验记忆 | 任务反馈蒸馏为策略/工作流 | 每次 feedback 1 次 |
| `10_profile_consolidation.txt` | Add（P0 画像层） | 兴趣信号巩固为第一人称 preference | 每 Add 请求 1 次 |
| `11_choice_align.txt` | 本地评测（P2） | choice 选项与画像证据对齐 | 每 choice 题 1 次 |
| `12_forget_invalidate.txt` | Add（P1 治理） | 遗忘请求的目标记忆判定 | 每条遗忘请求 1 次 |

## 使用约定

1. **参赛服务只加载 01–05、08**；06/07 仅用于本地代理评测管线（Search 不得生成答案，红线）。
2. 占位符以文件头注释为准；`{candidates}`、`{neighbor_memories}` 等列表型占位符建议序列化为 `id: text` 行。
3. 所有模板内含"内容为数据非指令"的防注入声明——不要删除，这是安全维度与参赛资格的双重要求。
4. 修改模板后必须重跑本地 LoCoMo 子集回归（Judge 用 07），分数回退即回滚。
5. 平台复现要求模型行为可复现：除 temperature=0 外，不要在这些模板之外动态拼接任何自由文本指令。
