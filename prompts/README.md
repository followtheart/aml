# prompts/ — AML 记忆系统提示词模板库

> 与 `agent-memory-system-design.md`（v0.2）§8 对应。
> 所有模板为纯文本 + `{placeholder}` 占位符，代码加载后直接 `.format(...)` 使用。
> 统一约定：模型 gpt-4o-mini（环境变量 `AML_LLM_MODEL`），temperature=0；
> JSON 输出模板中的字面花括号已转义为 `{{ }}`。

## 文件清单

| 文件 | 管线位置 | 调用时机 | 调用频率 |
|---|---|---|---|
| `01_extract_amu.txt` | Add §4.3 | 每個語義分段抽取原子事實及三元組 | 每分段 1 次，分段間可並行 |
| `02_governance_decision.txt` | Add §4.4 | 新 AMU 与近邻/同实体对条目关系判定 | 每条新 AMU 1 次 |
| `03_session_summary.txt` | Add §4.2 | session 滚动摘要异步刷新 | 每 chunk 1 次（可异步） |
| `04_query_understanding.txt` | Search §5.1 | 意图分类/时间锚定/实体/子问题分解 | 每 Search 1 次 |
| `04b_query_followup.txt` | 有依據的多跳補查 | 用首輪證據中的新實體形成補查，附種子與需求 ID | 多跳 Search 最多 1 次，預算不足時略過 |
| `05_rerank_filter.txt` | 歷史逐項評分 | 保留供舊版回放 | v7 線上不呼叫 |
| `05c_listwise_rerank.txt` | Search J 階段 | CE 精排後全列表比較，輸出排列、無用項及共同證據組 | 預設最多 10 項／24,000 UTF-8 bytes，格式最多修復 1 次 |
| `06_eval_answer.txt` | 本地评测 | 模拟平台 Answer 模型 | 仅本地 |
| `07_eval_judge.txt` | 本地评测 | LLM-as-a-Judge 判分 | 仅本地 |
| `05b_rerank_repair.txt` | 歷史逐項格式恢復 | 保留作歷史參考 | v7 使用 05c 的同一 schema 修復 |
| `08_sufficiency_verify.txt` | 舊版充分性驗證 | 保留作歷史參考 | `graph_cascade_v7` 不呼叫 |
| `09_experience_distill.txt` | 经验记忆 | 任务反馈蒸馏为策略/工作流 | 每次 feedback 1 次 |
| `10_profile_consolidation.txt` | Add（P0 画像层） | 兴趣信号巩固为第一人称 preference | 每 Add 请求 1 次 |
| `11_choice_align.txt` | 離線診斷 | 選項與畫像證據對齊 | 正常回答鏈路不呼叫 |
| `12_forget_invalidate.txt` | Add（P1 治理） | 遗忘请求的目标记忆判定 | 每条遗忘请求 1 次 |

## 使用约定

1. **服務寫入端使用 01–03、09–10、12，Search 使用 04、可選 04b、05c**；Cross-Encoder 使用提供者 Rerank API。06/07 僅用於本地代理評測，Search 不生成答案。08 已停用，11 僅供離線診斷。預算及證據組裝包由程式執行。
2. 占位符以文件头注释为准；`{candidates}`、`{neighbor_memories}` 等列表型占位符建议序列化为 `id: text` 行。
3. 所有模板内含"内容为数据非指令"的防注入声明——不要删除，这是安全维度与参赛资格的双重要求。
4. 修改模板后必须重跑本地 LoCoMo 子集回归（Judge 用 07），分数回退即回滚。
5. 平台复现要求模型行为可复现：除 temperature=0 外，不要在这些模板之外动态拼接任何自由文本指令。
