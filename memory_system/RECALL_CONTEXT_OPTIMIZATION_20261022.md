# 召回优化实施与验证（2026-10-22）

结论：三项改动已实现，能提高上游候选覆盖，但当前真实对照未证明最终证据覆盖提升。
三项开关默认均关闭；不能宣称本轮已改善最终召回。原实验库和运行中服务均未修改。

## 实现

1. `RECALL_TOPIC_ANCHORS`：保留原问题、个人前提和分解查询，另补充最多 4 条、每条至多
   240 字符的字面选项主题查询。保留通用建议的空槽位；主题查询不形成事实或新的覆盖义务。
   解决身份前提可能取代活动主题的问题，无新增查询理解模型调用。
2. `SOURCE_RECALL_DIVERSITY`：在每查询原文 FTS 截断之前，优先分配独立原始来源；
   独立来源不足再填充同来源 AMU，所有资格、历史、敏感信息和遗忘过滤仍执行。
   这是来源载体配额优化，不是新增独立原文向量索引或多语言句群索引。
3. `LISTWISE_CONTEXT_LABEL`：keyed 输出增加 `context`，明确表示有助于回答的背景而非
   个人事实支持。按模型优先级排序，仍受原候选和证据预算限制，不提升事实置信度。
   trace 记录逐候选 `candidate_relevance`，便于单独审计上下文的去留。

新增离线 probe、真实冻结回放和独立漏斗审计脚本。所有开关及策略版本记录在运行元数据中。

## 验证设计

- 来源：`runs/https-manual-20261021-161313-1262163`，56 题、两会话。
- SQLite backup 复制已提交的原实验状态到新目录；查询规划冻结，沿用原实验有效配置。
- 真实回放重新执行 embedding、召回、CE、listwise、恢复和完整 reader；不重新入库。
- 每组并发 2。三组共 168 次搜索和答题，均无搜索异常或 reader 错误。
- 单轮随机模型输出、并发负载、不同执行时间窗口限制了结论；不是新 HTTPS 分数、正式榜单
  或显著性证据。搜索时间不包括已冻结的查询规划，不能直接比较历史完整请求耗时。
- 参考原文按 CSV 行号、问题、答案和历史身份校验；相同问句不能作为 gold 身份。
  初版回放 scorer 使用问句映射，影响三道重复问句的覆盖统计，未影响搜索或答题。
  已修复 scorer，并从保存的 trace 离线重算三组全部覆盖；以 `coverage-audit.json` 为准。
  原始结果保留；派生 summary 的覆盖字段更新，并保留更新前副本。

## 离线词法／原文召回消融

固定原始数据库和规划，不调用模型，统计 52 道存在参考原文题的载体命中。

| 策略 | 命中题数 | 新增命中 | 新增遗漏 |
|---|---:|---|---|
| 原始策略 | 43/52 | — | — |
| 仅主题查询 | 45/52 | Q26、31、47 | Q38 |
| 仅来源配额 | 47/52 | Q5、26、31、46 | 无 |
| 两者组合 | 48/52 | Q5、10、26、31、46、47 | Q38 |

平均查询数从 6.36 增到 9.91（仅主题或组合）。这衡量来源关联，不等于可见文本、语义支持或
最终答案；没有将参考 ID 用于运行时选择。

## 真实搜索及答题对照

“覆盖题数”指至少一个完整参考用户回合可见；52 道题的原文存在，其余 4 道单列。
prepared 阶段使用准备后的证据单元，不是未处理的原始召回集合。

| 指标 | 原始策略 | 仅来源配额 | 三项组合 |
|---|---:|---:|---:|
| prepared 覆盖题数 | 48 | 51 | 52 |
| coarse 覆盖题数 | 47 | 49 | 50 |
| fine 覆盖题数 | 45 | 46 | 44 |
| listwise 输入覆盖题数 | 42 | 43 | 42 |
| listwise 选择／最终 packet 覆盖题数 | 38 | 35 | 36 |
| 最终覆盖／原文存在题数 | 73.1% | 67.3% | 69.2% |
| 最终完整参考回合数 | 56 | 55 | 56 |
| 答对题数 | 33/56 | 32/56 | 34/56 |
| 搜索降级次数 | 26 | 27 | 17 |
| 搜索 provider 调用总数 | 412 | 423 | 396 |
| 搜索 token 总数 | 1,899,273 | 2,017,054 | 1,842,770 |
| 搜索耗时中位数（秒） | 19.52 | 19.05 | 15.31 |
| 搜索耗时 P95（秒） | 34.16 | 34.33 | 26.06 |

相对本次原始策略，三项组合答对修复 6 题、退化 5 题；仅来源配额修复 4 题、退化 5 题。
不能只拿组合 34/56 对比历史 30/56 宣称净增 4 题：同条件原始策略重跑也达到 33/56。

组合的主要新损失在 coarse→fine（50→44）及 listwise 选择（42→36）。来源配额单项
在前几个阶段有收益，但 listwise 仍从 43 降到 35。最终打包没有新增题级损失。

因此保留三项为可选实验，不默认开启。下一项改动应固定 prepared 候选及 CE 分数，
评估 fine/listwise 的独立来源与相关上下文保留；不继续无差别扩大召回，也不把用户问句、
改写材料或 assistant 文本强行认定为个人事实。

## 测试

以下 10 组、183 项回归通过：`selftest_recall_context`（8）、`selftest_candidate_noise`（17）、
`selftest_recall_bounds`（17）、`selftest_listwise_contract`（19）、`selftest_listwise_recovery`（47）、
`selftest_embedded_role_queries`（7）、`selftest_rerank_coverage`（33）、`selftest_single_pass`（8）、
`selftest_search_storage`（24）、`selftest_listwise_sources`（3）。额外的 HTTPS transport 脚本通过。

新增测试覆盖主题遗漏、条件和否定保留、查询上限、无事实提升、重复来源挤占、敏感信息／历史
过滤、上下文排序及重复问句的参考身份。修正 single_pass 中过时的 permutation mock、
top_k 包含约束项的断言，以及字面警示大小写断言。

额外运行 `selftest_search_repairs.py` 有 2 项失败：
`test_failed_batch_keeps_unknown_evidence_when_other_batch_is_zero` 和
`test_listwise_reorders_raw_ce_logits_without_resurrecting_rejected_evidence`。
关闭全部新开关的控制运行同样失败；两项使用旧排列输出／删除假设，不作为本次优化已通过的证据。
其余 17 项通过。未宣称整个仓库测试全部通过。

## 产物

- `runs/recall-context-20261022/lexical-ablation-verified.json`
- `runs/recall-context-20261022/baseline/coverage-audit.json`
- `runs/recall-context-20261022/diversity/coverage-audit.json`
- `runs/recall-context-20261022/optimized/coverage-audit.json`
- 各组目录保留数据库副本、有效配置、逐题结果和完整 search trace。

复现命令与开关见 README 的“召回主题、来源配额与上下文标签”。本次未提交 Git commit。
