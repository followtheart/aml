# 来源绑定的写入侧联想视图

这是将 `memory_connection` 机制引入旧系统的第一阶段：增加独立的写入侧检索视图，复用旧系统已有的查询规划、查询向量、精排、原文审查和证据打包。`AML_ASSOCIATION_WRITE` 与 `AML_ASSOCIATION_RECALL` 均默认关闭；生产 HTTPS 未切换到实验配置。

随后按要求增加独立的 `AML_ASSOCIATION_QUERY` 开关，支持查询侧与写入侧同时运行，仍默认关闭。下面保留写入侧说明，并补充双侧接入方式。

完整 61 题隔离诊断已完成：粗排参考覆盖 47→47，精排 31→30，主审输入 27→27（2 增、2 失），未满足继续推进条件。结果与限制见 [第一阶段报告](ASSOCIATION_VIEWS_REPORT.md)。

## 实现

Add 仍先保存原始消息、抽取和治理 AMU。原有治理完成后，在同一个私有 staged 事务内按原始来源生成视图。每个来源最多 3 个视图，每批最多 4 个来源，默认最多 16 批。超出单来源 6,000 UTF-8 bytes 的消息不会被截断生成，日志记录跳过数量；原消息照常保存。超出批次额度的来源记录为 deferred。

每条视图必须有同一来源中唯一、逐字匹配的非空锚点。无效视图逐条拒绝，不连带丢弃同批合格视图。模型或 embedding 批次失败时，该批不新增视图；原始记忆仍按原流程提交。Fake 模式不生成声称具有语义价值的视图。

`association_views` 独立保存来源 request/message ID、来源内容及角色等属性的指纹、AMU ID 与版本、检索文本、原文锚点、向量空间、生成器和物化时间。视图不进入 AMU 事实、画像、三元组、来源正文、最终证据或主审提示词。

Search 使用原有查询向量扫描当前用户的视图。打分前联接来源、当前来源绑定和完全匹配的 AMU 版本，执行历史、敏感性、撤回、suppressed 和 stale 过滤；历史快照还排除未来才物化的视图。来源指纹和向量维度会再次校验。对一个 AMU 取最强视图分数，对一个原始来源最多提供一个候选席位，默认最多 12 个候选。

图融合取原有语义分数与折扣后联想分数的最大值，默认折扣为 0.8，不累加视图投票，不降低原来的单候选语义分数。增加候选仍可能改变图传播和有限候选池中的排序，因此不承诺原结果全部保留。后续 CE 与原文审查保持原实现；联想相似不被标为事实支持，也不直接满足查询覆盖义务。

版本变化后旧绑定失效，新的 Add 可以为其新来源建立新绑定。当前不自动重建所有历史来源的旧版本视图。DELETE 清理该用户的全部视图；并发删除或写入继续由旧系统 epoch/revision 检查阻止过期发布。

## 配置

| 环境变量 | 默认值 | 用途 |
|---|---:|---|
| `AML_ASSOCIATION_WRITE` | `0` | 新 Add 生成视图 |
| `AML_ASSOCIATION_RECALL` | `0` | Search 增加视图召回通道 |
| `AML_ASSOCIATION_QUERY` | `0` | Search 生成查询联想，最多一次模型调用 |
| `AML_ASSOCIATION_QUERY_VIEWS` | `3` | 查询联想总数，范围 1–3 |
| `AML_ASSOCIATION_QUERY_SECONDS` | `8` | 联想生成和向量化合计时限 |
| `AML_ASSOCIATION_QUERY_BYTES` | `16000` | 原问题与选项输入字节上限 |
| `AML_ASSOCIATION_VIEWS_PER_SOURCE` | `3` | 每条来源最多视图数，范围 1–4 |
| `AML_ASSOCIATION_RECALL_LIMIT` | `12` | 独立来源候选上限，范围 1–24 |
| `AML_ASSOCIATION_SOURCE_BYTES` | `6000` | 单条生成输入上限，最大 12000 bytes |
| `AML_ASSOCIATION_MAX_BATCHES` | `16` | 每次索引调用的批次数上限 |
| `AML_ASSOCIATION_GENERATION_SECONDS` | `30` | 单批生成时间上限 |
| `AML_ASSOCIATION_MIN_COSINE` | `0.3` | 候选相似度下限，尚未校准 |
| `AML_ASSOCIATION_SCORE_WEIGHT` | `0.8` | 融合分数折扣，尚未校准 |

开启写入开关不会自动回填旧记忆，也不会重做幂等请求。关闭召回开关后视图不参与 Search。开关和参数均进入运行版本元数据；开启召回时，`coverage_manifest.association_recall` 报告扫描、拒绝和候选数量。

## 验证与诊断

```bash
PYTHONDONTWRITEBYTECODE=1 memory_system/.venv/bin/python \
  memory_system/scripts/run_selftests.py selftest_association_views.py
```

新增测试覆盖逐字来源、单条拒绝、事务回滚、跨用户隔离、并发删除、版本失效、敏感和历史过滤、来源重复、embedding 空间、旧分数保护，以及 Add 幂等和召回开关。

`scripts/diagnose_association_views.py` 对已完成的 HTTPS 基线执行三阶段诊断：

1. `--phase index`：只读复制历史数据库，在新目录的副本上生成索引；核对原始消息、AMU、来源绑定、三元组、请求和版本历史的内容指纹未变。
2. `--phase replay`：固定历史查询规划和原召回通道，控制组不追加视图，实验组追加视图。共用查询向量，交替执行顺序；使用真实 CE 和相同的初始排名预算，在主审调用前停止。索引生成与排名诊断分别冻结脚本和文件指纹，生成依赖不得漂移。
3. `--phase analyze`：单独加载参考标注，计算粗排、精排和主审输入的来源载体覆盖及实际排名文本覆盖。参考答案不参与索引、召回或 CE。

示例需提前设置真实供应商环境；输出目录必须是新的隔离目录：

```bash
memory_system/.venv/bin/python memory_system/scripts/diagnose_association_views.py \
  --baseline memory_system/runs/https-validation-baseline-20261024 \
  --output memory_system/runs/association-example --phase index
# 同一 baseline/output，依次改用 --phase replay、--phase analyze。
```

该诊断不重新执行 Add 抽取，不调用主审或回答器，不是新的 HTTPS 成功率或在线延迟评测。两个分支的 CE 会重新评分，存在供应商波动；完整原文命中也不等于答案得到语义支持。进入下一阶段的初步条件是在主审输入增加参考覆盖且不丢失控制组参考，同时无 CE 故障；通过后仍需完整回答评测和回归集验证。

## 查询侧与双侧

完整四组结果、预算敏感性复核与当前采用决策见 [双侧诊断报告](DUAL_ASSOCIATION_REPORT.md)。

随后完成的全流程 HTTPS 答题成绩为基线 17/61、双侧 13/61；见 [HTTPS 配对报告](DUAL_ASSOCIATION_HTTPS_REPORT.md)。该结果与下述候选诊断分别报告，新开关继续默认关闭。

`app/query_associations.py` 为原问题和选项生成最多 3 个联想检索视图，每条必须绑定问题或选项中的唯一逐字锚点。选项只是待验证假设；查询视图不写入数据库，不增加事实、覆盖义务或主审证据，不递归扩展。

查询侧生成只执行一次，时间与供应商调用计入 Search 预算，并为后续主审保留一个调用及时间余量。输入超限、预算不足、模型或 embedding 失败时记录降级原因，保留旧检索路径；Fake 模式不生成语义视图。

三类新增连接共用一个最多 12 个独立来源的候选池：

- 既有查询向量 → 写入侧联想视图。
- 新查询联想向量 → 既有 AMU 向量。
- 新查询联想向量 → 写入侧联想视图。

候选按最大相似度合并，同来源不重复占席。双侧不把新增候选上限翻倍。旧系统的向量、全文和原文来源路径、图和场景扩展种子、问题覆盖要求继续保留；最终仍使用旧系统 CE 和原文审查。

| 组别 | `AML_ASSOCIATION_RECALL` | `AML_ASSOCIATION_QUERY` |
|---|---:|---:|
| C00：保留旧系统能力，不新增联想 | 0 | 0 |
| C10：仅新增写入侧 | 1 | 0 |
| C01：仅新增查询侧 | 0 | 1 |
| C11：同时新增双侧 | 1 | 1 |

需要为新 Add 建立写入索引时另设 `AML_ASSOCIATION_WRITE=1`；读开关不会自动回填旧数据。查询侧状态通过 `coverage_manifest.query_association` 返回。

四组诊断工具为 `scripts/diagnose_dual_association.py`。它复制已完成的写入侧隔离索引，固定每题查询联想和向量，对四组候选中的每个独立原文文档只做一次真实 CE 评分，再按共同字节／调用预算执行排名并在主审前停止。C01/C11 另计生成成本，保存全部查询视图、向量、CE 分数、原文输入及代码快照。这是候选运输诊断，不是完整答题或线上延迟评测。

```bash
memory_system/.venv/bin/python memory_system/scripts/diagnose_dual_association.py \
  --baseline memory_system/runs/https-validation-baseline-20261024 \
  --index-run memory_system/runs/association-views-v2-20261024 \
  --output memory_system/runs/dual-association-example --phase run
# 同样参数改用 --phase analyze，仅在完整运行结束后读取参考标注。
```

互补前提核验保持原实现，没有增加额外的严格删除环节。
