# PersonaMem 32k 漏斗分析：4/26（15.38%）

分析日期：2026-09-24。本次重新读取当前三份日志，使用 prepared dataset 对齐题目与 Gold，并与上一轮保存的审计结果比较。仅进行离线审计和纯函数重放；未修改业务代码、原始日志、模型或评测结果，未调用外部模型。

**主要瓶颈仍在答案前提抽取、语义校验和准入。得分从 1/26 升至 4/26，但 support 抽取最终失败仍为 17/26，空答仍有 13/26。新增答对 Q1、Q15、Q16；这些变化不能证明管线问题已经修复。** 本轮还发现两个普通润色请求被写成 rule，挤占受保护规则名额。

复算脚本：[audit_personamem_20260924.py](scripts/audit_personamem_20260924.py)。机器可读计数、SHA-256、逐题原句流转及上轮比较：[audit.json](runs/personamem-20260924-audit/audit.json)。上一轮报告：[1/26 分析](PERSONAMEM_20260923_FUNNEL_ANALYSIS.md)。

```bash
python3 memory_system/scripts/audit_personamem_20260924.py
```

**样本与版本检查**

| 项目 | 核对结果 |
|---|---|
| 数据范围 | 一个对话 `911d1d0140349ab4dd01`，26 道题；不是 prepared 文件全部 200 个对话 |
| 原始消息 | 189 条，95 user／94 assistant；角色、正文逐条匹配 prepared sessions |
| Add | 14 次提交，163 个快照行／唯一记忆 ID；189 条消息都有记忆来源关联 |
| 结果对齐 | 26/26 的 search ID、用户、问题、选项、packet hash 匹配；Gold 字母重算分数一致 |
| 搜索／答案策略 | `graph_cascade_v11-option-coverage`／`verified-source-choice-v5-semantic-witness` |
| 模型 | `gpt-4o-mini`，CE 为 `qwen3-rerank`，非 fake |
| 代码与提示 | 65 个文件 SHA-256 与本轮结果记录完全一致 |
| 指纹 | `a98be4ff4222d8e425fa71efcddc9cb19ab88305b1d42df672e9c522f31b7ffe`，与上一轮对话中读取的指纹相同 |

prepared 输入 SHA-256 未变，但三份运行日志均已更新，记忆数从 155 变为 163。因此不是固定记忆／固定 packet 下的答案端单变量实验，不能将 +3 分归因于某项代码修复。这里是本地代理评测，不是官方 AML 排行榜分数；每类别只有 1–5 题，也不足以评判领域能力。

**漏斗一：从消息到最终证据包**

候选阶段按 26 题累计出现次数计数，不是去重记忆数量，也不是 Gold 证据数量。

```text
189 条消息 → 14 次 Add → 163 个日志可见记忆 ID
                            ↓
                       融合候选 1,948
                       ├─ rule 208 → 保留 78 ─────────┐
                       └─ 普通候选 1,740              │
                            → coarse 1,300           │
                            → CE 全部评分 1,300      │
                            → fine 312               │
                            → listwise 输入 248      │
                            → listwise 输出 245 ─────┤
                                                    ↓
                                       选择 323 → 整项入包 323
```

| 环节 | 损失／状态 | 解读 |
|---|---|---|
| 规则筛选 | 208 → 78，`rule_limit` 130 次 | 每题 8 条规则只保留 3 条 |
| coarse | 1,740 → 1,300，删 440 次 | 每题最多 50 条普通候选 |
| CE | 1,300/1,300 评分，26/26 ok | 无旧版本报告中的 CE 大面积预算失败 |
| fine | 1,300 → 312，删 988 次 | 每题 50 → 12，是最大数量裁剪点，不等于 988 个错误 |
| listwise 输入 | 312 → 248 | 数量限制 52 次、prompt budget 12 次 |
| listwise 输出 | 248 → 245 | irrelevant 3 次；23 次 ok、2 次 recovered、1 次 fallback |
| 整项装包 | 323 → 323 | manifest 的 omitted、unit_omitted 均为 0 |
| 包内来源正文 | unit_budget 54 次、duplicate 116 次 | 是来源引用出现次数，不等同于独立事实丢失；重复正文可能保留在别处 |

只有 Q5、Q18 的 `search_degraded=true`：Q5 是 listwise recovery 的 `ReviewIncomplete`，Q18 是 listwise 两次失败后的回退。这两个题号与上一轮不同。大部分题目不是因为搜索调用失败。

最终包平均 15,448.96、最大 20,203 UTF-8 bytes，预算 32,000，使用的是字节保守上界，不是实际 tokenizer token 数。最终包未满不代表更早的 fine、listwise 或每项 6,000 bytes 的正文预算没有裁剪。

**写入层的新问题：普通请求被当作规则保护**

本轮 163 条记忆类型为：episode 57、preference 36、profile 27、fact 19、event 10、rule 8、plan 6。来源关联完整只证明原始消息可追踪，不证明事实分类、个人归属和治理正确；快照也不能直接当最终有效数据库全量。

8 条 rule 中，6 条是遗忘请求，另两条内容是：

- `amu_66ef6c20ae3c4754`：用户请求把一段 note 润色得清楚、专业。
- `amu_cf7efe2b71144309`：用户请求润色一段 professional bio。

这两条不是持续约束，却分别占用了 9 次和 8 次规则保留位，共 **17/78 = 21.79%** 的规则旁路输出。它们会挤占仅有的三个规则名额；但没有反事实重放证明因此具体丢了几分，不能把 17 次直接算成失分题数。建议 rule 写入需要明确约束语义，搜索保护也不应只依赖类型标签。

**漏斗二：从证据包到得分**

```text
26 题都有证据返回
  ├─ 13 题：最终 eligible 为空 → ValueError → 空预测 → 0 分
  └─ 13 题：输出有效字母
       ├─ 5 题：Gold 已被排除 → 0 分
       └─ 8 题：Gold 进入可选集合
            ├─ 4 题：选错 → 0 分
            └─ 4 题：答对 → 4 分
```

| 指标 | 上轮 | 本轮 |
|---|---:|---:|
| 正确 | 1/26，3.85% | 4/26，15.38% |
| 空答 | 14/26，53.85% | 13/26，50.00% |
| 有答案 | 12/26 | 13/26 |
| 已作答但 Gold 被排除 | 7 | 5 |
| Gold 可选但选错 | 4 | 4 |
| Gold 准入 | 5/26，19.23% | 8/26，30.77% |
| Gold 准入后的命中 | 1/5，20% | 4/8，50% |
| support 最终失败回退 | 17/26 | 17/26 |
| witness 引文不在原文 | 24 次 | 35 次 |

本轮 Gold 可选题为 Q1、Q2、Q9、Q14、Q15、Q16、Q17、Q24；其中 Q1、Q9、Q15、Q16 答对。有答案题准确率为 4/13 = 30.77%。

这是互斥的末端出口，不是唯一根因：例如 Q19 同时有 generic 校验否决和遗忘约束误杀，Q21 同时有来源损失和 support 回退。

**新增 3 分的真实路径**

| 题目 | 上轮 → 本轮 | 机制变化与限制 |
|---|---|---|
| Q1 Communication | 空答 → D，正确 | 两轮都发生 support 回退；本轮四项都被当 generic 放行，最终选择 D。不是 support 已修好 |
| Q15 Wellbeing | A → B，正确 | 本轮正常抽取了腿伤影响骑行的核心前提，B 为 partial；其他项 unsupported，B 唯一准入。部分建议仍被误抽成个人事实 |
| Q16 Health | D → A，正确 | A 的哮喘核心两轮都验证通过，A 仍为 partial；本轮 B/D 也变 partial，A 才进入同一层比较并被选中。支持度分层先于相关性的策略没有改变 |

上一轮唯一正确的 Q9 保持正确。Q6 从错误字母变成空答、Q19 也从错误字母变成空答；另一些题从空答变成了错误字母。准确率上升与链路稳定性改善不能划等号。

**答案侧最严重的问题：17 题回退清空了个人证据**

support 首次成功 **7/26**，首次 invalid **19/26**；19 次 repair 仅 **2 次成功**，最终仍是 9 次结构可接受、17 次 fallback。17 次回退中：11 次空答、5 次选错、1 次正确（Q1）。

[answer_choice.py](app/answer_choice.py) 的 `_answer` 在抽取重试失败后将所有选项设为 `kind=generic, claims=[]`。后续 `entailment_checks` 只从 claims 生成引用，因此此前可见的用户原文不会被带进这些选项的 entailment 检查。离线重放确认：**17 个回退案例均生成四个 sources 和 context_neighbors 为空的检查；26/26 的最终 eligible 集合可由当前纯函数复现。**

本轮 witness 为 6 次 ok、14 次 partial、6 次 fallback；共 35 条引文被 `quote_not_in_source` 拒绝。不能为提高覆盖率取消原文引用验证，应修复引文生成和局部恢复。

日志仍只有失败类型和 prompt hash，没有无效响应、具体字段路径、异常消息和 finish reason。因此无法确定 17 次失败分别由 JSON 格式、字段类型、claim 边界、长度截断还是 provider 行为导致；ValueError 也可能由本地语义结构检查触发，不能统称“JSON 解析失败”。

**调用 ok 也可能已经丢失 Gold：Q6 是本轮最清楚的例子**

Q6 的 Gold C 有完整花粉不适原文，答案目录为 `s9`；核心 claim `spring pollen tends to set off your seasonal allergies` 得到 supported。

但同时抽取了早晚安排活动、戴眼镜、回家淋浴等新建议，且其中一个 claim 是改写的 `you can wear wraparound sunglasses ...`，不是选项原文，出现 `claim_not_in_option`。C 的 option validation_errors 因此非空，被整体判为 unsupported，最终 eligible 为空。

代码中的 `validate_support` 只在某些结构错误或 **proposed_status=supported 且 claim_not_in_option** 时要求重试；本题这些不匹配 claim 被提议为 unsupported，带错结果仍可被记录成 support `ok`。后续 `_option_status` 对 option errors 一票否决。于是“抽取调用成功”与“可用、语义正确的前提集合”脱节。

本轮 13 个空答中，12 个四项 entailment 全 false；另一个就是带结构错误的 Q6。Q7/B、Q11/A、Q12/B 等通用建议仍被当成必须有历史证明。Q10/C 的设计师腰带、运动鞋和包等建议被抽为个人事实，Gold 直接排除；Q22/C 的 `golden era of cinema` 被抽出却未引用可见的老电影话题来源。

反方向也有漏放：Q1/Q2/Q14/Q17 回退后四项全按 generic 通过，部分选项明确包含个人前提；Q24/B 的“已有充足香料库存”在空引用回退后通过 generic，而原始历史并未明确证明库存。整体表现是语义判断不稳定，不是统一过严或统一过松。

**有来源却未采用，以及宽泛人设挤掉相关证据**

来源 S8:7 表示 Add chunk 8 的 message_index 7；Q 直接对应 qa_id，均从 0 开始。

| 题目 | 可见证据／阶段 | 本轮问题 |
|---|---|---|
| Q3，Gold C | S8:7 冲浪自述已在 `s1` | support 回退后 C 排除，只剩通用 B |
| Q8，Gold D | persona 的 Facebook family updates 在 `s1` | support 回退，另有旅行照片约束误杀 |
| Q13，Gold D | S11:3 动漫政治／历史主题提问在 `s15` | D 的复合兴趣前提未获支持；“Kansas-based KU fan”的 C supported，优先层独占准入 |
| Q14，Gold A | S2:12 家庭漫画／科幻展会经历在 `s9` | 上轮在 fine 丢失，本轮已经到达，但回退后仍选 B；不能沿用上轮“证据未入包”的归因 |
| Q17，Gold B | 汽车／混动相关问题在 `s5`、`s9` | 上轮 listwise 流失，本轮已可见，仍选 D；这些问题也不能直接证明偏好美国制造 |
| Q25，Gold A | S5:5 胆固醇变化提问在 `s6` | support 回退，Gold 排除，空答；原文关注足以进入兴趣／监测层审查，不应要求诊断 |

Q13 表明 supported 优先于 partial／generic 的准入策略可能把“容易证明、但与问题关系弱”的 persona 当作决定性证据。Q16 本次选对只是同层候选变化，不消除这种结构性风险。

**遗忘约束范围误扩仍在**

Q8/D、Q19/B 仍被 `Please forget that I take photos during family trips.` 拦截。一个是使用 Facebook 联系亲友、分享照片的新建议，另一个是摆放家庭照片与纪念品的通用装饰建议。遗忘个人历史不应泛化为禁止所有照片相关建议。

本轮 Q19/B 还同时被 generic entailment 否决，所以单改 blocked 判定也不足以恢复作答。Q9 的三项约束排除仍带来正确结果，应保留精确禁止范围，不应直接关闭所有治理。当前 prompt 的“不可推荐该活动”政策与用户“忘记我的某项事实”之间需要明确区分。

**上游损失：必须查最终可见原句，而非单个记忆 ID**

1. **依赖展开可救回看似丢失的来源。** Q0 的直接晨间记忆 ID 没进 fine／包，但工作疲劳记忆通过 `[support: ...]` 携带了晨间原文，答案 `s2` 可见。Q6 的直接花粉记忆未进入 listwise，哮喘记忆的依赖展开仍携带花粉来源到 `s9`。仅交叉比对原始 AMU ID 会错误归为召回／装包失败。脚本因此同时记录直接 ID、准备后的正文载体和最终答案目录原句。
2. **Q18 是相反情况：ID 已到，用户原句没到。** 包含 S4:7 的记忆入包，但 `If I have a tiny garden ...` 被标记 unit_budget 隐藏。不能将该条件句当拥有花园的 Gold 事实，但其流转确实丢失了用户上下文。
3. **Q21 仍有多处损失。** S3:19 去中心化资产、S6:8 layer-2 提问在 fine 被裁；含 S2:16 的记忆虽然入包，稳定币用户提问仍因 unit_budget 无正文；S2:18 CBDC 后续问题在 `s13` 保留。因此是个人兴趣证据被削弱，不能称“完全没召回金融”。本轮再叠加 support fallback，最终空答。
4. **Q10、Q20 存在部分 fine 裁剪。** 奢侈品牌提问 S11:11、欧洲 LP 问题 S4:9 未过 fine；Q20 的另一条欧洲唱片问题 S6:12 仍在 `s7`。数量删除并不自动意味着 Gold 必须由此丢失。

本次没有用 Gold 词相似度最高的单条记忆冒充“真实支持证据”。来源探针只验证传输与可见性，不自动判定其语义支持强度；也没有足够逐题人工 Gold 证据集来报告严格的 Gold Recall@K。

**逐题末端分类与优先解释**

∅ 表示空答案；“Gold 排除”是日志事实，不表示所有排除均不合理。

| Q | 分类 | Gold → 预测 | 出口 | 本轮关键观察 |
|---:|---|---|---|---|
| 0 | Health | A → ∅ | 空答 | 晨间原文经依赖可见；support 回退；Gold 活动组合／频率偏强 |
| 1 | Communication | D → D | 正确 | 回退后全部 generic 放行，选择 D |
| 2 | Health | D → B | 可选但选错 | 回退后全部 generic 放行 |
| 3 | Travel | C → B | Gold 排除 | 冲浪原文可见；回退 |
| 4 | Food | D → ∅ | 空答 | 回退；提问不必然证明已有烘焙习惯 |
| 5 | Health | A → B | Gold 排除 | 搜索部分降级；旧运动伤致腰痛缺少完整来源 |
| 6 | Outdoors | C → ∅ | 空答 | 核心花粉事实 supported，但建议误抽／claim_not_in_option 导致整项失效 |
| 7 | Home | B → ∅ | 空答 | 回退；通用 Gold 被否决 |
| 8 | Communication | D → ∅ | 空答 | Facebook 来源可见；回退与照片约束误杀 |
| 9 | Events | A → A | 正确 | 通用 A 准入，其他三项被约束阻止 |
| 10 | Fashion | C → B | Gold 排除 | 新建议被误抽为事实；品牌来源在 fine 丢失 |
| 11 | Motivation | A → ∅ | 空答 | 回退；通用 Gold 被否决，Marcus 是第三人 |
| 12 | Health | B → ∅ | 空答 | 回退；通用 Gold 被否决，Claire 段落非用户经历 |
| 13 | Entertainment | D → C | Gold 排除 | 动漫提问可见；KU fan 人设独占 supported 层 |
| 14 | Travel | A → B | 可选但选错 | 家庭展会来源本轮已到，仍回退选错 |
| 15 | Wellbeing | B → B | 正确 | 骑行／腿伤核心支持，B 为唯一 partial |
| 16 | Health | A → A | 正确 | A/B/D 同为 partial，本轮获得比较机会 |
| 17 | Automotive | B → D | 可选但选错 | 汽车来源本轮已到；回退，Gold 偏好依据需审查 |
| 18 | Outdoors | B → ∅ | 空答 | listwise 回退；用户花园条件句正文隐藏；support 回退 |
| 19 | Decor | B → ∅ | 空答 | generic 否决与照片约束误杀叠加 |
| 20 | Shopping | B → ∅ | 空答 | 黑胶来源部分可见；support 回退 |
| 21 | Finance | C → ∅ | 空答 | fine 与正文预算削弱证据，support 回退 |
| 22 | Entertainment | C → D | Gold 排除 | 老电影提问可见，抽取却未给 Gold 引用 |
| 23 | Outdoors | A → ∅ | 空答 | 回退；湖水问题不直接证明游泳爱好 |
| 24 | Food | B → D | 可选但选错 | 回退后无来源的库存前提被当 generic 放行 |
| 25 | Food | A → ∅ | 空答 | 胆固醇原文已到，support 回退 |

**证据充分性的独立边界**

Q0 的“每天瑜伽且冥想”、Q4 的“已经喜欢在家烤面包”、Q5 的“旧运动伤造成轻微腰痛”、Q17 的“偏好美国制造汽车”、Q18 的“已有蔬菜香草花园”、Q24 的“已有充足香料库存”都比部分可见原文更强。来源包括一般提问、条件句和不完整描述，不能把所有 Gold 未支持归因于系统忘记了事实。Q20、Q23 的具体偏好也应人工复核。

这不代表判定官方 Gold 错误；需要结合历史转换和任务对偏好推断的要求审查。不能把标准答案写回记忆或取消引用校验来消除这类差异，也不能从本次日志推算修好工程问题后的必达分数。

**建议下一步，按可验证收益优先级排序**

1. **定位 support 的 17/26 失败。** 补充失败字段、原始结构化响应摘要、finish reason，区分 provider、解析与本地验证；按选项修复并保留已经验证的 claims。固定本轮 packet 重放，先看 65.38% 回退率能否下降。
2. **修正事实／建议边界和验证状态传播。** 用 Q6 验证带 claim_not_in_option 的结构错误不能静默成为 ok 后一票否决；用 Q7/Q10/Q11/Q12/Q19 检验通用建议；用 Q13/Q16 检验相关核心证据不被宽泛 persona 挤出。
3. **收窄遗忘约束，清理 rule 误分类。** Q8/Q19 检查照片建议误杀，Q9 保留精确约束；对两个润色请求明确不走规则保护旁路。
4. **修正来源正文预算与精排保护。** Q18/Q21 要优先保留用户原句；对 Q0/Q6 保持依赖展开后原文可见，指标按最终载体和原句计算，避免错误归因。
5. **分离写入、检索、答案实验。** 先固定 packet 做答案对照，再固定记忆做检索对照，最后单独评估写入。报告空答率、Gold 准入率、语义无效但标记 ok 的比例、关键原句可见率及总分；单次 1→4 分不足以说明稳定提升。

当前已完成输入／分数／版本对齐、逐题来源追踪、与上轮比较，以及确定性准入和回退路径重放。尚未进行修改后的模型重跑或消融，因此上述为有日志依据的修复顺序，不是已验证的提分承诺。
