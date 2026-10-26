# AML Memory System（M1–M3 实现 + ULM 生命周期）

2026-10-25 已恢复召回迭代。最新分批补审成对 HTTPS（相同冻结入库／规划）为
对照 **37/56（66.07%）**、处理 **33/56（58.93%）**，均无终止错误。固定处理组
自身上游的精确打包回放确认参考题 40→41、完整参考回合 56→59、原参考无损，
但答题未通过不退步门槛，`AML_LISTWISE_OVERFLOW_BATCHED` 保持默认关闭。
最近重新入库的单组基线仍为 **34/56（60.71%）**，与冻结对照分开报告。
本轮实施、测试、成本、漏斗及下一轮方向见 [2026-10-25 迭代记录](RECALL_ITERATION_20261025.md)。

完整原选项 CE 诊断已完成两组全部 117 题：回归集主输入参考覆盖 40→42，补充集
27→26；234 次真实 CE 请求及独立来源／选择／成本审计完成。触发题 CE tokens
增加约 63%–65%，`AML_CE_FULL_OPTION_TEXT` 保持默认关闭，不进入对应 HTTPS。
这是精排和主审输入诊断，不是新答题成功率。

主审窗口24,000→32,000字节的234次HTTPS对照已完成并通过终态审计。回归集最终
参考覆盖40→43题，答题33/56→32/56；补充集参考25→28题，答题20/61→18/61。
Search tokens增加6.65%／9.84%。召回改善但两组答题均退步，生产默认保持24,000字节。

后续两项精排预筛均未通过：扩展已可见原文语境在最新117题无主输入参考增益；
完整原文CE只提名一个补充席位，在其原始117题中回归主输入46→45、补充27→28。
两项均未接入应用。CE移除规划器重复子问也已完成117题诊断、606次真实调用及审计：
回归主输入参考40→40、补充28→25；触发题CE tokens降低8.17%／9.31%，但参考损失
未通过门槛，不接入运行时，也不产生新的HTTPS答题成绩。

新增默认关闭的 `AML_FINE_NEW_COARSE_INSURANCE`：全coarse CE前16中被新增候选核心
配额挡住的单位，可竞争一个现有补充席位，保留原核心和全部容量限制。117题生产
回放与离线提议一致：回归main参考40→40、补充28→30，两组原参考回合无损；85项
相关测试通过。234次成对HTTPS及全部终态审计已完成：回归答题29→35/56、补充19→21/61，
但最终参考回归39→37题、补充26→29题，两组均丢旧参考，未通过保留门槛，默认仍0。
后续“仅补缺席者”方案已完成234份快照回放与生产一致性审计：回归main参考40→40，
补充28→29，均无逐回合损失。独立开关`AML_FINE_NEW_COARSE_MISSING_ONLY`默认0，
92项相关测试通过；新的234次HTTPS已完成。回归答题34→31/56、最终参考39→40题；
补充答题17→23/61、最终参考26→26题（新增Q292、丢Q308）。回归答题与补充原参考
保留均未通过门槛，开关保持默认0。Q12部分CE评分用冻结生产救援路径完整审计；
补充选择审计在机器重启后离线补齐，没有重跑答题或追加模型调用。

随后完成117题coarse残余补审诊断：116次gpt-4o-mini调用，记录307183 tokens，
保持原包前缀后参考覆盖39→40、26→28，新增Q31/292/340；两次取消请求的未上报
计费未知。请求、引用、逐回合交付和账本审计通过，但新增来源包含明显无关保留，
语义检查未通过，未接入应用、未启动该方案HTTPS；不产生新的答题成功率。

第10轮完成117题词法／fusion／粗排及真实CE诊断。Porter直接替换索引丢旧参考载体，
因此拒绝；保留旧通道、最多追加20个新候选的方案使回归粗排参考49→51题，但主审
输入40→43题的同时丢Q26，补充组28→26题且有三题逐回合损失，也未通过门槛。
599个新增文本对、117次CE调用、254148 tokens；查询向量另72次调用、15804 tokens。
两组最终离线审计通过，原部分CE失败保留；应用默认未变，没有对应新HTTPS成绩。

第11轮“保留旧粗排候选”预检通过：117题生产回放一致，回归主审参考40→41题，
补充组28题不变，均无旧参考回合损失。新增默认关闭的
`AML_RECALL_PORTER_SUPPLEMENT`，相关78项检查通过。回归HTTPS已完成：答题
35/56→33/56，最终参考题39→39、参考回合57→56，丢失4个旧回合，未通过采用门槛。
新增主审候选人工检查也发现无关材料进入最终包，开关保持默认关闭。补充验证答题
已完成22/61→22/61，处理组含1条连接错误并计入分母；最终参考27题/39回合完全不变。
234条选择回放、两组漏斗/账本及联合审计全部完成，联合采用门槛未通过；两组全部
29个新增主审候选实例的人工相关性检查完成，10个无关实例进入最终包（非盲审）。
Search tokens增加4.93%/8.09%。
按本机2核4GB限制，客户端并发1、评测进程树CPU上限单核75%、内存硬上限1200MiB，
公网服务保持运行。因审计回收停滞，内存软限先从768调至1024MiB；全部答题结束后
末臂自动审计再调至1152MiB，硬限不变。延迟不作同资源比较。
离线终态审计曾触发一次1200MiB进程组OOM；已修正父子进程的数据加载顺序并完成
剩余审计，保留失败记录，未重跑任何答题。公网健康检查仍正常。

第12轮仅延后低CE、只匹配助手原文的独立补充项，完整117题离线回放全部精确复现
原选择；回归21题、验证20题的主审成员改变，但主输入参考分别保持40题/56回合、
28题/40回合，无逐回合增益或损失。未通过增益门槛，不接入应用、不启动对应HTTPS。
精排后到主输入仍缺失11/4个完整参考回合，原记录主要为字节预算拒绝；后续先做
原文重复占用的只读库存。结果见`runs/assistant-main-deferral-20261025`。

第13轮以一个更高CE、来源未覆盖的候选替换低CE助手词面补充项，完整117题精确
回放通过；回归fine/main覆盖均不变，验证fine31题/44回合→32题/45回合，但main仍
28题/40回合。唯一新增fine参考被24KB主审预算挡住，故仍拒绝，不接应用或HTTPS。
仅抽查的5个回归新增main实例中2个无关，未冒称完整语义审计。来源重复库存也不支持
新增展示格式。两项均零模型调用、串行限资源执行，报告见迭代记录与
`runs/fine-boundary-swap-20261025/final-decision.json`。

第14轮对主候选及support正文中的逐字重复摘要做无损共享，117题原结果精确复现，
整体仅节省2407 bytes，主输入参考无增益或损失；不接入应用、不启动HTTPS。
零模型调用，回放进程组峰值约181MiB、无内存限额事件。记录见
`runs/main-shared-bodies-20261025/final-decision.json`。

第15轮在原main中最多用一个更高CE、用户原文有主题匹配的完整fine候选替换仅助手
词面匹配项，保护预留/依赖/用户规则并保持全部容量。117题原选择精确复现，回归6题、
验证12题发生替换，但参考覆盖均无变化，不接入应用或HTTPS。零模型调用、峰值约
186MiB，结果见`runs/main-source-swap-20261025/final-decision.json`。

第16轮允许新增项的词项匹配来自任意可见角色，117题原选择全部复现；回归9题、
验证18题替换，参考覆盖仍无变化，未接应用或HTTPS。第17轮沿用fine资格、不额外
要求词项匹配，回归main参考40题/56回合→41题/57回合，验证28题/40回合不变。
2026-10-26完成全部59对原展示正文的非盲人工复核：14对语境收益、12对语境损失、
10对各有取舍、23对无明确收益。数值门槛通过但语义检查拒绝；未接应用、未启动
对应HTTPS，不能当作答题提升。记录见`runs/main-ce-swap-20261025/final-decision.json`
与`runs/main-ce-swap-20261025-review/manual-review-summary.json`。两轮零模型调用，
回放峰值均约186MiB；本次全部59对复核不等于此前各轮也做过全量人工检查。

第18轮只将CE辅助指令改为查找相关语境，问题/前提/子问与原文保持不变。完整117题
预检通过；回归main参考40题/56回合→41题/59回合但丢Q54，验证复验28题/40回合
→29题/41回合但丢Q283，另有fine丢Q282。完整选择/来源/账本审计通过，仍拒绝采用，
未接应用或HTTPS。验证原尝试在Q288超时，随后单独登记完整61题复验；失败原件和
重复成本均保留。合计683次请求、已报告5697421 tokens，取消请求计费未知。
日志按用户要求采用4MiB分批读取；串行CPU50%、内存硬限768MiB，最高约489MiB，
未发生内存限额事件。联合决策与漏斗见`runs/ce-relevance-guidance-batched-20261026/`
下的`final-decision.json`、`funnel-analysis.json`及`validation-failure.json`。

第19轮仅移除CE评分正文中的推断型主摘要，保留标签、support及全部已可见来源。
8项边界检查、117题精确预检及完整评分/来源/账本审计通过；696次调用无失败，
已报告5692959 tokens。回归main参考40题/56回合→41题/57回合（新增Q2第0回合），
验证28题/40回合不变，均无逐回合损失，成本调整一致。全部55题成员变更的93个新增、
97个移除实例完成非盲语境复核：13题收益、8题损失、9题取舍、25题无明确收益，
另20题仅顺序变化；39个新增实例判为无关。数值通过但语义门槛拒绝，未接应用或HTTPS。
4MiB分批读取、串行CPU50%/内存硬限768MiB，评分及审计峰值约408MiB，内存事件0。
终态见`runs/ce-inferred-primary-20261026/final-decision.json`与`manual-review-summary.json`。

后续support载体盘点完成117题精确回放：1948个main候选实例中78个带继承来源，
43个仅在继承来源命中词项，10个主来源已完整出现在其他认证展示中；这不证明主摘要
无关或可删。与fine→main参考遗漏交集仅回归Q2/Q8/Q37，验证集无交集。15个完整
展示认证差异均为文档episode的排名节选，不能将其包内来源字节当成排名可见字节。
补充审计通过，冻结数据库与前轮审计一致。零模型调用、无新HTTPS成绩，未修改选择；
4MiB分块读取、SQLite每批256条，CPU50%/内存硬限512MiB，盘点峰值约154MiB、
补充审计约71MiB。结论和限制见`runs/support-carriers-20261026/findings.json`。

HTTPS测评入口`evaluate_frozen_https.py`的结果、trace和准备数据读取也改为4MiB
二进制分块解析，只保留需要的字段；跨块UTF8、CRLF、空记录、超块记录和无末尾换行
检查通过，相关8项测试全部通过。单条JSON记录仍需完整解析，不能将4MiB当作总内存上限。

第20轮CE诊断完成：只替换“全部原始消息已完整可见”的可选bridge摘要，保留
主文、标签和全部来源；声明依赖、规则、路径及不完整视图不改。七项边界测试、
117题精确预检与完整账本/来源/选择审计通过。80题/162文档改变，480次请求全部
成功，已报告3962767 tokens；其余37题明确复用历史分数。回归fine46题/67回合、
main40/56，验证fine31/44、main28/40均逐回合不变，无主审参考增益而拒绝采用。
candidate仅节省3895 tokens（约0.20%）；峰值约388MiB，内存事件0。无新HTTPS成绩
或应用改动，结论及漏斗见`runs/ce-bridge-summary-20261026/final-decision.json`、
`funnel-analysis.json`，模型及原预算不变。

后续原fine直接打包的完整117题离线诊断：原接口包/manifest全部精确复现，仍在
top_k50/32000 bytes内，最终参考回归39题/57回合→46/67、验证27/39→30/43，旧
参考回合无损。但原文目录增加82.67%/87.78%，4题16个旧来源卡片未完整保留，
包含1条遗忘请求；新增/扩展3178个卡片实例尚未完成全量语境检查。未接应用或HTTPS。
零provider调用、恢复运行峰值约196MiB；初次归因类型错误及恢复记录保留。
结论与后续门槛见`runs/fine-direct-packet-20261026/findings.json`。

AML（Agent Memory Leaderboard）参赛记忆系统的参考实现，对应
`../agent-memory-system-design.md` v0.4、通用设计 `../llm-memory-survey/memory-system-design.md`
（ULM）与 `../prompts/` 模板库。

2026-10-23 按用户要求，当前布尔开关默认值采用
`https-fine-context-fill-20261023/control` 的配置：新增默认开启粗排扩展、核心配额、
自适应精排、敏感召回、来源多样性、选项主题、上下文标签、共用来源、overflow 补审，
以及原文保留／5 KiB／用户词项席位，共 12 项。全部 65 个记录过的布尔开关默认值
已与该对照逐项核对。`PACKET_FINE_CONTEXT_FILL` 仍默认关闭，LLM 仍为 gpt-4o-mini。
环境变量显式设置为 `0` 可关闭对应功能；敏感召回仍要求请求允许 `include_sensitive`。
下文旧实验中“保持默认关闭”记录的是当时决策，当前默认值以此说明和 config.py 为准。

成功率按完整队列分别报告：原回归集56题，新补充验证集61题；固定请求诊断、
参考覆盖与选中案例重复试验均不能替代答题成功率，不跨队列计算配置退步。

最近完整重新入库的补充验证基线为`https-validation-baseline-20261024`：
**19/61（31.1%）**，两会话分别8/31、11/30，无终止回答错误。29批／474条消息
逐字入库审计、完整题目身份、716份冻结文件、83个当前app文件、24份prompt、
模型与65项历史默认核验通过；客户端／自动审计exit0，服务退出，18603已释放。
这不是新方案处理组；与原56题的62.5%使用不同历史且重新入库，但暴露了当前配置
在补充队列上的弱表现。参考可用48题，粗排47→精排31→主输入27→主选23→最终27。
42道错题中15道缺可用参考、18道参考已送达、9道输入无完整参考；已送达的18道
中9道没有显式引用参考来源。六次内部witness校验失败单列，未成为终止回答错误。

`all-user-source-ce-20261024`诊断已完成，覆盖新61题和原56题完整control，
原qwen3-rerank评分全部合格完整用户原文，实际233＋190次CE调用，无调用错误；
gpt-4o-mini不变。离线审计已重建全部请求、选择、交付与成本。
来源最高分及其与历史整单元分数的最大值均未通过采用条件：新集fine参考题31→33，
但main输入27→23；原集fine47→43，main输入46→34／35。
该方案未接入生产，未新增成对HTTPS；这些是参考覆盖诊断，不是新的答题成功率。
实验循环已按用户要求暂停。

原56题最新完整 HTTPS：`https-unmatched-noun-context-20261024` 对照与实验均为
**35/56（62.5%）**，无终止回答错误。两组都关闭`CHOICE_CITED_SCOPE`，保持用户指定的
65项历史默认和`gpt-4o-mini`。本轮再次达到60%以上，但单轮成绩不证明稳定水平。
新名词关系规则追加35题，固定实验组自身上游后参考题仍45/52、用户回合仍64，
没有直接参考召回增益；阅读目录增加36,631 bytes（5.56%）。跨组42→45来自独立
主审判断，不能算作新规则的收益。3胜3负中仅Q39触发追加，其新增来源没有显式引用；
Q8／18阅读目录完全相同仍退步。新开关继续默认关闭。

本轮参考漏斗（对照／实验，分母52）：粗排50／50，精排47／47，主审输入46／46，
主审后35／38，overflow后36／39，最终42／45。实验组21道错题中，15道已交付完整
参考用户原文，其中7道未显式引用参考来源；4道缺失，另2道参考用户原文在输入中
不可用。Q22原文由原有流程找回、新规则未追加，但仍未被显式引用且答错。
因此继续区分检索漏失与证据使用，不把补入原文等同于答题收益。全部112包／阅读
目录、705份冻结源码／资源、83个当前app文件、唯一开关差异与模型约束已核验。
费用、原始错误和逐题审计见迭代报告末尾及本轮运行目录。

后续主审原文展示顺序诊断已完成，目录`main-source-layout`：56题×两臂，仅将共享
原文JSON字段移到候选卡之前；29题实际改变布局，27题保持原请求。所有来源、ID、
角色、正文、选项、指令、schema与字节预算不变。112次调用、707,615记录tokens
全部审计；12次分组契约错误与完整候选判断分开报告。改变布局的29题主审候选阶段
参考17→20，但新增参考回合在历史最终包中已可见，且丢Q12两个回合、Q22仍被排除。
未满足预登记条件，不采用、不新增HTTPS，生产配置和最新62.5%完整成绩不变。

`main-relevance-axes`诊断已完成，比较主审的两种判断结构：原单一relevance，及分别
判断原问题／选项相关性、仅两者均irrelevant时排除。56题固定候选、来源、选项与
原调用余量，8项边界检查通过；112次调用、743,043记录tokens已审计。54个候选判断
有效配对中参考题32→33、完整回合47→46，并出现明显不相关保留，因此不采用。
候选／分组格式错误和19项完整候选人工检查另行记录，未声称所有新增候选都相关。

补充验证基线`https-validation-baseline-20261024`已完成。按原数据
顺序选择未出现在352份已保存结果文件中的两个完整会话（索引11／12），共61题；
保留原56题作为回归集。仅评估当前默认配置，模型仍gpt-4o-mini，完整分数见上文。
该基线使用不同历史且重新入库，不能与原56题的62.5%直接比较或当作新规则收益。
队列选择及局限见`runs/validation-cohort-20261024/manifest.json`与预登记。

以下为运行期间的诊断记录，进度数字不代表当前状态。入库审计已扩展为多会话：
第一会话14批／222条消息的原文、顺序和归属逐条通过，
第二会话按顺序等待第一会话答题结束。运行快照已完成8／61题、无终止回答错误，
不报告中途正确率。前3题漏斗中的Q282／283参考载体在精排后因主审字节预算遗漏，
又分别被补审数量／字节上限挡住；独立补审请求为8,010／8,443 bytes，均可装入12 KB。
这只是后续分批补审的调查线索，尚未验证剩余调用预算、相关性判断或最终答题收益。
原始进程继续运行，716份冻结文件、83份当前app文件与65项历史默认核验一致。

第二批补审离线检查已覆盖原56题两组和新基线前16题：保持原主审／第一批结果，
按Search结束时剩余call/token和保守时间代理提议最多4项／12 KB。原两组各需54批、
145项，但最终参考覆盖上界仍42／45；新队列早期13批／28项的上界2→4，仅补Q282／283。
尚无实际补审、最终打包或答题收益，不直接开启。新基线前17题中12题参考原文可用，
粗排12→精排6→主输入3→最终3，精排遗漏也需在全61题结束后优先评估。

第一会话31题现已结束，完整基线仍在运行：该会话23题参考可用，粗排23→精排13→
主输入10→最终10。复用既有CE边界2席位规则做离线检查，原56题主输入46→45，
新会话精排13→12，均无新增参考题，故不接入生产或启动该方案HTTPS。
第二会话已提交6批／108条，原文逐条通过；不把第一会话结果当作完整61题成绩。

随后两会话29批／474条消息已全部入库并逐条审计通过，第二会话正在答题。
新增单组基线审计`audit_validation_baseline.py`，明确区分进度与终态，核对源文件、
模型／256维、运行配置、输入顺序、参考交付／引用及Add／Search／reader账本。
第一会话Search 290次／1,390,373记录tokens与日志一致，3次内部witness错误没有
变成终止回答错误。已有每候选前两条用户原文的CE视图只覆盖21／23题可用参考，
遗漏Q306／311，不能直接复用为无损精排方案。以上均未改变生产策略。

来源CE诊断工具现支持显式检查全部合格完整用户原文；默认仍保留旧的前两条规则。
首段31题的默认兼容性逐题通过，全来源视图覆盖23／23题、32／32个coarse参考回合，
预计121批，相比旧视图93批增加28批。原56题全部fine／main复现通过，全来源预检为
190批、3,847个题内去重视图、无超限遗漏。尚未执行这些CE调用或改变生产策略，
先等完整61题漏斗决定是否采用该诊断方向；视图可见不代表实际精排或答题收益。

上一轮完整 HTTPS：`https-exclusive-shared-context-20261024` 对照 **31/56（55.4%）**、
实验 **33/56（58.9%）**，两组均无终止回答错误。相对本轮对照6胜4负，其中7道变化题
没有触发新增上下文；Q0／31的阅读目录完全相同仍分别退步／进步。不能把净增2题全部
归因于新规则，也不能把它与上一轮60.7%直接解释成同配置退步：上一轮实验启用
`CHOICE_CITED_SCOPE`，本轮两组均关闭。历史最好单轮成绩、当前默认成绩、新方案成绩
须分别报告，56题每差一题约1.79个百分点。新上下文开关仍默认关闭，用户指定历史
默认和`gpt-4o-mini`保持。逐题核对见本轮`score-attribution-audit.json`。

后续两项离线检查未形成新收益：112包中没有只能跨候选合并片段才能恢复的完整用户
消息；选项主题预留改用原始来源统计词频，在三份56题快照中精排仍47、主审输入仍46，
只改变Q33且增加542字节。两者均未接入生产、未启动新HTTPS，当前全量成绩不变。
产物见本轮`prepared-source-union-audit.json`与`option-source-frequency-proposal.json`。

`https-unmatched-noun-context-20261024`已完成。实验只为原问题中字面词匹配
漏掉的名词关系补一个完整候选，排序分数必须来自尚未交付的完整用户消息。三份快照
在原本缺Q22的两份中补回该参考，无原回合损失，目录增加约5%；WordNet词义歧义仍
可能引入无关内容。36项测试、168包生产回放通过；完整结果为上述35/56与35/56，
本轮没有直接参考召回增益，新开关默认关闭。模型和历史默认保持。

上一轮完整 HTTPS：`https-cited-scope-shared-20261024` 对照 **31/56（55.4%）**、实验
**34/56（60.7%）**。56 次真实 Search 响应逐字共享给两个独立 reader，只在实验 reader
打开已有 `CHOICE_CITED_SCOPE`；两组参考覆盖均 44/52，无终止回答错误，112 个完整
回答的全部 712 次调用已离线精确复现。4 胜 1 负中，Q51 的历史前置输出相同，
扩大检查目标后模板占位符被正确识别；固定该前置输出再测三次，关闭均错、开启均对，
姓名地址事实仍保留。其余四题前置输出不同，整体收益尚未证明稳定，开关仍默认关闭。
Q11 的旧失分路径另有三轮固定前置输出诊断支持；本次新 HTTPS 两组 Q11 均正确。
后续目标分句诊断已完成316次真实调用、1,043,436记录tokens，全部响应和56题历史
选择通过审计。使用的是较早的fine-semantic-reserve实验组补审集合；120对改变布局、
38对原样回退。改变布局的有效正判47→47，参考覆盖表面36→38来自无效判断被保留，
并丢失Q4参考。因此不采用、不新增默认开关，也不产生新的全量HTTPS成功率。
产物和8份归档脚本见最新run下的`target-sentence-study.json`与
`target-sentence-study-scripts/`。81个app文件及65项历史默认再次核验一致。

随后“仅原问题补审”使用最新control的159个实际补审请求完成318次配对调用，
970,108记录tokens，全部响应及56题历史选择审计通过。有效正判56→69，但唯一有效
参考恢复Q22在原最终包中已经可见；另四题的阶段增益来自无效保留，且阶段参考丢失
Q11／25。全部26项新增正判另行检查，发现多项明显不相关内容，因此也未采用。
此诊断没有新HTTPS答题成绩；产物见最新run下`question-only-review-study.json`。

`AML_PACKET_EXCLUSIVE_SHARED_CONTEXT=1`（默认关闭）实验：共有选项词至少出现在两个
原始选项、只见于当前已准备coarse语料的一条原始消息、且未在原包来源中出现时，
提议追加一个完整候选。保留原三席位，只用5 KiB剩余容量；不改变来源角色、不晋升
个人事实。三份同56题快照均新增Q7参考、无原回合损失，目录增加3.9%–4.5%；43项
测试和168包生产回放通过。完整HTTPS `https-exclusive-shared-context-20261024`
已完成，答题成绩为上述31/56与33/56；离线参考覆盖收益不能替代答题成功率。

历史默认和模型保持，完整漏斗、费用、限制及下一步见迭代报告末尾。Search 来源身份
稀有词去重提议在 168 次回放没有改变选择，未接入生产。

后续原文语义诊断保持上述 HTTPS 成绩不变：在同一冻结数据库测量 215 条合格用户原文、
两句窗口及 182 条真实相邻问答，复用原模型 `openai/text-embedding-v4`／256 维，
共 126 次 embedding 调用、87,749 tokens。六种有预算限制的现有粗排载体补充方案
均未增加最终参考题（仍 44/52）；两种 RRF 方案补齐 Q17 的一个用户回合，完整集合
38→39，目录却增加约 7%。该原文是汽车技术提问，不直接证明美国车偏好，不能算答题
收益。上游实际重放与 55 次真实 CE 后，两句窗口 prepared 覆盖达到 52/52，但主输入
反而 46→43；问答视图精排 47→48，主输入仍为 46 且少一个用户回合。保留原包的
追加方案可补回 Q40，却只是 153-byte 泛化润色请求，Q40 原本已答对。完整问答可见性
检查后这一局限仍在，未接入生产或冒充新 HTTPS 改善。另查出 33 条无法配对的用户
来源：21 条位于写入请求末尾、11 条与相邻助手无共用载体、1 条下一消息不是助手；
后续从明确的对话顺序与载体关系验证上下文衔接，不凭请求 ID 推断跨请求邻接。

后续相邻来源与固定前置输出诊断也没有产生新 HTTPS 成绩。明确同 request 邻接、但
没有共用载体的 11 组来源已完成向量测量与完整交付检查；两个受限追加方案均保持
44/52 参考题、61 用户回合、38 完整集合，未增加参考覆盖。现有 core-gap 补原文与
cited-scope 的 2×2 局部试验完成 24 次回答：新增原文后 Q47 仍可能错误推断文化血统。
再固定全部前置输出测试一般身份依据规则，24 次尾段执行中，规则关闭 5/12、开启
4/12，未改善；这是 Q11／41／47 的重复诊断，不能当作全量成功率。新增规则未接入
运行时，现有 81 个 app 文件及 65 项历史默认核验不变。相关工作共 180 次 provider
调用、813,935 记录 tokens，含一次 HTTP503；失败调用记录 0 不代表已确认零计费。
完整归档为最新运行目录的 `adjacent-grounding-study.json`，细节见迭代报告末尾。

最新离线 Search 检查进一步区分“命中来源未交付”与“载体含其他新消息”：19 个追加
组合、1,064 个题×方案均保留原包，但参考题仍 44/52。最好方案补齐 Q17／21 两个
用户回合（61→63、完整集合 38→40），Q21 原本已答对，Q17 原文仍不证明美国车偏好。
额外 4 KiB 也没有新增参考题。全56题原文／FTS 路线重建后，按实际命中来源重分组
使原文通道参考载体题 48→45／46，与其他历史通道并集仍 50；未采用。
本轮零 provider 调用，没有新的 HTTPS 或默认变更；完整审计与八份脚本归档为
`source-anchor-allocation-study.json`、`source-anchor-study-scripts/`。

上一轮完整 HTTPS：`https-sparse-source-context-20261024` 对照 **34/56（60.7%）**、实验
**30/56（53.6%）**，两组均无终止回答错误。新增稀有词同源完整载体规则直接补回 Q30，
固定实验组自身原包时参考题 41→42、用户回合 58→59；跨组最终参考覆盖均为 42/52。
Q30 新原文进入真实阅读请求但未被显式引用，两组仍答错。2 胜 6 负中，Q11 原目录相同，
新增来源被引用后产生前提范围／蕴含判定冲突并拒答，不能把全部下降归为随机波动。
新开关继续默认关闭，用户指定的 65 项历史默认和 gpt-4o-mini／qwen3-rerank 保持。
完整审计及下一步见报告末尾。HTTPS 冻结 v40；本地 v41 修复来源身份计数，已用全部
112 个实际包验证阅读输入完全一致，不冒充另一次真实测评。

上一轮完整 HTTPS：`https-core-gap-context-20261024` 对照 **35/56（62.5%）**、实验
**32/56（57.1%）**。此前实验组 34/56（60.7%）是单轮结果，未证明稳定达到 60%。
本轮 1 胜（Q47）、4 负（Q2／26／27／28）；四道退步题均未触发新规则，Q26／27
连最终阅读目录也完全相同，说明存在回答判断波动，不能把总分差直接归因于新规则。
最终参考覆盖 43→46／52，但固定实验组自身上游后，新规则只贡献 Q47 一题。
新增原文确实进入 Q47 的引文请求，却没有被显式引用；答对伴随个人前提范围判断改变，
尚未证明来自新证据的正确使用。因此新开关继续默认关闭，历史 65 项默认和
**gpt-4o-mini／qwen3-rerank** 保持。完整核验及漏斗见迭代报告末尾。

上一轮完整 HTTPS：`https-fine-semantic-reserve-20261024` 对照 31/56（55.4%）、实验
34/56（60.7%），无终止回答错误。新增默认关闭的 `AML_FINE_SEMANTIC_RESERVE=1`
在原 8 项补充容量内，为核心池 CE 前 16 中被预留名额挤出的候选提供至多一次检查
机会；原核心、依赖优先级、来源文本和预算不变，无新增模型调用。Q47 的原文进入
fine／overflow 后仍被模型判为无用，最终两组参考覆盖均 44/52。4 胜 1 负的五题均
未触发新规则，不能把总分差归因于该改动，故继续默认关闭。56 项相关测试、112 题
生产选择重放与源码／模型／默认配置审计通过，详见迭代报告末尾。

后续候选内来源布局诊断完成 316 次真实 gpt-4o-mini 调用：正判 61→72、无效判断
52→40，但固定补审后参考题 37→36、完整用户回合 52→52，Q47 仍被排除。未进入生产
或完整 HTTPS，当时端到端成绩为上段结果。七份近期对照的描述性平均为 57.4%，
范围 50.0%–64.3%；同一 56 题的重复执行及不同协议不能当作独立验证集。

`AML_PACKET_CORE_GAP_CONTEXT=1` 是默认关闭的后续实验：原三项上下文保留，从 CE
核心池前 16 中未进 fine 的候选再追加至多一个完整来源。额外单项最多 4 KiB，原三项
5 KiB 配额不变，总局部上限 9 KiB，仍受请求总字节和条数上限约束；没有新模型调用或
事实升级。三份固定快照都补回 Q47、原来源无损，目录约增 0.4%；35 项测试和 168 次
生产回放通过。完整双臂 HTTPS `https-core-gap-context-20261024` 已完成，62.5%→57.1%，继续默认关闭。

`AML_PACKET_SPARSE_SOURCE_CONTEXT=1` 是默认关闭的新实验：原三席位和 fine／主选择
保持，从已评分 coarse 中寻找用户原文尚未交付的稀有需求词；按最佳原始候选排序来源，
优先尝试携带同一条完整原文的较小已有候选。最多追加一项，仍共用原 5 KiB 上下文
额度，不增加 provider 调用、不裁剪来源、不升级个人事实。三份冻结快照中两份补回
Q30，另一份无新增参考，原参考回合无损；目录增约 4%–5%。31 项测试、168 次生产
组包重放通过。完整双臂 HTTPS 已完成：60.7%→53.6%，未证明答题收益，继续默认关闭。

`AML_PACKET_QUESTION_CONTEXT_CE=1` 是默认关闭的新实验：保留原两项 CE 和一项用户
词面上下文，在原 5 KiB 剩余容量内至多追加一个完整来源。只用原问题对剩余且可交付的
fine 原文做一次 CE 排序；每次调用都检查原请求剩余时间、调用和 token 预算，失败时
保留原包。依赖 reserve／5K／lexical 开启，fill 与 source-query 关闭。该排序不认定
个人事实。56 题冻结诊断覆盖 43→44，生产实现已用保存分数逐题重放一致；随后完整
HTTPS 对照两组均答对 32/56（57.1%）、最终原文覆盖均 44/56。新增 CE 实际 45 次、
52,827 tokens，固定实验组自身原包时无新增参考回合，目录增加 1.97%，因此不采用。
LLM 为 gpt-4o-mini，CE 为 qwen3-rerank，详见迭代报告末尾。

`AML_PACKET_SOURCE_FIRST_RESIDUAL=1` 是默认关闭的后续实验：保留原三席位，仅从已
评分 coarse 中接纳原问题原文的首个 source_text 命中，候选须未进入 fine 或主选择，
在原 5 KiB 剩余容量内至多追加一项完整来源，无新增模型调用。依赖 reserve／5K／
lexical 开启，fill／source-query／question-CE 关闭。三份相同 56 题的离线快照均
补全 Q17 一个用户回合，目录增加约 0.6%–0.8%。完整 HTTPS 对照为 33/56（58.9%）
→32/56（57.1%）；固定实验组自身原包确实补全 Q17，目录增加 0.64%，无原来源损失，
但未证明答题收益，因此仍默认关闭。详见迭代报告末尾。

`AML_PACKET_MARGINAL_QUERY_CONTEXT=1` 是默认关闭的新实验：保留原三席位，复用原问题
和字面选项主题的 source_text 名次，按其相对已交付来源的名次增益排序，在同一 5 KiB
余量内至多追加一个完整来源。不增加 provider 调用，不把检索命中当个人事实。
依赖 reserve／5K／lexical，其他原文追加实验关闭。三份重复开发集快照均新增 Q7
参考并补全 Q17，目录增加约 4.4%–5.0%。完整 HTTPS 为 28/56（50.0%）→30/56
（53.6%），固定实验组自身上游新增一个参考题、两个完整用户回合，目录增加 4.39%。
43 项测试及完整审计通过，仍默认关闭。此前对照 33/56 在本轮未稳定复现，已逐题
核验排序／回答变化及 reader 内部错误；不能据单轮多 2 题宣称稳定提升，详见报告末尾。

`AML_CHOICE_WITNESS_KEYED_OUTPUT=1` 是默认关闭的回答端实验：用显式选项键组织引文
候选输出，继续要求所有选项齐备、来源与逐字引文通过原校验。56 题固定目录成对诊断
校验失败 4→0，有效逐字引文 172→277；这不是答题率。112 个请求／响应生产重放
一致，47 项测试通过。完整同证据 HTTPS 已完成：56 次实时搜索供两个独立回答进程，
对照 36/56（64.3%），实验 34/56（60.7%）；引文阶段回退 4→0，但没有答题率收益，
继续默认关闭。两组输入／目录和 112 个引文请求均已核验；共享搜索有一次 CE 超时，
其终态指标缺失、费用未知，已按 trace 单列。此前同一对照的 58.9%→50.0%→64.3%
是跨轮波动，不能将峰值视为稳定成绩；本轮新增分析见迭代报告末尾。

`AML_PACKET_QUERY_CONTEXT_COMBINED=1` 是默认关闭的组合实验：来源首席、原 lexical
席位和一项查询边际增量共用原 5 KiB 额度，不增加 provider 调用。三个 control 快照
共 168 次离线生产重放全部一致，参考回合无损失，目录增加约 4.2%–4.4%；这不是
答题率。完整 HTTPS 两组均为 32/56（57.1%），实验组含一次回答失败，按原始 0 分
保留；固定实验组上游的参考题 42→44、完整参考集合 36→39，但未带来答题率收益，
继续默认关闭。64 项测试及离线审计通过。另已修复 provider 取消终态漏记，实测两组
Search 调用数均与日志对齐；取消调用的费用仍明确标为未知。

此前对照 58.9%→57.1% 已逐题复核：33→32 是 4 题变对、5 题变错，56 题
fine／主输入均相同，Q21／47 连回答证据目录都相同仍发生答案翻转。最终参考题覆盖
反而 44→46，不能把答题变化直接归因于检索退步，也不能把单轮峰值视为稳定水平。
后续全量复核标签诊断完成 276 次真实 gpt-4o-mini 调用：有效正判 56→54、无效
判断 42→44；新标签没有被模型输出。虽固定复核回放多一题参考原文，尚无机制或
答题收益证据，未增加生产开关。65 项历史默认值再次核验，详见迭代报告末尾。

此前完整测评 `https-witness-lexical-fill-shared-20261024`：对照和实验均 **33/56（58.9%）**，
两组无终止回答错误。新增默认关闭 `AML_CHOICE_WITNESS_LEXICAL_FILL=1`，在原语义／
coverage 引文之后用已有用户来源词项匹配补空余名额；原引文不变、每选项最多三条，
无新增模型调用。实验实际向 8 题补 12 条，固定自身输出时正确选项参考来源提示
11→13；Q8 两组都对，Q55 两组都错，未证明答题收益，继续关闭。336 次历史生产
回放、54 项测试及本轮 224 次提示回放通过，65 项历史默认和模型硬要求均保持。

## 架构

```
app/
  config.py          环境变量配置（模型/密钥/路径/阈值/ULM 生命周期参数）
  schemas.py         Add/Search 契约的 pydantic 模型
  llm.py             LiteLLM 统一抽象 + 离线 FakeLLM（AML_FAKE=1）
  embeddings.py      LiteLLM embedding（text-embedding-3 原生 Matryoshka 短向量）+ 哈希 Fake
  store.py           SQLite 存储（AMU + scenes + FTS5 + triples + 幂等账本 + 热度/分层字段）
  segment.py         语义边界切分（SeCom/EverMemOS，相邻消息嵌入相似度骤降）
  scenes.py          MemScene 增量聚类、MemoryOS 热度/晋升/驱逐、艾宾浩斯冷热分层、画像稳定性
  experience.py      显式任务反馈→策略/工作流/技能/手册（成功与失败经验）
  graph.py           实体-AMU 二部图 + PPR（HippoRAG 式单步多跳）
  add_pipeline.py    切分→MemCell 抽取(episode+facts+triples)→新颖度门控→治理→多索引→场景巩固→滚动摘要
  search_pipeline.py 查詢規劃→多路召回／多跳補查→圖融合→固定證據單元→粗排→Cross-Encoder→LLM 列表排序→證據組裝包
  eval_scoring.py    直接基于同一证据包回答；选择题确定性计分
  main.py            FastAPI: /add /search /feedback /memory/{user_id} /health
scripts/
  selftest_contract.py   契约自测 11 项（幂等/回显/隔离/422/health…）
  selftest_ulm.py        ULM 生命周期自测（切分/场景/门控/热度/遗忘/单轮召回/前瞻）
  convert_locomo.py      locomo10.json → 评测格式（1542 QA）
  local_eval.py          本地代理评测 + 消融开关
data/
  sample_eval.json       内置小样例
  locomo10.json          LoCoMo 原始数据（已下载）
  locomo_eval.json       转换后的评测数据
```

存储与检索细节（包括 ULM 各环境变量）见 [STORAGE.md](STORAGE.md)。

## 快速开始

抽取按事实校验证据。时间短语若存在于已引用的原消息中但被短引用遗漏，
会扩展为该消息的完整原文，并继续语义校验；不会跨未引用消息借用时间或
自动猜测日期。单条事实失败时保留其他有效事实，将失败来源合并保存为
episode；引用本身无效、无法定位时保留整批原文。语义批次校验被拒绝时，
才追加逐条校验以隔离错误。日志包含批次起点、事实序号和来源消息序号。
實驗開關 `AML_EXTRACT_TRIPLE_ISOLATION=1` 在逐條語意拒絕且有圖三元組時，
移除該條的全部三元組，再獨立驗證未改寫的事實；來源引用、角色、時間、狀態及隱私標籤均不變。
只有第二次完整驗證通過才保留原子事實，不能僅因引文有效而接受；未通過仍回退原文 episode。
無三元組、來源引用不合格或原本驗證通過的事實不增加呼叫。
此路徑最多為每條被拒絕的事實增加一次 `add.verify_fact_without_triples` 呼叫，
`semantic_triples_omitted` 記錄被省略的圖投影和原始拒絕原因；預設關閉以便消融。

Search 的最終證據包預設上限為 32,000 UTF-8 bytes（保守 token 上界，
不是 32k 個模型 token），可透過 `SearchRequest.evidence_token_budget` 或本地評測
`--evidence-token-budget` 設定。整條證據在搜尋結束時裝包；回答端核對 packet hash
後讀取同一包，不再另行截斷。舊格式、沒有 hash 的輸入仍使用 24,000／2,400
字符的總額／單條上限。

讀取鏈路使用 `graph_cascade_v9`：一次查詢規劃、多路召回、可選一次有來源依據的多跳補查、圖融合、三級排序和原子證據組裝包。
規劃器以 `option_queries` 為每個選項返回一條不超過 240 字元的短前提查詢；通用建議用空字串。
缺失／無效項目回退到選項首句或前提，保留否定與條件，不把選項寫成記憶。
原問題、每個選項和最多 3 個子問題保留；6 條軟上限只限制額外擴展，不再擠掉第六個選項。
選擇題不注入畫像 digest，初始擴展須限於題目／選項詞彙；無選項查詢仍可使用相關畫像。
多跳可用已見證據中的新詞補查一次，預設最多 2 條；附帶種子證據並保留連接路徑，不能憑空添加實體。
冷記憶仍可召回；不執行充分性模型。規劃或 embedding 失敗會記錄降級，詞面路徑仍可用。
向量／原文／全文／畫像的預設配額為 40／40／20／8，規則獨立保留，配額不受 `top_k` 拉高。
graph／scene 在多跳、敘事、文件查詢，或任一查詢匹配的獨立原始來源不足 3 個時啟用；
兩者合計最多 12 條。可用 `AML_RECALL_*` 調整；`top_k` 是最終上限，不保證填滿。
原文增加 `source_text` 检索路由，仍通过来源关联的 AMU 检查用户、历史和敏感权限。
圖融合以向量相似度和查詢詞匹配建立先驗，再沿已記錄的有向三元組、版本依賴及補查連接傳播。
同一來源的重複項降權；重複命中路徑不累加票數。圖權重依查詢類型調整；沒有有效邊時使用內容先驗。
圖 metadata 讀取前先准入最多 `AML_GRAPH_FUSION_MAX_CANDIDATES=256` 個候選，使用者規則另外保留。
圖邊是检索關聯，不能當作因果、權威性或邏輯蘊涵的證明。

三級排序程式預設為 **粗排 50 → Cross-Encoder 精排保留 16 → LLM 列表比較 16**，
分別由 `AML_CASCADE_COARSE_LIMIT`、`AML_CASCADE_FINE_LIMIT`、`AML_CASCADE_LLM_LIMIT` 控制。
粗排／精排只為可見原文支援的需求保留位置；一般詞彙覆蓋保留作診斷，不取得硬保留名額。
精排依分數順序選擇，僅延後原文已由其他選中項完整涵蓋的候選；部分來源重疊仍可保留新增證據。
已選保留項會傳遞到 LLM 名額與字節檢查，無法容納時記錄 `reservation_missing`，不重新換代表掩蓋損失。
CE 查詢只加入有界的選項前提與子問題，不拼接完整建議答案；正文的主體、否定與時間仍保留。
部分 CE 失敗時，按查詢分面保留少量未評分候選給 LLM 比較。
固定證據單元含必要原文、來源與版本依賴，上限是 `AML_SEARCH_ITEM_MAX_BYTES`、
`AML_CE_MAX_DOCUMENT_BYTES` 和包預算的最小值，預設有效上限 4,000 UTF-8 bytes。
無法容納的完整必要引文會明確省略，不會為模型默默截斷。

Cross-Encoder 使用提供者的真正 Rerank API。DashScope 自動沿用相同服務商憑證及 `qwen3-rerank`；
SiliconFlow 自動使用 `BAAI/bge-reranker-v2-m3`。其他部署可設定 `AML_CE_API_URL/MODEL/API_KEY`，
支援平面 Cohere 相容格式及 DashScope 原生格式。憑證只會在相同 origin 自動沿用。
CE 每批最多 24 條、最多 2 批並行，每次最多 10 秒、整階段最多 12 秒；仍受 Search 總預算限制。
單批逾時在原階段期限與剩餘額度內最多小批重試兩次，每批預設 4 條，受 `AML_CE_RETRY_BATCHES`／`AML_CE_RETRY_BATCH_SIZE` 限制；已成功項不重送，最後一次 LLM 呼叫仍預留。
CE 分數保留原始有限值（包括負值／零值），作為排序訊號，沒有跨 LLM 批次校準或固定相關性門檻。

LLM 一次看到最多 10 個候選，回傳完整索引排列、明確無用項和必須共同使用的證據組。
完整提示詞最多 24,000 UTF-8 bytes；格式錯誤最多修復一次，連線錯誤直接降級。
CE 與列表排序呼叫會先預留輸入／輸出 token 保守上界，並在結束、失敗或取消後釋放預留。
CE 不可用時明確記錄降級；LLM 失敗時保留 CE 順序，兩者均不可用才使用圖先驗，
未評分回退最多 `AML_EVIDENCE_FALLBACK_ITEMS=8` 條。明確拒絕的候選與未選中尾部不補包。
`AML_FAKE=1` 的 CE 詞匹配只供離線管線測試，不代表模型品質。

打包保留列表相對順序。證據組整組准入或省略，避免 `top_k`、字節預算、時間過濾或來源去重拆斷必要鏈。
使用者指令及遺忘規則保留；來源、權限、版本與字節預算限制仍適用。
`coverage_manifest.selection_mode=graph_cascade`；`fusion`、`cascade` 和 `evidence_groups` 記錄各階段選中與省略原因。
`rerank_status` 區分 `ok/recovered/partial/fallback/not_run`，未恢復的錯誤會設置 `search_degraded=true`。
舊 `AML_RERANK_MAX_CANDIDATES/BATCH_MAX_CANDIDATES/CALIBRATION_ANCHORS` 和
`AML_EVIDENCE_MIN_RELEVANCE` 僅留給歷史回放，對目前線上級聯無效。
完整設定、限制與驗證見 [GRAPH_CASCADE_IMPLEMENTATION.md](GRAPH_CASCADE_IMPLEMENTATION.md)。
画像按查询相关性召回；相同偏好合并展示并保留全部来源 ID。遗忘规则优先装包，
计入总字节预算和 `top_k` 总条数上限，因此返回条数始终不超过 `top_k`。
来源节录按相关性选取，正文优先于短润色请求，保留引文、邻近语境和消息角色。
預覽與最終節錄使用相同短查詢，按句子局部匹配密度排序，避免長篇 persona 靠散落詞彙占位。
同段的 `Marcus wrote:` 等人物歸屬與必要引文一起保留；最終裝包僅移除包內已可見的重複来源片段，保留引用。
只有來源集合、預覽及語義狀態相同的 episode 才在重排前合併。
trace 的 `query_specs` 記錄短查詢及選項對應，`expansion` 記錄擴展原因、直接命中數及配額。
新版 episode 索引带角色的原始对话，不再用“请求润色”等摘要替代正文。
抽取校验失败本身不再把普通原文标记成敏感；显式来源／模型隐私标签仍保留。
推断型记忆显示来源归属提示；主体、个人前提和遗忘范围由回答模型统一判断。

單選題使用 `verified-source-choice-v5-semantic-witness`：先對完整可見來源做一次 LLM 語意選句，
每個選項最多保留 3 條、每條最多 320 字元的原文引用，不以關鍵詞交集預先排除候選。
語意相關只是候選提示；後續仍獨立驗證引用歸屬、個人前提、遺忘約束及選項資格。
語意選句最多一次呼叫、20 秒、48,000 UTF-8 bytes 提示；個別無效引用會被拒絕，其他有效引用保留。
提供者失敗、輸出無法驗證或預算不足時回退關鍵詞匹配，詳情記於 `answer_witness_retrieval`，
呼叫記於 `answer_calls` 的 `eval.choice_witness`。`AML_CHOICE_SEMANTIC_WITNESSES=0` 可關閉；
`AML_FAKE=1` 使用明確標記的離線關鍵詞回退，不代表語意品質。
實驗開關 `AML_CHOICE_WITNESS_SPAN_REFS=1` 將完整可見來源無損分為不超過 320 字元的原文片段，
模型僅選擇來源及片段 ID，程式還原精確引用；不預先依關鍵詞篩選來源，也不新增模型呼叫。
來源歸屬、個人前提、否定、條件及語意驗證仍使用原本完整來源和獨立門檻。
未知 ID、跨來源引用及重複來源會明確拒絕；其他有效引用保留，全部失敗時仍明確回退。
`answer_witness_retrieval.citation_format=source_span_refs_v1` 標示此路徑；
`answer_calls.response` 保存模型原始引用 ID，`compiled_response` 保存驗證後還原的引文。
此開關預設關閉，須比較同條件實驗及其他對話的結果再決定是否啟用。
回答通常需語意選句、前提支持、語意驗證三次 LLM 呼叫；有約束、復核、格式修復或多個合格選項時增加。
獨立單選回答預算最多 9 次提供者呼叫；共享既有預算時不擴張上限。其他選擇題仍使用 `direct_evidence_v2`。
Cross-Encoder 批次與 embedding 另計入 Search 的提供者呼叫預算。
`choice_alignment.py` 及其 replay 腳本保留作歷史離線診斷。
實驗開關 `AML_CHOICE_SOURCE_NOTES_SELECTION=1` 僅在多個合格選項全被判定為通用建議時，
將既有選擇呼叫改為逐選項來源編號筆記，再選出答案；資格、遺忘約束及後續復核不變。
每個筆記最多引用兩個來源或約束 ID，本地檢查完整選項覆蓋並還原原文；
ID 有效不代表語意支持成立。診斷記於 `answer_selection_notes`，預設關閉。
完整 16 題閱讀流程配對結果為 4/16 對 4/16；其中一個回退案例在固定前序輸出後仍為
原選擇器 2/2、筆記選擇器 0/2，因此不建議啟用。另一段完整 26 題對話回歸為
15/26 對 15/26，兩組共 42 題均為 19/42；沒有總分收益。
`AML_CHOICE_BOUNDED_SUPPORT_SPANS=1` 可在模型超出每項 8 個前提時，接受至多 16 個
不重複的原選項片段。可用片段 ID，或唯一逐字匹配該片段的原文；自由改寫、重複、
跨選項引用及畸形引用仍拒絕。生成結構仍要求至多 8 項，後续證據與語意驗證不變。
此開關預設關閉；原文表示恢復已通過 15 項邊界與整合測試。完整 HTTPS 對照為
4/16 對 6/16，但所有來源目錄均改變，三個新增正確題都未觸發超長恢復；
實際觸發的題目仍錯。固定前序診斷確認減少格式修復，尚未證明準確率收益。
實驗開關 `AML_CHOICE_INDEPENDENT_SELECTION_REVIEW=1` 使既有最終相關性復核
不接收首選答案字母，僅獨立比較同一證據、前提判定和合格選項，不增加呼叫或放寬准入。
診斷仍保存首次選擇以便對照，`answer_selection_review.mode` 為 `independent`；
復核失敗或預算不足仍保留首次選擇。預設關閉，需用固定輸入比較是否減少選擇偏差。
實驗開關 `AML_CHOICE_TYPED_SCOPE_CONSISTENCY=1` 讓既有語意驗證同時檢查無引用前提，
並提供完整選項作語法上下文，不增加模型呼叫。只有完整選項及其每個前提／事件子檢查
一致判為 `no_personal_premise`，後續復核沒有改判，且原文語法符合既有非斷言形式，
才將誤抽取的建議片段移除。模型三態標籤本身不能抹除所有權、習慣或真實過往；
局部片段需由完整原文中的建議／條件語法交叉確認。
缺少某個前提判定、真正未支持的個人事實、矛盾或無效引用均不因整體選項通過而升級。
原文、來源驗證、遺忘約束和預算上限不變；提示可能變長。
`answer_typed_scope_consistency` 記錄原始三態判定與移除片段。預設關閉，需跨對話實測。
實驗開關 `AML_CHOICE_DIRECT_REQUEST_SELECTION=1` 僅改變原有答案選擇提示：
逐一依使用者要求的明確交付項、限制與目標比較同一組合格選項；個人化只有實際幫助完成
要求時才相關，不依字數、文風或看似具體而加分。不增加呼叫、不改變證據或合格集，
診斷版也會沿用於既有復核。預設關閉，需用相同輸入成對驗證。
實驗開關 `AML_CHOICE_STRUCTURED_REQUEST_SELECTION=1` 在原選擇呼叫中共同產生
1–5 個明確請求準則與每個合格選項的 met/missed 判定。程式本地用同一組準則優先度計算
覆蓋分數，只以模型選擇解決完全同分；每個選項和每個準則均須完整、唯一列出。
證據、合格集與呼叫上限不變；模型的覆蓋判定仍可能出錯，因此須在凍結輸入上與一般選擇器比較。

### 实验归档和漏斗审计

在 `memory_system/` 中执行 `bash scripts/run_experiment.sh`。每轮会在独立的
`runs/experiment-*` 下保存 `output.log`、`exit-code`，以及 `completed/` 中的
记忆日志、检索日志和结果 JSONL，不覆盖运行中服务的数据库或其他轮次。
`AML_EVAL_CONVS`、`AML_EVAL_LIMIT` 可设置样本范围；实验开关应作为唯一消融变量。

`AML_CHOICE_RECOVERY_BEFORE_SCOPE=1` 是默认关闭的阅读阶段实验：在裁剪未引用的
非个人前提前，先执行一次主前提引用恢复。它保留原有的引用校验和独立蕴含验证，
不会把找到引用直接视为支持。该开关与 `AML_CHOICE_RECOVERY_CLAIM_ONLY` 独立，
验证时应分别消融；单题回放改善不能替代完整 HTTPS 与独立样本验证。
当前对照未支持启用此开关：28 道固定上游样例两组均为 6/28，另一段完整阅读
对照从 16/26 降至 11/26。保留开关用于复现，默认流程不采用此顺序。
使用 `scripts/summarize_experiment.py --root <run>/completed` 检查分数与日志对应，
再运行 `scripts/audit_experiment_funnel.py --run <run>` 和
`scripts/audit_answer_gates.py --run <run>` 定位来源传递与回答门槛。
原文探针按用户隔离，回答审计按对话与问题 ID 隔离；来源标号重复或旧审计的
问题归属不唯一时明确报错，不跨用户借用证据。预先标注的相关段落探针只衡量
原文传递，不证明正确选项被语义支持；分数提升仍须另做同条件、跨对话对照。

已有数据库在打开时自动补建原文全文索引，旧敏感标签不会被自动清除。要验证新的
episode 写入和抽取回退行为，需要重新运行 Add／从原始数据重新评测。
證據包 hash 升至 v4，新增覆蓋需求 ID 與分數類型；舊 v1/v2/v3 包仍可讀取。

`SearchResponse.evidence_status` 使用 `retrieved` / `conflicting` / `not_found` / `incomplete`；
最後一項表示搜尋降級且沒有可交付證據，避免將失敗當成不存在。
`coverage_manifest.coverage` 和 `missing_evidence` 區分候選已找到、已裝包、尚未覆蓋；僅表示可觀測覆蓋，不保證語義充分。
普通檔案資料庫以唯讀 WAL 快照搜尋，向量快取按資料庫、使用者、空間、刪除 epoch 及更新代數隔離。
預設精確分塊 top-k；近似搜尋需設定 `AML_VECTOR_APPROXIMATE=1`，診斷會標示近似與掃描上限。
首次建索引、精確距離計算與歷史投影仍可能隨資料量增長；取消會等待背景工作停止後關閉快照。
九項修正、設定與驗證限制見 [SEARCH_PIPELINE_REPAIRS.md](SEARCH_PIPELINE_REPAIRS.md)。
`verification_status=not_run`；`retrieved` 不声明证据充分。旧的
`AML_SEARCH_MAX_ROUNDS`、`AML_CHOICE_ALIGN`、`AML_CHOICE_AUTOPICK`、
`AML_PERSONA_VIEW_FILTER`、`AML_CORE_PROFILE_INJECT`、`AML_CORE_PROFILE_TOKEN_BUDGET`、
`AML_SEARCH_MIN_RELEVANCE` 和 `AML_ABSTAIN_CONFIDENCE` 已停用，旧环境变量不再改变链路。
写入侧的来源校验、治理、敏感数据与版本/删除一致性约束继续执行。

遇到服务端限流（HTTP 429 / `RateLimitError`）时，所有模型调用默认额外重试
6 次，异步等待约 15、30、60、120、120、120 秒（带随机抖动）。若服务端
返回 `Retry-After` 秒数或 HTTP 日期，至少等待该时长。日志中的
`retry_delay_s` 显示实际等待时间。可通过 `AML_RATE_LIMIT_RETRIES`、
`AML_RATE_LIMIT_BACKOFF_SECONDS`、`AML_RATE_LIMIT_MAX_BACKOFF_SECONDS` 调整；
修改后需重启评测/服务进程。持续限流或配额不足仍会在重试耗尽后报错，
本地评测结果应检查 `error_stage` / `error_type`，避免将调用失败当作答错。
离线重试测试：`python scripts/selftest_metrics.py`。

单次 Search 有硬截止时间与调用/Token 预算（默认 45 秒 / 12 次 / 64k token，
`AML_SEARCH_DEADLINE_SECONDS`、`AML_SEARCH_MAX_CALLS`、`AML_SEARCH_MAX_TOKENS`）。
慢模型（如 qwen3-14b）叠加限流退避很容易超过 45 秒；超时的 Search 抛出
`TimeoutError` 并写入 `status=error` 的 trace，服务端返回 500，本地评测
把该题记为 `error_stage=search`、得 0 分后继续跑后续题目，不再中断整轮。
遇到超时请先调大 `AML_SEARCH_DEADLINE_SECONDS`，而不是重跑整批。

主动节流：同一进程、事件循环内，同类同模型的并发在途调用上限由
`AML_PROVIDER_CONCURRENCY` 控制（默认 4，设为 1 即串行）；同一请求内的多个
抽取分段会并发执行。默认最近 60 秒
最多发起 30 次请求（含失败尝试）；已返回的累计 token 达到 60,000 后，
后续请求等待旧用量退出窗口。分别通过 `AML_PROVIDER_RPM` 和
`AML_PROVIDER_SOFT_TPM` 配置，0 禁用对应阈值；日志显示 `provider_throttle`。
这是本地保守阈值，不代表服务商实际额度，也不预估下一次请求的 token；
请按账户额度留余量调整。其他进程/账户共享用量以及服务商的其他限流仍可能
触发 429，届时继续使用等待重试。修改后重启评测/服务进程。

Qwen3 对话模型默认开启思考模式，隐藏推理 token 会显著拖慢结构化抽取；
当 `AML_LLM_MODEL` 含 `qwen3` 时默认在请求体附带 `enable_thinking: false`，
设 `AML_LLM_DISABLE_THINKING=0` 可恢复思考模式。

DeepSeek V4.1 Flash 的 API 模型設定為 `AML_LLM_MODEL=deepseek/deepseek-flash`。
使用 `deepseek/` provider 時，同一開關預設傳送 `thinking: {type: disabled}`，
以支援結構化輸出的強制 `tool_choice`；開啟思考模式時，此強制選擇會被 API 拒絕。
`qwen3.7-text-embedding`／`qwen3.7-text-embedding-flash` 會傳送 `AML_EMBED_DIM`
指定的原生輸出維度，可維持 256 維而不在本地截斷向量。

文本数据集现已支持 **LoCoMo-Refined、LongMemEval-S、BEAM-100K、CLBench、PersonaMem-v2**，
并提供 ScriptMem/Refined 授权数据导入入口。下载、转换、评分差异及命令见
[DATASETS.md](DATASETS.md)。ScriptMem 原始剧本仍需另行提供；LoCoMo-Refined 已从公开仓库下载。

```powershell
python -m pip install -r requirements.txt -r requirements-eval.txt
python scripts/download_datasets.py all
python scripts/local_eval.py --data data/prepared/longmemeval-s.jsonl --inspect
```

`data/prepared/` 由 `prepare_dataset.py` 生成（本次已生成四套公开数据，完整转换
命令见上方文档）。本地结果为代理评分，不代表 AML 官方榜单分数。

启动时会自动读取 `memory_system/.env`（也支持当前目录下的
`.env`），已存在的系统环境变量优先，不会被文件覆盖。

```bash
pip install -r requirements.txt

# 离线契约自测（自给自足：自动启用 Fake 模式 + 独立临时库，无需任何 env/key）
python scripts/selftest_contract.py

# 启动服务（生产：配置真实模型，详见 .env.example）
export AML_API_KEY=<memory-system-key>        # 平台调用时的鉴权

# 方案 A：OpenAI（第一期默认）
export AML_LLM_MODEL=gpt-4o-mini              # litellm 模型串，可换
export AML_EMBED_MODEL=text-embedding-3-small
export OPENAI_API_KEY=<...>

# 方案 B：SiliconFlow
# export AML_LLM_MODEL=siliconflow/Qwen/Qwen2.5-7B-Instruct
# export SILICONFLOW_API_KEY=<...>
# export AML_EMBED_MODEL=siliconflow/BAAI/bge-m3
# siliconflow/<model> 会自动使用 https://api.siliconflow.cn/v1
# 和 SILICONFLOW_API_KEY；无需重复设置 API_BASE/API_KEY。

# 方案 C：阿里云百炼 DashScope（OpenAI 兼容模式）
# export AML_LLM_MODEL=dashscope/qwen3-14b
# export DASHSCOPE_API_KEY=<...>
# export AML_EMBED_MODEL=dashscope/text-embedding-v4   # 支持 dimensions 参数
# dashscope/<model> 会自动使用 https://dashscope.aliyuncs.com/compatible-mode/v1
# 和 DASHSCOPE_API_KEY。qwen3 系列默认附带 enable_thinking:false（百炼要求
# 非流式请求关闭思考模式），设 AML_LLM_DISABLE_THINKING=0 可改回。

uvicorn app.main:app --host 0.0.0.0 --port 8000

# 本地代理评测（真实打分需要 OPENAI_API_KEY）
python scripts/local_eval.py --data data/locomo_eval.json --convs 1 --limit 100

# 默认显示 Add / Search / Answer+Judge 实时进度条；重定向时可关闭
python scripts/local_eval.py --data data/locomo_eval.json --no-progress

# HTTPS 评测：远程 Add/Search，本地 Answer/Judge（同 local_eval 评分协议）
python scripts/https_eval.py --url https://localhost:8443 \
    --ca ../https_client/certs/server.crt \
    --data data/sample_eval.json --convs 1 --limit 5 \
    --output results/https_eval.jsonl

# 每次 LLM/Embedding 调用会输出 aml.metrics 日志：阶段、模型、
# attempt/retries、耗时和 prompt/completion/total tokens，不记录正文或密钥。

# 消融
python scripts/local_eval.py --data data/locomo_eval.json --convs 1 --limit 100 \
    --no-graph          # 去掉图谱 PPR 路
    # --no-governance  去掉治理（全部 ADD）
    # --no-rerank      去掉相关性过滤/重排
    # --no-keyexp      去掉查询理解扩展
```

`https_eval.py` 使用 `AML_API_KEY` 或 `--token-file` 访问服务；本地回答和裁判
仍需配置 `AML_LLM_*` / 模型供应商密钥。`--inspect` 仅校验和统计，不发请求。
数据集、分块、全局 `--limit`、证据预算和评分方式与 `local_eval.py` 一致。
HTTPS 始终校验证书与域名，不跟随重定向，不自动重试；公信证书可省略 `--ca`。
每次运行创建独立用户，写入后以 `min_revision` 检索，实验记忆保留在服务端。
写入失败终止运行；检索、回答或裁判失败逐题记零分并记录 `error_stage`，最终非零退出。
消融与远程模型配置需在服务端设置；客户端 `AML_FAKE=1` 仅控制本地评分。
结果中的 `fake`、`model`、`versions` 指客户端，服务端配置未由接口公开，记为未知。

`POST /feedback` 接受显式任务成败与轨迹，蒸馏最多 3 条程序性记忆；普通
对话不会推断任务结果。`DELETE /memory/{user_id}` 执行合规硬删除并返回不含
正文的 receipt。敏感记忆只有服务端开关与请求 `include_sensitive` 同时开启才可访问；
当前服务端开关按选定对照默认开启，`AML_SENSITIVE_RECALL_ENABLED=0` 可全局关闭。
完整正文调试日志默认关闭。

## 验证状态（2026-09-15）

- 契约自测 **11/11 通过**（含幂等重放、ID 逐字节回显、user_id 隔离、
  422、分块会话、无鉴权 health）。
- uvicorn 真实 HTTP 端到端冒烟通过（Fake 模式）。
- LoCoMo 真实数据管线冒烟通过（Fake 模式，1542 QA 可加载）。

## 待办（M4 前）

1. **真实跑分**：配置 `OPENAI_API_KEY` 后在 LoCoMo 全量上跑
   `local_eval.py` 及各消融组，出分项报告（Fake 模式的分数无意义，
   仅验证管线）。
2. 第二期开放（预计 2026-09-20）后核对 Full 前置清单，确认
   `AML_LLM_MODEL` 默认值。
3. 生产化：SQLite → Postgres+pgvector（`store.py` 同接口替换），
   Docker 化，公网 HTTPS 部署。
4. Fake 模式已知噪音：FakeLLM 词面匹配会误删部分候选
   （abstention 阈值在真实模型下正常），仅用于管线测试。

## HTTPS 调用日志

应用加载 `memory_system/.env`，已有进程环境变量优先；修改配置后需重启服务。
HTTP 请求日志默认写入 `memory_system/logs/access.jsonl`，可用 `AML_ACCESS_LOG`
指定路径。每条记录包含 UTC 时间、服务端生成的 request_id、方法、路由模板、
状态码、耗时和未处理异常类型；响应头 `X-Request-ID` 用于关联请求。
不记录 Authorization、查询参数或请求/响应正文。日志每 10 MiB 轮转，保留 5 份备份。
模型调用阶段、耗时和 token 用量仍通过 `aml.metrics` 写入进程输出日志。
当前后台 HTTPS 部署的访问日志为 `/root/aml-runtime/access.jsonl`，
进程和模型日志为 `/root/aml-runtime/https.log`。

## 平台评测部署清单（契约与数据处理）

Agent Memory Leaderboard 要求评测数据"仅用于本次评测、避免留存日志、运行结束后 30 天内删除"。
正式评测（smoke / full）前逐项确认：

1. **调试日志关闭**：`AML_MEMORY_DEBUG_LOG` 与 `AML_SEARCH_DEBUG_LOG` 保持为空（默认）。两者会写入完整记忆正文与检索响应，只允许在本地实验开启。
2. **访问日志仅元数据**：`AML_ACCESS_LOG` 只记录时间、request_id、路由、状态码、耗时，不含正文与 token；保留轮转默认值（10 MiB × 5），评测结束后清理。
3. **进程日志**：`aml.metrics` 只输出阶段/模型/耗时/token 数；不要把 uvicorn 的 `--log-level debug` 用于评测部署。
4. **数据删除**：评测结束 30 天内对评测 `user_id` 调用 `DELETE /memory/{user_id}`（会同步清理调试日志中的该用户记录），并删除 `AML_DB_PATH` 中残留的实验库或整库重建。
5. **契约字段**：Search 只会收到 `{query, options?, user_id, top_k}`；`reference_time/as_of/min_revision/evidence_token_budget` 不会出现。时序锚点回退为该用户最新消息时间（`AML_ANCHOR_FALLBACK`），证据预算走意图分档（`AML_EVIDENCE_BUDGET_DEFAULT` / `AML_EVIDENCE_BUDGET_BY_INTENT`）。
6. **返回内容面向 Answer 模型**：`content` 内的日期以文字形式双写（绝对日期 + 原始相对表述），被替换的版本带 `[outdated: …]`，用户规则带 `[user rule]`；平台忽略 `evidence_status/coverage_manifest` 等元数据。
7. **文档类输入**：平台不传 `source_kind`，服务端按结构/指令密度识别粘贴文档（`AML_DOCUMENT_*`），整段保留并按 `AML_DOCUMENT_ITEM_MAX_BYTES` 原文打包。
8. **选择题路径**：`/search` 中与选项相关的逻辑只在请求带 `options` 时启用；本地评测的 `answer_choice` 管线不参与线上服务（`local_eval.py --answer-policy platform` 可模拟平台 Answer 模型）。
9. **容量**：Full 评测含 BEAM（百万 token 级）数据。先用 `scripts/stress_capacity.py` 在目标机器上压测单用户大规模写入与检索延迟，再填写并发上限。
10. **模型规则**：开源方法赛道 Add 阶段须使用 gpt-4o-mini（`AML_LLM_MODEL`）；Search 侧模型是否受限以平台当期清单为准。

## 本地选择题实验（直接阅读器）

實驗開關 `AML_CHOICE_DIRECT_CATALOG_READER=1` 使用原始來源目錄和遺忘約束單次作答，
替代單選題的支持抽取、資格與獨立語義驗證流程，預設關閉。離線 100 題直接閱讀
與來源筆記均為 48/100，這一離線比較不等同於端到端收益。
cohort20 完整 HTTPS 為原流程 9/29、直接閱讀 14/29，但全部來源目錄均有變化；
另跑同一組 29 題的固定證據對照，兩組來源與約束目錄完全一致，結果同為 9/29
與 14/29（8 題改善、3 題退步），答案呼叫記錄為 184 與 29 次。
單次對照仍包含原流程 3 次證據候選校驗回退；直接閱讀亦重現第三方經歷被套用
到使用者的錯誤。cohort21 的完整 20 題前瞻固定證據對照為 8/20 與 10/20，
兩組合計 17/49 與 24/49，答案呼叫記錄 303 與 49 次；只覆蓋兩段對話，
且仍有 6 題退步。再補跑早期完整 26 題的雙組即時呼叫對照，結果為 18/26 與
8/26；三組總計原流程 35/75、直接閱讀 32/75（16 改善、19 退步），呼叫記錄
469 與 75 次。直接替換閱讀器不予採用，預設保持關閉。cohort21 已啟動的 HTTPS
實驗已完成：7/20 與 10/20，兩次寫入的全部 20 題來源目錄均改變，
此局部收益不推翻三組固定證據總體退步。總評見 `runs/https-cycle-20261011-direct-catalog-75qa-aggregate.json`。
結果見 `runs/https-cycle-20261011-direct-catalog-cohort20-fixed-reader-final-review.json`
及 `runs/https-cycle-20261011-direct-catalog-cohort20-final-review.json`。
來源角色與約束保留於提示詞，但此分支不保證語義支持或約束遵守；診斷明確記為
`selection_only_no_entailment_check`，漏斗記為 `S6:direct_selection`。

閱讀規則單獨消融（`scripts/probe_direct_reader_guidance.py`）固定來源與單次答案格式，
只添加證據使用指引，三組 75 題普通與加指引均為 31/75，3 改善、3 退步，
不予採用。完整校驗見 `runs/https-cycle-20261011-guidance-ablation-final-review.json`。

問題先行的來源刪選診斷（`scripts/probe_question_source_filter.py`）已完成三組
75 題完整成對對照。篩選呼叫只接收問題、原始來源及全部遺忘約束，最多返回
6 個原有來源 ID；答案呼叫保留這些來源的原文、欄位和原順序，以及全部約束。
對照組直接讀取完整目錄，兩組答案格式相同；篩選組多一次呼叫。
這與 `CHOICE_QUERY_CONTEXT` 在完整目錄旁重複提示來源的實驗不同。
`scripts/summarize_question_source_filter.py` 重建提示雜湊並驗證完整配對；
篩選或回答失敗仍以零分保留。本次無執行錯誤：三組分別為 9/26 → 11/26、
15/29 → 13/29、10/20 → 10/20，總計 34/75 → 34/75，8 改善、8 退步；
呼叫記錄由 75 增至 150 次。篩選會刪掉漫展計畫、北極研究經歷及旅行紀念物等
有用原文；部分得分改善仍依賴未獲支持的所有權或既往傷病推斷。不予採用，
服務預設值不變。這是固定證據的閱讀診斷，並非 HTTPS 端到端改進；完整結果見
`runs/https-cycle-20261011-question-source-filter-final-review.json`。

後續來源角色對照使用同一診斷腳本的 `--selection-mode first_party`：保留所有
user/persona 來源，僅移除 assistant 來源，原文、原順序及全部遺忘約束不變。
兩組均一次答案呼叫，沒有六條來源上限或語義篩選呼叫。三組 75 對已完成，
分別為 8/26 → 11/26、15/29 → 14/29、10/20 → 9/20，合計 33/75 → 34/75，
無執行錯誤；此微小增益不足以啟用。新 cohort22 的 HTTPS 基線為 8/24，完整
24 題固定證據前瞻對照為 7/24 → 8/24；四組總計 40/99 → 42/99，仍不啟用。
部分得分提升未有充分個人事實支持，部分得分下降反而避免了既往病史推斷。
完整分析見 `runs/https-cycle-20261011-first-party-reader-99qa-review.json`。
user 角色仍可能包含第三方故事、
轉貼及假設，不能當作個人事實驗證。進度見
`runs/https-cycle-20261011-first-party-reader-queue.json`；服務預設不變。

`scripts/inventory_https_prompt_budget.py` 可離線核對完整 HTTPS 對話的提示預算：
先精確重現每題原有精排候選、列表排序入選 ID 和提示位元組數，再測量同一候選
上限下的完整提示。cohort22 全部 24 題重現成功，17 題受到原 24,000 位元組
限制，最大完整提示為 33,384 位元組。QA579 的第一人稱編髮日記在 32,000
位元組下仍被刪除；35,000 位元組的兩輪診斷均恢復原文及該前提的支持判定，
得分由 0/2 升至 2/2。完整 24 題成對診斷已完成：8/24 → 8/24，3 改善、
3 退步，無執行錯誤；其中一題退步的初始排序提示完全相同，不能歸因於新增
排序證據。另有改善未恢復新的相關原文，不能全算作來源傳輸修復。
早期完整 26 題回歸為 16/26 → 15/26；兩組總計 24/50 → 23/50，
因此不採用全局擴大預算，也不啟動此候選的 HTTPS 處理組。兩組均無最終回答
錯誤，但包含已由修復流程處理的呼叫錯誤。總評見
`runs/https-cycle-20261011-budget35k-50qa-final-review.json`。完整組別結果見
`runs/https-cycle-20261011-cohort22-budget35k-final-review.json`。
這些固定檢索與 CE 的重播未計入上游已耗用的時間和 token，不能代替 HTTPS
端到端驗證。見 `runs/https-cycle-20261011-cohort22-579-budget35k-review.json`。

`scripts/probe_question_first_answer.py` 新增固定原文的閱讀診斷：先僅提供問題、
全部原始來源及遺忘約束，產生短擬答與來源 ID，再提供原本完整選項提示和該
可被修正的擬答。對照組直接選答；兩組交替執行，分別一次與兩次呼叫。
這是從 LongMemEval 閱讀／檢索分離得到的工程假設，並非論文直接驗證的方法。
不刪除來源，不將擬答作為證據；失敗保留零分且不回退重答。
`scripts/summarize_question_first_answer.py` 核對完整組別、提示雜湊、原始來源與
約束、選答結果和故障保留。26 對模擬驗證含缺欄位、超時、未知 ID，以及
七種篡改／不完整輸出拒絕檢查，詳見
`runs/https-cycle-20261011-question-first-validation.json`。三組 75 對真實診斷已
啟動，進度見 `runs/https-cycle-20261011-question-first-reader-queue.json`。
首組 26 對已完成：10/26 → 9/26，含兩次擬答格式失敗；部分得分改善仍帶有
比原文更強的個人事實斷言。分析見
`runs/https-cycle-20261011-question-first-reader-first-review.json`。其餘兩組為
14/29 → 13/29、9/20 → 9/20；完整 75 對合計 33/75 → 31/75，5 改善、
7 退步，擬答組含三次格式失敗。記錄的模型呼叫由 75 增至 147；新增擬答仍
不能防止最後選項帶入無依據的個人前提。因此不採用、不啟動此候選的 HTTPS
處理組，服務預設不變。總評見
`runs/https-cycle-20261011-question-first-reader-final-review.json`。

`scripts/probe_unspecified_attribute_scope.py` 對「依未指定喜好選擇」與「已知的具體
個人喜好／狀態」做 18 條語義控制、兩種順序的提示對照。補充提示的 36 次判定
中誤報 3 次、漏判 1 次；原提示為誤報 3 次、漏判 3 次。換序後仍會把未指定
價值觀當成事實、漏掉否定的所有權斷言，兩組也都錯判条件句。因此不接入
服務，詳見 `runs/https-cycle-20261011-unspecified-scope-review.json`。控制標籤未
送入模型，四次呼叫的提示、完整性和錯誤分類均重新核對。

cohort23 標準 HTTPS 基線完整完成：4/22，14 個 Add 區塊的 212 條原始消息
全部精確保存，無孤立來源，客戶端與漏斗審計均退出 0。18 題錯誤中有 12 題
基準相關原文已進入來源目錄；閱讀門檻記錄為 7 題選答、9 題獨立蘊涵驗證、
2 題支持提取，這些階段標籤不等同人工核驗的根因。QA600 的自述手術原文在
精排第 36 位被 fine16 截掉；QA601 同時存在把未指定價值觀建議當成事實、
以及藏書偏好實際屬於郵件收件人的問題。完整總評見
`runs/https-cycle-20261011-cohort23-final-review.json`。下一組依資料順序的
cohort24 已啟動（28 題、230 條消息），服務仍採標準設定。

`scripts/typed_entailment_prompt.py` 提供僅供診斷的提示契約統一：驗證器已使用
`personal_supported`、`no_personal_premise`、`unsupported_personal`，但正文仍有
舊式 `entailed=true/false` 指令。此轉換只替換這些指令，保留原有語義規則、
完整 checks 區塊和來源；未知模板變動會拒絕。`scripts/replay_typed_entailment_prompt.py`
先離線精確重現完整讀取器，再固定驗證前的模型響應，比較原提示與統一提示的
下游結果。cohort23 全部 22 題已通過離線回放，完整成對真實診斷已啟動。
`scripts/summarize_typed_entailment_prompt.py` 核對順序、原文、固定響應、实际提示
及分數；五種篡改拒絕檢查已通過。進度見
`runs/https-cycle-20261011-typed-prompt-cohort23-reader-summary.json`。完整 22 對
已完成，4/22 → 4/22，兩組各 43 次下游呼叫、無最終回答錯誤；關鍵的
價值觀建議範圍及自述外交談判投入仍被誤判。未採用此提示改動，總評見
`runs/https-cycle-20261011-typed-prompt-final-review.json`。此重播使用擴大預算
且不計前序歷史耗時，不是 HTTPS 處理組的準確率。

`scripts/probe_citation_highlight.py` 與 `scripts/summarize_citation_highlight.py`
完成 22 對驗證器呈現對照：只移除重複的短引文欄位，保留完整原文、角色、時間
和全部檢查項。44 次呼叫無錯誤，15 項判定改變，但 QA609 的全部判定未變，
QA602 更把三條具體個人偏好斷言誤判為無個人前提；不採用，亦不聲稱最終
答案準確率改善。總評見 `runs/https-cycle-20261011-citation-highlight-final-review.json`。
後續代碼與目錄核對發現 QA609 的職業前提已有一條正向引文，因此跳過補充來源
搜尋；同目錄中 s4、s6、s18 的直接談判經歷未納入該前提的驗證引文。這是下一項
待驗證假設，尚非已實證修復，見
`runs/https-cycle-20261011-cohort23-609-citation-gap.json`。

`scripts/probe_post_verification_citations.py` 掃描 cohort23 全部題目，選出 7 題、
11 個「初始支持且有引文、其後驗證拒絕」的前提，做兩輪補充來源診斷。全部
14 次搜尋提議都給出不支持，原支持旗標因此阻止附加引文，前提判定無變化。
`scripts/probe_post_verification_proposal_gate.py` 再使用同一批已存提議，只讓
精確原文與第一方主體校驗決定是否送交獨立驗證。28 次對照呼叫無錯誤，只有
QA603 兩次由「無個人前提」改為「不支持」；沒有任何目標變為支持。QA609
兩輪都補入 s4 的直接談判經歷，仍被拒絕，因此來源補充缺口並非充分解釋。
兩個審計腳本重建全部提示和新增來源；篡改來源、判定、提示、順序均被拒絕。
不採用、不啟動 HTTPS 處理組，總評見
`runs/https-cycle-20261011-post-verify-final-review.json`。這些是前提判定診斷，
並非最終答題準確率；有效引文不自動授予語義支持。

HTTPS 漏斗新增 `full_user_turn_coverage_v2`：保留舊片段命中計數，但以完整
使用者輪次的字面覆蓋決定送達標籤；部分命中和無法對應原始消息的片段分開
呈現。同一原始消息的片段可合併覆蓋，重複片段不得重複計數。另記錄助手
輪次覆蓋與兩種角色的字符數，避免將「用戶請求润色」的送達誤解為個人經歷
已獲支持。cohort24 QA621/623 的 CSV 用戶片段本來就只有短請求，具體經歷
出現在助手產生的原始／潤色邮件中；初判為用戶正文截斷已被推翻。
此修改不改動答題服務。新增覆蓋控制、原傳輸控制、14 個漏斗測試及 6 個身份
測試通過；舊 `selftest_funnel_repairs.py` 的 9 失敗、3 錯誤已在改動前冻结
代碼中復現。詳見 `runs/https-cycle-20261011-funnel-coverage-review.json`。

cohort24 標準 HTTPS 完整完成：6/28，230 條消息精確保存，客戶端及審計均退出
0。總評見 `runs/https-cycle-20261011-cohort24-final-review.json`；下一組按資料
順序的 cohort25 已啟動（28 題、235 條消息），設定不變。

`scripts/probe_recovery_source_order.py` 先精確重現 cohort24 全部 28 題，再對
所有 16 個引文補充批次比較原順序與逆序，來源文字、角色、ID、指令及輸出
契約不變。16 對無執行錯誤，29 項提議改變，但 QA636 的自述搖滾演出來源仍
未獲引用；QA623 還出現無使用者手術自述卻提議支持的變化。因此不採用簡單
逆序，也不聲稱答題得分改善。審計腳本為
`scripts/summarize_recovery_source_order.py`，總評見
`runs/https-cycle-20261011-recovery-source-order-final-review.json`。

`scripts/inventory_snippet_provenance.py` 對六組已完成對話的 149 題做離線角色／
格式盤點，核對 CSV 問題、答案及歷史身份，篩出「用戶只有短編輯請求、助手
帶 Original 文件標記」的 3 題。QA10 的正確答案本身是通用穿搭建議且基線
答對；QA621/623 的具體經歷只在助手草稿中出現，後者隨後也沒有使用者確認。
此狹窄模式不能解釋整體低分，更不能據此放寬來源歸屬規則。格式篩查不是
語義判定、完整歸屬錯誤清單或準確率上限；基準標註不進入運行時。
總評見 `runs/https-cycle-20261011-snippet-provenance-final-review.json`。

`scripts/probe_full_history_reader.py` 以相同來源目錄、約束與指令，比較檢索內容與
額外完整原始歷史的直接閱讀。預先指定 cohort24、first 共 54 對，交替呼叫
順序，不截斷歷史、不加入 QA 標註。新增歷史也含更正與遺忘請求，因此不是
純事實補充或 oracle 上限，亦不能直接當成 HTTPS 準確率。兩組最大提示詞
約 209 KB。`scripts/summarize_full_history_reader.py` 重建提示詞及完整身份，
離線整合測試保留超時／缺欄位失敗並拒絕五種篡改。結果尚待完成，預設不變；
啟動記錄為 `runs/https-cycle-20261011-full-history-queue-v3.json`，前兩次隊列
啟動語法錯誤發生在任何模型呼叫前，原始腳本與日誌保留。

完整歷史對照 cohort24 已完成 28 對：檢索內容 6/28、增加歷史 8/28，
3 勝 1 負且無執行錯誤。QA633 攝影有明確新增使用者原文；QA624 個人
病史、QA631 瑜伽墊所有權仍缺乏清楚支持。QA636 既有搖滾證據仍未利用。
QA642 有遺忘約束卻由金標 B 改選 A，但 A 本身沒有斷言個人抹茶偏好，
不能只憑失分宣稱違反遺忘。first 組仍待完成，預設不變。詳見
`runs/https-cycle-20261011-full-history-cohort24-review.json`。

完整歷史對照兩組 54 對已完成：合計 18/54 → 18/54，6 勝 6 負，
108 次記錄呼叫，無錯誤。first 組為 12/26 → 10/26；完整歷史仍未選出
已有明確使用者原文的學生傾訴與衝浪答案，若干得分提升也依赖較弱的個人
前提。因此不採用完整歷史注入，不聲稱 HTTPS 改善。總評與逐題限制見
`runs/https-cycle-20261011-full-history-final-review.json`。

`scripts/inventory_source_admission.py` 對六組 149 題追蹤完整使用者原文的
粗排、精排、重排准入與選擇，核對輸入、封包及標註身份。19 題最遠停在
精排、2 題停在重排准入；67 道錯題已有完整使用者片段。這是文字傳輸
盤點，不證明答案成立，亦不將片段或助手文字誤算成完整使用者原文。
11 項合成檢查通過。cohort25 QA649 的補水郵件粗排31、精排44，被16條
上限排除，尚未進入重排；不是提示詞位元組超限。相關報告為
`runs/https-cycle-20261011-source-admission-inventory.json` 與
`runs/https-cycle-20261011-cohort25-649-admission-review.json`。

`scripts/probe_option_prefix_admission.py` 對 149 題精確重現原精排，測試
僅以選項開頭個人條件做主題預留；5 項邊界檢查、輸入不變性與元資料
保留通過。3 題列表改變，完整片段覆蓋107→108，唯一增加QA528是同事
安排露營的郵件，不能證明野生動物驚嚇及焦慮。QA649仍未恢復：水瓶
自述虽跨過詞面閾值，第三人稱「有些人喝水少」問題仍獲得名額。因此
不採用此方案，也未聲稱答題改善。詳見
`runs/https-cycle-20261011-option-prefix-final-review.json`。

cohort25 中期跨例複核：QA656 的泛視力討論、QA665 的泛補水建議佔據
主題名額，而原始個人情境被精排排除。但 QA661 雖漏掉基準指定片段，
另一組北歐犯罪小說提問 s4/s6 已送達且補引成功，仍被獨立閱讀驗證拒絕。
故不能用指定片段漏失代表所有可用證據缺失，也不能一律降低疑問句權重。
來源、引文及驗證回應見 `runs/https-cycle-20261011-cohort25-crosscase-review.json`。

cohort25 HTTPS 完整完成5/28，235條原始消息精確保存；客戶端與審計退出0，
服務器正常終止。23道錯題的片段傳輸為已送達12、封包前8、未召回3。
QA661完整凍結重放確認：引用及明確的興趣提問規則均已在驗證輸入，仍遭
拒絕。總評 `runs/https-cycle-20261011-cohort25-final-review.json`。
下一組cohort26按數據順序啟動，26題、226消息、15塊，保持原設定。

`scripts/probe_verifier_assessment.py` 對 cohort25 完整凍結重放後，比較原標籤
驗證與額外提供180字元以內證據依據／來源ID的驗證。原證據、系統指令、
3072輸出上限不變，交替執行兩臂；格式錯誤保留，不重試。
`scripts/summarize_verifier_assessment.py` 核對完整批次、原提示詞雜湊、
身份及判斷；來源ID存在不代表理由成立。結果待完成，預設不變。
記錄：`runs/https-cycle-20261011-verifier-assessment-cohort25.jsonl`。

證據依據驗證對照已完成28對、56次呼叫，無格式错误，16項標籤改變。
QA661的小說興趣仍被拒絕，依據文字卻承認同一興趣；QA649將已有个人
前提改成無個人前提，QA670又要求未來建議活動有歷史證明。未測最終
答題分數，不採用此版本。總評：
`runs/https-cycle-20261011-verifier-assessment-final-review.json`。

`scripts/probe_boolean_verifier.py` 以完整28個既有驗證批次比較三分類標籤與
兩個布林值：是否斷言個人事實、是否全部有支持。兩臂使用相同格式正規化，
檢查物件／來源／指令政策／輸出上限不變；三種合法組合映射為原標籤，
false/false視為不一致並保留失敗。這是輸出編碼診斷，尚非答題改善。
審計腳本為 `scripts/summarize_boolean_verifier.py`；預設不變。

布林驗證對照28對完成，16項標籤改變、無格式錯誤。新增下游追蹤腳本
先精確重現28題基線，再注入固定驗證輸出；僅變化後的路徑做17次實際
呼叫，其餘依提示詞雜湊重放。56條完整路徑再次離線核對通過，條件得分
5→6，唯一提升QA655來自將有支持的另一選項D降級為partial，並非修復
A的證據；QA661仍失敗。因此不採用、不聲稱HTTPS改善。總評：
`runs/https-cycle-20261011-boolean-verifier-final-review.json`。

`scripts/repeated_topic_interest.py` 是尚未啟用的狹義興趣規則候選：完整
主題需出現在兩條不同使用者提問，引文原文與來源ID保留；不外推所有權、
診斷、頻率或專業資格。12個反例與4組獨立主題對照通過。完整cohort25
固定響應回放5→6，唯一QA661從B變C，2次下游實際呼叫，全路徑重放
審計通過。這次提升符合原始小說興趣證據；仍需跨組驗證，非HTTPS改进
結論。cohort24回放中，cohort26預先保留作完整後續檢验。詳見
`runs/https-cycle-20261011-repeated-interest-cohort25-audit.json`。

狹義興趣候選v1在完整引用校驗後仍誤接納翻譯任務、考題抄寫、明示zero
interest的引用，故拒絕啟用。v2要求引文就是完整直接提問，排除任務用途
和否定語境；保留v1原始記錄。新語境控制零誤接納、原12反例仍通過，
cohort25重新回放及審計仍5→6、只有QA661改正。這仍是開發樣例收益，
cohort26完整後續驗證改用v2，預設保持不變。詳見
`runs/https-cycle-20261011-repeated-interest-v2-review.json`。

狹義興趣v2的獨立完整驗證：cohort23為4→4（22題）、預先保留的
cohort26為10→10（26題），兩組均無規則觸發、無新增模型呼叫，完整回放
審計通過。收益仍僅見開發組QA661，不啟用預設。cohort26基線完整
10/26、入庫226條/15批；16道錯題中11道原始使用者片段已送達，
4道在packet前丟失、1道recall丟失。閱讀漏斗為選擇8、獨立驗證5、
支持2、資格1；不把片段送達當成個人前提成立。詳見
`runs/https-cycle-20261011-cohort26-final-review.json`。
等待隊列曾在最終manifest寫入前提前檢查而停止；核對完整成功狀態後
僅重啟對照，原基線未重跑，事件保存在queue-race.json。
下一完整cohort27已按資料順序啟動：23題、244訊息、15批，預設不變。

第26組閱讀範圍診斷發現QA695將「cooking something you enjoy」
當作既有烹飪偏好。新增診斷候選 `scripts/indefinite_advice_scope.py`，
只處理不定對象的建議表述，保留完整選項驗證；5正例/9反例通過。
完整26題固定響應回放10→11，僅QA695改正，3次下游呼叫與完整審計
通過。完整cohort25為5→5，未觸發。第24組和第26組重複驗證仍在跑；
預設不變，不聲稱HTTPS提升。候選也改變下游其他選項的資格，需區分
穩定的範圍修正與模型重抽樣效果。詳見
`runs/https-cycle-20261011-indefinite-advice-review.json`。

不定對象建議候選的第26組重複回放仍為10→11、3次下游呼叫；
cohort24完整28題6→6、零觸發，審計通過。第27組完整23題已預先
指定為後續檢驗，隊列等待包含client/audit/server終態的manifest，
避免前一等待器過早讀取中間狀態的問題。

不定對象建議v1新增語境測試暴露誤刪歷史前提的風險（remember /
recall / helped及bought尾隨限定），因此拒絕啟用。v2改為要求直接
建議補語或連貫例示清單，拒絕歷史與中間個人限定；原14項與新增7項
語境測試通過。第26組v2完整回放仍10→11，僅QA695改正、3次呼叫。
第27組對照隊列在答題前改為v2，保留被取消的v1隊列記錄；基線不重跑。
詳見 `runs/https-cycle-20261011-indefinite-advice-v2-review.json`。

第26組QA701顯示閱讀階段的引用目標混淆：s13已有明確教學/研究
自述，恢復步驟卻給相同背景前提引用戶外休閒提問。完整26題精確回放
後盤點12批恢復請求，只有701有完全相同的重複前提。新增診斷比較
原始、去除option、去除option且合併重複前提，三臂各2次，共6次
無錯誤，提示/來源/展開映射審計通過。三臂仍把背景判為interest，
去重新增s12也只引用休息提問，未選s13；不啟用，不聲稱答題改善。
詳見 `runs/https-cycle-20261011-grouped-recovery-final-review.json`。

第26組4處packet前丟失均在fine容量：679(rank19)、682(rank30)、
686(rank20/37)、696(rank27)。其中679的朋友拜訪描述來自assistant
代擬郵件，不能當作使用者偏好；686則明確為使用者自作板上移動文字。
新增完整可見user原文去重診斷，只改冗餘覆蓋，不改topic預留、分數、
profile區隔、時間狀態相容或圖關係要求。9項控制通過；完整8組203題
原fine名單精確重現，候選改變38題名單，但完整片段覆蓋139→139、
零新增/零丟失，未恢復第26組焦點來源，故不啟用或進一步聲稱答題收益。
詳見 `runs/https-cycle-20261011-visible-source-dedup-final-review.json`。

第26組QA699補查：原始user提問完整入庫，但不在ranked pool；已選
的是相關assistant解釋。user只問暖季湖岸/海岸社區的社會凝聚力，
游泳由assistant提出，不能推出使用者游泳偏好。gold本身亦只是活動
建議，因此恢復該snippet不能直接歸因為偏好修復。保留召回傳輸失敗
標記，另記語義歸因限制，不為命中標籤而放寬使用者證據標準。詳見
`runs/https-cycle-20261011-cohort26-699-recall-review.json`。

建議範圍v2首組26題完整回放15→15、零觸發、審計通過。新增
`scripts/inventory_indefinite_advice.py` 在8組203題完整凍結閱讀路徑
逐次核對提示與結果，候選僅觀察不施用、零provider呼叫，唯一觸發
仍是開發QA695。這提供有限的無干預證據，沒有獨立正向收益，不能
宣稱整體準確率提升。第27組244訊息/15批已完整入庫，原文一致、
無孤立來源，已進入答題；完整後續v2對照等待基線終態。

第27組基線完整完成8/23，244訊息15批原文核驗通過，client/audit
退出0、server正常終止。15道錯題：閱讀漏斗獨立驗證9、選擇4、支持2；
片段傳輸為已送達8、packet前丟失6、原輸入無標註片段1。
預先固定的建議範圍v2完整對照8→8、零觸發/零新呼叫，審計通過；
僅開發QA695有收益，預設不變。詳見
`runs/https-cycle-20261011-cohort27-final-review.json`。
第28組已自動啟動19題、293訊息16批，執行中設定與計畫白名單一致。

第27組packet丟失細分為錯題fine容量5、coarse容量1；額外正確題715
也有fine丟失，因此傳輸不等於答題因果。703無beer-over-wine比較、
711拒稿敘述屬assistant代擬，不能以恢復片段當作個人前提修復。
714自述鋼瓶常放包內、717繪圖板/螢幕工作則有明確相關使用者來源，
分別fine23與coarse50前被排除。完整23題前綴預留重測14→14片段
覆蓋，仅1題名單改變，沒有傳輸改善，預設不變。詳見
`runs/https-cycle-20261011-cohort27-admission-review.json`。

完整可見user原文用於option-topic預留（不是去重）對照226題，
原fine基線全部精確重現，27題名單改變、36題連同預留元資料改變，
片段fine覆蓋153→155，新增669/709、無損失；6項邊界控制通過。
第27組下游對照需包含僅預留改變的708/713，加上709/719，共4題
雙臂fresh ranking/reader，其餘完整19題基線精確重放。64次呼叫無
錯誤，完整得分8→8、無預測變化，全部46結果與packet哈希及呼叫
提示重放審計通過。初版漏掉預留元資料變化的8→9不作完整結論。
審計對無序預留元資料做規範化，候選順序、語義欄位仍嚴格核對；
原失敗記錄保留。第25組完整28題繼續驗證另一新增來源669，預設不變。
詳見 `runs/https-cycle-20261011-visible-topic-review.json`。

完整原文topic預留第25組28題下游對照完成6→5，93次呼叫無錯誤，
完整提示/packet/結果重放審計通過；共6題名單或預留元資料改變，
其餘22題精確重用。與第27組合計51題14→13、157次呼叫，無可靠
正向收益，不啟用。唯一配對分數損失654為原方案重新取樣C正確、
候選B錯誤；原HTTPS亦是B錯誤，因此不聲稱穩定回退。669新增來源
確實通過fine/listwise且入選，但兩臂仍選C；原文只談低糖烹飪，
未建立家族糖尿病史，完整歷史相關問句亦為假設情境。片段覆蓋增益
不等於正確個人前提或答題收益。詳見
`runs/https-cycle-20261011-visible-topic-review.json`。

### 2026-10-11：第28组与后置当前问题来源诊断

第28组完整19题得6分；293条消息、16个写入块逐字核对通过，客户端与审计退出0。13个错误中，来源片段已送达10题、包选择丢失2题、召回丢失1题；读取端独立验证6题、支持判断1题、最终选择6题。第29组32题已按原配置启动。

后置当前问题来源只在引用恢复调用时加入临时q0，完整第26组26题10→10，34次调用、离线逐提示审计通过，未启用。第694题明确当晚经历仍被恢复器拒绝；v2仅补充明确自述与假设/询问的区分，继续全组诊断。当前问题不持久化，仍须引用与独立验证。

后置当前问题来源v2完成第26/28组45题：16→15，75次调用，两组逐提示离线审计通过。727通过q0精确引用与独立验证修复；694恢复器越界支持调查旅行细节，独立验证全部拒绝；698与742退步。未采用。另第28组约束配对审查定位libraries被简化为librarie、theater/theatre不一致，正在测试仅词形规范化的完整19题诊断；语义约束判定与既有保护保留。

词形规范化诊断：第28组完整19题6→8（731、739），4次新调用；第27组完整23题8→8，0次新调用。两组离线逐提示审计通过。245题完整配对库存只有731/739新增两对，无删除。两项收益都来自开发会话，尚未推广；其余历史组完整读取回放与审计继续运行，待第29组结束后做前瞻验证。

词形规范化现已接入app/constraint_term_forms.py，通过AML_CHOICE_CONSTRAINT_TERM_FORMS开启，默认0；配置进入运行元数据。运行实现与诊断实现1929段文本逐项一致，86项答案模块测试通过。已启动完整第28组新数据库HTTPS双臂（18444关闭、18445开启），不将既有回放收益当作端到端结果；第29组终态成功后另排队做完整前瞻回放及审计。

词形规范化全部10组245题完整回放结束：78→80，仅731与739发生预测变化；4次新调用、无提供方错误，全部原始控制与逐提示审计通过。其余9组未触发新的调用，不能将零触发组视为独立收益复现。端到端双臂和第29组前瞻验证仍待完成。

目的从句只读调查：245道完整历史题加第29组当时已完成10题，共255题，发现18条紧随so/so-that的原始支持声明；本地is_suggestion未识别15条。已补充最终保留/移除与资格信息，避免把本地未识别等同最终失败；3条历史格式异常单独记录。747将预期安全感截为既有事实，但其他从句可能包含原有习惯等预设，尚未添加运行规则。

目的从句语法候选在3正例/10负例基础检查通过后，被额外反例否决：建立提醒以知道自己有兄弟/银行账户，以及旧伤疼痛，仍有事实预设；规则却将其当作非事实目的。已标记REJECTED，0次提供方调用，未用于读取回放或运行代码。后续需保留嵌套事实的从句分解，不能统一删除so从句。

第29组749/756数据缺口归因复核：两题历史关联片段不在准备数据，但当前问题已提供对应信息，不能直接归因召回。749的金标B通过泛化/独立检查后，被忘记社区花园志愿者经历的约束误拦截；使用场地不等于志愿服务。C虚构地址却被视为泛化，是另一事实校验失败。756四项均合格，选了完整路径而金标脱敏。263题快照中18道敏感信息题得8分，10错题中6道金标合格、4道不合格；标签仅用于离线分析，未输入运行决策。

志愿服务约束动作区别诊断已启动：只向包含第一人称volunteer约束的既有语义判定提示补充“服务行为不等于场地使用”，配对和验证流程不变。第21组完整20题7→7、0触发调用，逐提示审计通过；不能据此声称新提示有效。第29组完整回放已排队；另外用场地参观/租用、除草帮助、恢复志愿服务、避免无偿工作等6个语义边界案例做原提示与新提示双重复对照。

志愿服务边界对照完成：6个自建案例各重复2次，原提示两次均误禁止租用花园作摄影工作坊；候选两次均正确允许，同时仍禁止帮忙除草、维护和恢复志愿服务。4次提供方调用、无错误；提示SHA、案例全集与判定统计审计通过。此为开发边界证据，不是基准准确率提升。

第29组完整32题结束，9分，235条原消息/15块逐字核对通过、无孤立来源，客户端0/审计0/服务正常结束(-15)。第30组25题已启动并核对运行配置，约束词形开关显式0。词形规范化及志愿动作提示的第29组完整回放均已自动开始；新HTTPS双臂仍在运行。

第29组完整候选回放完成并审计：约束词形9→9，0新调用，累计277题87→89，收益仍仅开发组731/739。志愿动作提示9→9，11次新调用；749仍误拦截金标B，并额外拦截C，错误C→错误A。自建边界用例的成功未迁移到完整上下文，未采用。764则确认原文洗衣气味正向体验载体精排第12名通过后，未进入24k预算的列表重排；既有增预算/共享来源策略历史回归，暂不重启相同改法。

结构化动作证据诊断完成：第29组32题冻结预检一致，全部5个志愿约束批次，两臂各重复2次，共20次调用、无提供方错误，离线提示/模式/映射审计通过。第749题金标B两次仍被误判same_action；30条候选判定中22条引用整段选项，引用逐字有效却未实现动作定位。该版本未进入答案回放、未采用。HTTPS双臂两侧293条原始消息均完整核对通过，评分继续。

词形规范化HTTPS关闭组完整19题7分、终态审计0；开启组当时16题8分，731/739两题均修复，但尚未完成。已排队在双臂完整成功后联合审计，再分别对两套固定证据包反转同一开关（关闭→开启、开启→关闭），逐题隔离新写入/召回变化与开关效应；默认仍0。

词形规范化采用决定：完整新HTTPS19题关闭7→开启9，联合审计通过；两臂19个来源目录均不同。固定关闭组证据包开启为7→9（4次调用），固定开启组证据包关闭为9→7（3次调用），逐提示审计通过，变化均仅731/739。结合277题历史回放87→89，下一组31按数据顺序完整运行并显式设AML_CHOICE_CONSTRAINT_TERM_FORMS=1；应用默认仍0，保持旧日志回放兼容。收益仍来自同一开发会话，后续需独立触发验证，不能当作跨会话普遍提升。

漏斗诊断新增遗忘约束细分：保留reader_gate主分类，只在一般资格失败且同时有gold被阻止记录和有效正向约束匹配时，reader_gate_detail标为S5:forget_constraint；错误引用、否定判定、其他选项匹配均不计入，早期验证失败保持优先。17项漏斗单元检查及运输分类检查通过；第29组32题主分类全部不变，仅749从一般资格细分为约束拦截。单独归档复核文件，未覆盖先前最终漏斗。第30组完成后已排队完整固定包开关回放，检查新会话回归。

遗忘拦截优先级复核：277题190错题，8道金标有有效匹配且被记录阻止。571/722/724/744金标分别推荐或断言手写清单、常去本地食材市场、已整理数字文件夹、既有黑巧克力储备；历史有对应明确forget，当前问题未重新陈述偏好。后续关键词扫描未发现再次提及（非语义穷尽证明），不能为了金标放松遗忘约束。仅有拦截记录并不等于模块错误；降低继续提示微调优先级，关注更广泛来源准入与个人事实读取问题。

第30组778的读取归属诊断：金标含“我也发现可重复使用的清洁布……有效”通用方法表述，被提取成用户个人事实并用环境友好提问作引用，独立验证拒绝。288题快照含29条第一人称声明、12条最终保留，不能一概当成回答者口吻（还包括自述工作偏好/丧亲经历）。诊断候选只覆盖独立句“I have/’ve found that”后的通用表述，排除人称、姓名、特指主体、历史和引用代写；3正例10负例通过，当前仅命中778三项相同表述。已排队完整第30组25题回放/审计，独立选项验证保留，未改运行默认。

第30组完整25题9分，226条消息14块精确写入，运行/审计均通过。词形修复固定包开关完整回放9→9，0新增调用、逐提示审计通过，属于未触发的回归检查。第一人称通用观察候选完整25题9→9（4次调用），778金标资格恢复但仍选C；整体验证还把B“下一次食谱拍摄”判为无个人前提，需要复核，候选未采用。第31组完整28题已启动，词形修复显式开启，运行配置核验一致。

第30组全25题来源粒度精排诊断完成（382次CE调用）：全部原粗排候选和查询固定，整块评分前16完整用户轮次覆盖16题，全部来源最大分仍16题，用户来源最大分降至14题；用户模式新增784/791、丢失777/778/788/798。797漫画收藏承载块31→26仍落选。审计逐项核验查询、文本、来源角色、候选顺序及聚合，且拒绝查询/角色/重复题目三类篡改。未做下游答案回放、未改运行默认；排名覆盖不等于语义支持或最终准确率。

第30组chunk7旧上下文消融完成：全部4段、两臂各重复2次，共8对67调用。审计确认仅抽取提示的旧摘要/局部前文改变，新消息及后续引用/语义校验保留；拒绝提示篡改、伪造引用、重复样本三类错误。候选两次都提及farm box但全部四个焦点输出仍漏漫画收藏，且一次把“考虑制作”强化成“计划制作”。新增事实数量不代表质量，不全局删除抽取上下文、不持久化诊断事实；没有下游准确率收益结论。

第31组803暴露支持抽取将“Given your love for Arctic cruises”整项判generic且无声明；已有因果前提保护只能保留已提取声明。307题快照19个明确兴趣前缀中3个最终无声明。诊断候选仅在有效generic且无声明时补入原文前缀为unsupported，仍经过引用恢复与独立验证。9项正负边界及不可变/无引用/未解决条目保持检查通过；第20组完整29题9→9（4次调用），逐提示审计通过，541B从资格集合移除但答案仍错。已排队第31组完整28题，不修改默认，不视为准确率提升。

第31组810四个选项均因未指定种类的“your favorite snacks”声明而无法获得支持，最后弃答。315题快照38条favorite/preferred声明含多种真实类别/历史限定，不能统一豁免。诊断候选仅识别明确建议中的未指定零食对象，排除具体类别、历史、来源限定和引用代写；11项边界检查通过，当前仅命中810四项，完整选项验证保留。已排队完整第31组28题并等待兴趣前缀回放结束，以限制并发日志加载；尚无分数结论，未改默认。

第31组完整28题7分（25%），234条消息15块精确写入，client/audit=0、server=-15。21错题：读取选择8、支持8、独立验证5；相关片段运输到目录10、包内丢失8、原始数据无对应片段2、召回1。词形规范化显式开启，但单组分数不能证明开关效果，完整反向开关回放另行审计。803旅行邮件已进入列表却被判无关，恢复引文同时错来源ID、把原逗号改句号，没有逐字匹配可供单纯重定位；保留校验，未放宽事实要求。

第31组后续回放均审计通过：词形开启→关闭7→7、0新增调用（未触发）；明确兴趣前缀补回7→7、4调用，803B/C恢复声明后缺乏支持，选择仍D；零食建议作用域7→7、4调用，810从弃答变错误C。兴趣候选两组57题合计16→16，两个新候选均不采用。来源准入完整28题：目录14、精排5、粗排1、候选池无完整原话1、列表准入1、列表选择1、无原始用户轮次5。826航天问题是窄兴趣的合理候选证据；822家事、823糖尿病分型、827期刊阅读习惯均强于所列原话，不能将恢复片段等同金标得到证明。

第31组826精排诊断发现此前q240“逐选项”请求仍保留原查询中所有选项前提，未真正隔离。新诊断scripts/probe_ce_isolated_options.py对完整28题固定粗排候选及原文，将每个分支改为当前问题+单个完整备选答案，控制臂保留原组合查询；轮询合并名次并按题号位置轮换起始选项。审计将复现原精排并应用真实保留名额/容量，避免只报告理想前16名。该诊断扩展调用预算，未满足现有线上预算；尚无排序或准确率结论，未改默认。

第31组真正逐选项CE诊断完整28题已审计：原精排与新控制逐题一致，按实际保留名额计算完整用户轮次覆盖16→18，仅805/823新增、无丢失；826航天载体合并后第24仍落选。控制225次CE调用、候选946次，约4.2倍，当前方案不符合线上调用预算。已启动完整28题两臂新列表重排及答案回放，冻结包28/28预检一致；尚无完整下游分数，不采用。

第32组完整19题8分，228条消息15块逐字写入，client/audit=0、server=-15。开关反向回放8→8、0调用，逐提示审计通过。11错题按原话准入分为精排4、粗排1、候选无完整原话1、原数据无关联轮次1、已送达4。832偏头痛前提被作用域判断移除后误成generic；831实际因相机/日记前提失败，恢复自然散步原话不保证修复。830助手代写、844他人游记、848泛化医学提问不能直接证明用户属性。第33组22题已启动并核对显式词形开关为1；没有为金标放宽归属或事实校验。

下游CE对照审计器的合成检查通过：有效双臂样本可复现，提示/阶段/缺臂/顺序/包哈希/准入/资格/分数/多余调用9类篡改全部拒绝。新增scripts/analyze_ce_reader_transport.py，在完整逐提示审计成功后自动核对精排、列表准入、选择及读取目录的原话覆盖；两对真实记录的结构预检通过，但不代替整组审计。

第32组832句中因果前提问题只读库存：349道完整历史题中，仅一处scope_not_personal删除的声明能被“句首因果保护应用到后续句子”找回。涉及偏头痛诊断，不能从熬夜头痛直接推出；未扩大句子切分保护或更改默认。第31组精排下游局部复核803/805已将相关原话送入列表选择，但金标兴趣前提分别被独立验证拒绝/保留unsupported，答案均未修复；整组仍在运行，不据局部结果作采用结论。

真正逐选项CE方案完整下游对照结束：28题两臂各7分、逐题无得失分，443次下游调用完整提示/包/准入/资格/答案审计通过，无提供方错误。相关原话覆盖按精排16→18、列表准入15→17、列表选择14→17、读取目录14→17；新增803/805/823均未答对。控制225次CE调用、候选946次（约4.2倍），未采用、不改默认。证据运输改善不等于个人前提得到证明，且826航天原话仍未进入精排。第33组221条消息12块全部逐字写入且无孤立来源，评分继续。

上述目录净增3题的逐题分解：新增803/805/823/827，丢失821；五题两臂均错误。不能把净增3误写成只有三题发生运输变化，827与821体现列表准入/选择的伴随变化。

第33组完整22题7分，221条消息12块逐字写入，client/audit=0、server=-15。词形开启→关闭完整回放7→7、0新调用，逐提示审计通过。15错题：读取端独立验证7、支持1、选择7；原话准入端已送达10、列表准入2、候选池无完整原话2、准备数据无对应原话1。853助手创作、855未指明部位手术、857站位习惯均不自动证明标签属性；858赞歌原话已引用但附加文化融合偏好仍不足。第34组完整22题已启动，250条消息16块，显式词形开关1配置核验一致。

第33组860向量路由复核：原始Hindi用户笔记完整存储并挂到多主题episode；摘要虽漏欧洲设计细节，但_embedding输入使用完整原文，不能归咎摘要。新脚本probe_archived_vector_merge.py以原查询重新嵌入、只读既有向量，精确复现原40条vector路由。目标片段五条查询排名41/30/70/29/24，合并未截断排名64，因此在40条路由限额前丢失。仅定位机制，不提高容量或声称下游收益；与855只有笼统编辑请求的来源缺口分别处理。

向量路由容量完整第33组预检：初版诊断错误排除原实验允许的敏感记忆，14/22路由一致；修正为显式匹配敏感记忆策略后22/22原40条路由逐项一致，旧结果保留。离线80条限额使原话AMU覆盖14→16，但其他渠道已覆盖17题，排除重复收益后仅860新增。此为原始AMU正文覆盖，不保证后续融合、粗排、呈现或答案；默认40不变，下一步完整22题固定数据库、新下游双臂验证。

向量40/80完整下游实验已实现scripts/replay_vector_capacity.py：每臂独立数据库副本、原查询计划与时间基准固定，搜索20次预算保留，排序与答案均新调用；输入/响应事件记录，完整逐输入审计排队。首轮7条记录后发现run_metadata缓存导致候选实际80条路由被记为40，已主动停止并保留未完成记录；每臂配置内清空元数据缓存，40/80/80/40检查通过，v2从完整22题重跑。不能引用首轮部分分数作结论。第34组250条消息16块全部逐字写入。

向量容量审计器通过合成有效样本及9类篡改拒绝检查（嵌入文本、CE文档、LLM提示/模式、缺臂、路由、精排、包哈希、分数）；真实完整44结果审计仍待运行完成。v2已完成记录的配置40/80与实际路由条数一致，控制路由均复现、每次搜索未超20调用。

第34组872揭示因果关系误判：原文仅证明东欧背景与遗产保护兴趣，模型却验证“这些因素导致当前羞耻/悲伤”，选D。381题快照7条词面因果声明中只有此条被支持。诊断候选只给既有独立验证/复核追加因果关系提示，不改事实、来源、模式或本地判定；4项触发/不可变检查及畸形输入保持检查通过。完整第20组29题回放已启动，第34组完整22题在基线终态和容量回放结束后排队；尚无结果，不改默认。872当前问题已自述海平面上升导致社区迁移，不能把当前内容误当应遗忘的旧城市扩张经历。

第34组完整22题8分，250条消息16块精确写入，client/audit=0、server=-15；词形反向开关8→8、0调用，逐提示审计通过。14错题读取门为选择10、支持3、独立验证1；准入为已送达8、呈现1、精排2、候选池无完整原话1、准备数据无原话2。完整第35组31题已启动并核验配置一致。因果提示完整第20组29题9→9、1新调用、审计通过，34组仍按资源队列等待。句中条件兴趣库存392题仅880命中，不据此扩展默认规则。

容量80对照局部机制860：原Hindi用户轮次已通过融合/粗排/精排/列表，完整进入候选读取目录s6，控制目录无该原话；候选仍将金标欧洲家具偏好判unsupported并选错B。说明恢复运输尚未修复支持；全22题对照与调用审计继续，未作采用判断。

准入报告新增render_missing_details：保留全部原分类与字段，另记录全文被拆分的原始载体后续所在阶段。完整第34组22题原字段逐项不变；874补水/休息片段仍在精排输入，随后第28名容量落选，不能仅因完整邮件缺失就认定语义证据在呈现时丢失。该邮件建议与会者补水，仍不证明个人偏头痛。

容量回放861控制出现结构化解析错误后正常修复；离线审计器已依据原answer_calls.failure_phase复现LLMError的ResponseParseError因果链，避免错误回放路径。仅停止等待中的旧审计队列并替换，实时44结果实验未重启；合成错误链及9类篡改检查通过，真实整组审计仍待完成。第35组写入继续，第36组22题已按数据顺序排队。

因果关系提示两组完整51题17→17、4次新增调用，逐提示审计通过；872新选择先B后被复核改回错误D，因果声明仍错误支持。候选不采用。

向量40→80完整22题新下游对照5→6，44记录664提供方事件全部离线重建审计通过；读取目录完整原话14→15（新增854/860、丢失863）。新增原话两题仍错；858得分伴随文化融合偏好被误删为generic，865资格相同而选择改变，849条件兴趣误当事实后弃答。历史控制7分、新控制5分，单次净增1不足以证明稳定收益，不采用。两臂搜索调用220/218、搜索tokens587550/566623，固定计划原调用成本省略，不能当完整端到端成本。详见vector-capacity80-final-review。

第35组因磁盘满在结果flush退出，末行截断，整组无效并保留。47份旧日志经SHA核验无损gzip归档，清理pip/npm缓存；启动器加入512MiB预检和清单原子替换，低空间拒绝路径验证通过。完整31题按原配置重跑，36组及反向开关改为等待重跑终态。

阅读端句中条件兴趣诊断：保留原词汇/嵌套既有事实排除，只将已有条件作用域规则应用于无引用/代写/括号/缩写的后续独立句；十项正负边界及不可变检查通过。第34组仅880D触发，完整34/20组51题回放与逐提示审计启动，默认未改。849的keen-on词汇只读393题仅一例且原基线已generic，不据扩容回放单次弃答另扩词表。

句中条件兴趣完整51题17→17、2次新增调用，两组逐提示审计通过、无提供方错误；880的D已被正确视作条件建议，资格B→BD，但仍选错B。该诊断修复局部作用域未带来准确率收益，不采用、不改默认。47份压缩后持久化日志全部再次解压校验原大小与SHA一致。

全通用合格选项两两比较诊断完整51题17→16（34组8→7，20组9→9），新选择及后续调用28→97；两组逐提示/模式/来源/资格/锦标赛/最终答案审计通过，无调用错误。876候选C赢下三场但与正确A直接比较仍错；两两简化未解决相关性判断，增加成本且退步，不采用。

第35组失败前缀仅作线索：898团队跨欧洲/北美的用户自写履历原话已存储，但载体精排第34/41未入选。CE前提提取不识别“it might help to…”建议尾句，漏该团队前提而保留其他人口属性前提。只读424题有42个选项提取变化；29题第20组和22题第34组原CE查询全部精确重建，候选实际改变2/1题查询。六项否定/条件/嵌套语义边界检查通过；完整第20组相同粗排文档、新CE双臂诊断启动，默认未改。第35组重跑222条消息14块已逐字入库且无孤立来源。

help-to精排边界候选第20组完整29题已审计：新控制精排逐题复现原排序，完整原话覆盖19→19，无新增/丢失；CE调用173→174。查询/文档/分数长度/重复题/缺题5类篡改被拒绝，原控制查询和元数据逐项验证。没有下游分数结论；等待35组重跑成功终态后做完整31题对照，暂不改变默认。

35组903另有遗忘候选生成缺口：完整约束r0已送达，allergic与allergies词形不一致，6个约束词仅seasonal/pollen共享2个，未达半数门槛。414题快照词形库存仅新增903B:r0一对；该题正确A已在支持门排除，仅补B约束无法修复答案。分别记录约束漏检和支持失误，不据单例增加词族规则或付费回放。

第35组完整重跑31题11分，222条消息14块逐字入库，client/audit=0、server=-15、exit-code=0。20错题读取门为选择8、独立验证8、支持4；原话准入为已送达12、精排4、准备数据无对应原话4。完整反向词形开关11→11、0新调用、逐提示审计通过。完整第36组22题已启动并核验原配置，第37组23题按数据顺序排队；35组help-to精排边界对照继续。

help-to精排边界两组完整60题已审计：全部新控制精排复现原排序，完整原话覆盖38→38，逐题无新增/丢失；CE调用367→369。35组898自写团队履历载体40→28，仍未进入前16。语法边界得到修复但未兑现预期运输收益，不采用；未运行该候选下游读取回放，不能声称最终准确率不变或提高。

第35组898精排前16含6个画像字段，各字段评分文本附带同一组宽泛persona片段。新诊断仅在CE评分时保留原字段正文与全部自身引用所在原文片段，去掉其他独立persona片段；真实精排预留、列表与读取证据包仍保留原完整文本。来自SeCom笔记的检索去噪假设，区别于此前全来源max评分或改变证据包粒度。完整31题1550粗排文档中115个改变；正文、引用、输入不可变及多类排除边界检查通过。完整35/20组60题同查询双臂评分及审计已启动，尚无效果结论。

画像字段CE视图完整60题两组均审计通过：新控制精排60/60复现，候选改变41题精排名单但完整原话覆盖38→38，逐题无新增/丢失；CE调用367→359。898相关原话仍第40，去掉附加片段后国籍/种族评分反而提高，教育/语言部分下降，不支持“重复片段单独造成高分”的解释。五类查询/来源/候选评分文档/缺臂/分数长度篡改被拒绝。不采用，未做下游回放、无准确率结论。36组231条消息15块已全部逐字入库。

36组924自写给朋友的邮件明确记述固定朋友同行看球，三份原话载体精排44/45/47落选。检索已有各选项的活动主题，但CE在无可提取字面前提时仅保留笼统问题/子问题。新增单查询诊断：仅当所有选项均有唯一既存主题且无字面前提时，将全部主题作为假设附加，保留原CE查询和1800-byte辅助上限；缺项/重复/超限原样返回。35组31查询精确重建、9改变；36组局部21查询精确重建、6改变。完整35/36两组对照与审计排队，默认不变。925单集电视剧不喜≠避开肥皂剧，926晨间训练≠明确健身房，另保留语义支持边界。

第36组完整22题4分，231条消息15块逐字入库，client/audit=0、server=-15、exit-code=0。18错题读取门选择7、支持5、独立验证6；完整准入另存cohort36-admission-inventory。词形反向开关4.0→4.0、0调用，逐提示审计通过。完整第37组23题运行并核验配置，第38组37题按数据顺序排队。

更正924定位：逐字检查实际CE query确认已有Premise4“if there is a home game...friends/meze/Bosphorus”，不是只含笼统子问题。当前候选要求所有选项均无字面前提，因此924不触发、两臂相同；前述将924解释为选项主题完全省略的说法不成立。完整两组继续按实际触发范围审计，不静默扩大条件或将924计为候选机制验证。

选项主题CE候选完整53题两组均审计通过，新控制精排53/53复现；完整原话覆盖35→34，丢失943、无新增，CE调用353→382，不采用。没有下游准确率结论。新增scripts/check_ce_query_focus.py，在付费查询干预前核对焦点题确实变化且原查询逐字重建；真实924不触发/929触发、未知ID拒绝检查通过，避免将未触发案例误当机制实验。

读取端新增两阶段来源关联诊断：仅2–4个已合格且generic的选项，先独立生成逐选项原始来源ID，再交原选择器及复核参考；不生成解释、事实或引文，不改变资格。区别于此前同调用生成说明并选择答案的source-notes失败方案，纳入完整第19组回归（含520）。第36组22题预检逐调用复现，实际8题触发且包含928；五类缺项/重复/未知/额外说明检查通过。36/19两组完整38题双臂真实选择回放与审计启动，默认未改。

两阶段来源ID关联完整38题7→7，逐提示/模式/来源/资格/答案审计均通过，无提供方错误，选择及后续调用22→39；逐题无得失分，不采用。928两臂首次均A、复核均改错C，不能归因为关联提示提高初选。19组新控制3分低于历史4分，520两臂均错，不把控制漂移归因候选。补充446题只读盘点：95题通用选项复核共9次纠正、6次退步，一律禁用的记录反事实141→138，保留现有复核；该盘点不等同修改运行时后的实验。

### PersonaMem-v3：跨平台文本问答代理评测（2026-10-11）

新增 `personamem-v3` 适配器，来源为 [官方仓库](https://github.com/bowen-upenn/PersonaMem-v3) 和 [预生成数据](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3)。数据固定在 HF revision `7b00a090b35b7293e6efeeb19494207f32b5a9ee`；协议参考仓库 revision `ad80a3b1b322129fb0585ead33b129a87bb4e387`。下载保留逐文件 SHA-256 收据，转换前重新校验。派生数据遵循上游 CC-BY-NC-4.0，代码 MIT。

首批支持 `chatbot_personalized_response`，已准备用户1的19题和用户2的11题。用户1用于开发，用户2作为保留测试用户，不依据其参考答案调参。其余任务保留在原始 `test.json`，转换 manifest 明确列出各任务数量和未支持数量。推荐排序、主动提醒、工具操作等尚未接入，不能使用本子集成绩代表完整 v3。

每道题建立独立用户命名空间，只摄入 `source_timestamp < query.ts` 的五个平台事件，排除题目关联的 held-out source object；毫秒时间戳传给现有入库接口。事件按时间稳定排序，聊天角色原样保留，社交内容明确标注为观察到的平台事件。仅使用白名单观察字段，排除 `preferences`、`embeds_pref_idx`、隐藏画像、置信度等生成注释；不下载 `profile.json` 作为记忆。参考答案和目标偏好仅进入评测字段，不进入 Add/Search/Answer。原始图片/视频不作视觉处理，只用随数据发布的文本描述；这与官方多任务、工具和画像可见性设置不等价。

```bash
# 默认仅下载用户1；需要更多用户时显式指定
.venv/bin/python scripts/download_datasets.py personamem-v3 --persona-v3-users 1
.venv/bin/python scripts/prepare_dataset.py --dataset personamem-v3 \
  --input data/raw/personamem-v3/index.json \
  --output data/prepared/personamem-v3-user1.jsonl

# 保留测试用户独立下载和转换
.venv/bin/python scripts/download_personamem_v3.py --users 2 \
  --output-dir data/raw/personamem-v3-heldout
.venv/bin/python scripts/prepare_dataset.py --dataset personamem-v3 \
  --input data/raw/personamem-v3-heldout/index.json \
  --output data/prepared/personamem-v3-user2-heldout.jsonl

.venv/bin/python scripts/selftest_personamem_v3.py

# 首题完整历史冒烟测试；convs 在本数据集中指独立时间快照
AML_EVAL_DATA=data/prepared/personamem-v3-user1.jsonl \
AML_EVAL_CONVS=1 AML_EVAL_LIMIT=1 AML_EVAL_RESULT_NAME=personamem-v3.jsonl \
.venv/bin/python scripts/run_isolated_https_experiment.py \
  --run-dir runs/personamem-v3-example --port 18444 -- --no-progress
```

完整开发子集可设置 `AML_EVAL_CONVS=0 AML_EVAL_LIMIT=0`；每题重新摄入完整时间快照，成本明显高于共享一个用户全历史的方式。隔离运行器仍执行 `scripts/run_experiment_https.sh`，v2 默认输出路径保持不变，v3 使用独立结果名和 `audit_personamem_v3_https.py`。审计验证输入哈希、题目全集/顺序、独立用户标识、SQLite 原文逐条一致、时间界限、搜索日志存在和评分聚合；不会套用 v2 的 gold snippet CSV 漏斗。

评分协议为 `personamem-v3-chatbot-text-proxy-v1`：当前模型按“回答问题且不虚构用户事实”和“恰当体现指定偏好”两个维度评分，单项取0/0.5/1，再求均值。参考示例不要求逐字匹配。它不是官方 v3 多维裁判得分，也不能与 v2 选择题准确率直接比较。`probe_personamem_v3_no_memory.py` 可对相同题目运行无记忆 Answer/Judge 对照；单题结果仅用于验证链路，不能证明系统有效性。

接入验证结果：10项 v3 边界/审计测试及34项现有数据集测试通过。`runs/personamem-v3-smoke-user1` 首题真实 HTTPS 运行完成，client/audit 均退出0；199条源消息、11个批次逐条审计一致。记忆系统与无记忆对照的两个维度分别均为 `[1, 0]`，均值均为0.50。首题未观察到个性化评分提升；不能据此推断完整子集效果。完整记录在 `runs/personamem-v3-integration-review.json`。

### PERMA：带检查点的个性化记忆选择题（2026-10-12）

来源：[ustclsc/PERMA](https://huggingface.co/datasets/ustclsc/PERMA)，固定数据 revision `440e64e4fb8baec6f7ad10c1de135505f93e7cb1`；格式和评测规则参考 [官方代码](https://github.com/PolarisLiu1/PERMA) revision `d678640987170e8cfbe9260b311e0493b9cd2c31`。数据卡标注 Apache-2.0。下载器保留逐文件 SHA-256 收据，转换时重新验证历史和题目元数据。

当前接入 MCQ 协议，支持 `single` / `multi` / `both` 领域及 `clean` / `noisy` / `style` 历史。默认只下载 `user108`：114道八选一题，含单领域75题、多领域39题；按检查点分为类型1的27题、类型2的27题、类型3的60题。已在本机准备三种历史版本，每种114条评测记录，它们是同一批题的不同历史条件，不能视为342道独立题。交互式多轮代理、工具执行和其成功率指标尚未接入。

官方 `evaluation.py` 在类型1作答时明确使用 `No memory found.`：本适配器因此保留空历史、不构造占位消息，也不摄入该行附带的上下文。类型2/3仅摄入该任务提供的检查点 `context`，每题使用独立用户命名空间；不混入其他检查点或其他用户。`preferences`、`affinity_links`、`user_use_topic_dialog`、画像、任务目标和答案不作为记忆来源，Search/Answer 不接收正确标签。

日期处理按上游精度：题目日期只有天，检查点允许包含当天历史；拒绝晚于题目日期的对话。Morning/Afternoon/Evening 按上游约定映射为06:00/15:00/19:00，以 UTC 毫秒适配本地接口，原始日期字段同时保留。这些小时是规范化约定，并非采集到的精确时刻。八个选项完整保留，按原始顺序从 `A:` 转为本地 `A.` 表示，不截断或改写答案正文。

```bash
# 默认 user108、干净历史、单领域和多领域
.venv/bin/python scripts/download_datasets.py perma --perma-users user108
.venv/bin/python scripts/prepare_dataset.py --dataset perma \
  --input data/raw/perma/index.json --output data/prepared/perma-user108-clean.jsonl

# 单独保存其他历史条件，避免覆盖默认下载清单
.venv/bin/python scripts/download_perma.py --users user108 --variant noisy \
  --output-dir data/raw/perma-noisy
.venv/bin/python scripts/prepare_dataset.py --dataset perma \
  --input data/raw/perma-noisy/index.json --output data/prepared/perma-user108-noisy.jsonl
# 将 noisy 改为 style 可准备风格化历史。

# 全量离线预检，不调用模型
.venv/bin/python scripts/https_eval.py --data data/prepared/perma-user108-clean.jsonl \
  --convs 0 --limit 0 --inspect
.venv/bin/python scripts/selftest_perma.py

# 真实 HTTPS 小批次：前两题覆盖无记忆和有记忆检查点
AML_EVAL_DATA=data/prepared/perma-user108-clean.jsonl \
AML_EVAL_CONVS=2 AML_EVAL_LIMIT=2 AML_EVAL_RESULT_NAME=perma.jsonl \
.venv/bin/python scripts/run_isolated_https_experiment.py \
  --run-dir runs/perma-https-example --port 18444 -- --no-progress
```

`--convs` 在此表示独立检查点快照。完整干净版本含6325个入库批次；不建议把114题默认当作低成本冒烟测试。下载其他用户可使用 `--users user109` 等官方用户ID。当前尚未运行真实付费模型的 PERMA 成绩实验，离线 fake 流程的分数没有质量意义。

`audit_temporal_https.py --dataset perma` 验证原文入库、问题全集及顺序、时间界限、结果标签评分和搜索日志，并按领域/检查点汇总。原 PersonaMem-v3 审计入口保留兼容性。PERMA 使用确切标签匹配的0/1分数，但本地检索、前提验证和选择策略不同于官方基线，因此协议标为 `perma-mcq-local-v1`，不声称是官方排行榜结果；它也不能直接和 PersonaMem-v3 的双维度开放回答评分比较。

### Listwise 用途判断、删除复核与排序实验

`AML_LISTWISE_STAGED=1` 启用分阶段协议：先分批识别支持、反证、限定、桥接、背景、重复、未知或无关，再复核拟删除项，最后对包括恢复项在内的候选只排序。初判依据包含目标 ID、候选自有来源引用和重复项的替代候选；非法引用、上下文不完整、未验证替代项不能授权删除。用途标签不构成事实支持。排序失败保留候选并沿用 CE 顺序。groups 只携带存活候选中已记录的完整、不重叠依赖组，不再占用主判修复次数。

该协议默认关闭，保留旧协议用于对照；2026-10-21 扩大预算后的完整 HTTPS 实验得分 14/26，未超过历史旧协议基线 16/26，因此暂不默认启用。`AML_LISTWISE_PURPOSE_BATCH_SIZE=10` 是单次用途判断的软上限，不降低候选总量，已记录依赖组不会被拆开。`AML_LISTWISE_CONTEXT_REVIEW_LIMIT=2` 限制一次补读请求的候选数；补读使用原始来源的完整文本或相邻句窗口，不改写最终证据包。窗口仍不完整时继续保留。所有调用共享原有 deadline、调用和 token 预算。默认已有的 listwise 提示词也明确区分检索用途与选项充分支持，候选摘录诊断会进入 trace。

staged 各次用途判断、补读和最终排序使用 `AML_LISTWISE_STAGED_MAX_PROMPT_BYTES=48000`，独立于旧协议及候选 admission 的 24000 字节上限。`AML_LISTWISE_CONTEXT_MAX_BYTES=12000` 控制每个候选补读来源的累计 UTF-8 字节数（原为 6000）；补读候选数仍默认 2。启用 staged 时，搜索总 token 预算默认 128000（旧协议仍为 64000），显式的 `AML_SEARCH_MAX_TOKENS` 始终优先。增大字节上限不会绕过总 token、调用数或 deadline 检查；扩大预算后的 HTTPS 实验中，26 次最终排序均通过预算检查，20 次排序成功、6 次输出校验失败回退；搜索降级为 7/26，最终完整来源覆盖为 32/36。平均搜索耗时 17.36 秒、搜索 token 1041973，较历史旧协议基线分别增加 22.4% 和 14.2%。该实验使用相同原始消息重新入库，候选抽取存在随机性；来源覆盖不等于事实正确性，结果不构成严格因果验证。详见 `runs/https-staged-budget-20261021/comparison.json` 和 `default-decision.json`。

以下命令在 `memory_system` 目录运行，产物位于忽略的 `runs/`，可能包含原文：

```bash
# 导出盲审表及独立的历史模型判决；不会生成或伪造人工标签。
.venv/bin/python scripts/evaluate_listwise_staged.py export \
  --run runs/https-fine-capacity-20261021 --output runs/listwise-benchmark

# 冻结同一候选集和 CE 分数，比较表示、排列和批大小；真实调用需要 --execute。
AML_FAKE=0 .venv/bin/python scripts/evaluate_listwise_staged.py run \
  --run runs/https-fine-capacity-20261021 --output runs/listwise-ablation \
  --qa 20,22 --variants legacy_shared,staged_shared,staged_reverse,staged_batch8,staged_batch12 \
  --tokens 128000 --execute

# 人工填写 review.jsonl 的 annotation 后计分；未审核项不进入误删率分母。
.venv/bin/python scripts/evaluate_listwise_staged.py score \
  --labels runs/listwise-benchmark/review.jsonl \
  --predictions runs/listwise-ablation/predictions.jsonl \
  --output runs/listwise-ablation/human-metrics.json

.venv/bin/python scripts/selftest_listwise_staged.py
```

比较初判时统一使用 `proposed_deleted`；旧协议的 `selected_ids` 是初判输出，新协议则是复核和排序后的输出，不能直接混算。`context_incomplete`、`deletion_blocked`、`judgment_checks`、`context_supplements` 和 `ranking_status` 分别记录上下文缺失、未授权删除、判断校验、补读及排序回退。降级不等于证据被删除；最终打包还需单独做漏斗审计。

### 结构化任务轨迹与经验复核

`POST /feedback` 保留原有 `task/outcome/trace` 调用方式，原有请求的幂等哈希不变。
`outcome` 现在支持 `success/failure/partial/unknown`；可选 `trajectory` 保存任务类型、目标、
约束、验收标准、模型/数据版本/配置、步骤输入输出、指标及基线、错误和产物路径。
步骤 ID、指标名必须唯一，数值不接受 NaN/Infinity。环境签名、代码及依赖版本仍使用原有顶层字段。
`distill=false` 只记录反馈和轨迹，不调用模型；默认仍蒸馏。响应返回 `feedback_event_id`，
`GET /feedback/{user_id}/{event_id}` 可读取原始记录。新接口沿用现有认证和用户作用域。

```json
{
  "user_id": "u", "session_id": "experiment-1", "feedback_event_id": "trial-1",
  "task": "修复 listwise 请求超限", "outcome": "partial",
  "trace": "扩大预算后请求超限消失，但正确率未提高。",
  "execution_kind": "replay", "distill": false,
  "environment_signature": "local-test", "code_version": "commit-sha",
  "dependency_versions": {"python": "3.14"},
  "trajectory": {
    "task_type": "search-optimization", "goal": "降低超限且保持准确率",
    "constraints": ["同一候选池"], "acceptance_criteria": ["准确率不得下降"],
    "dataset_version": "dataset-sha", "settings": {"prompt_bytes": 48000},
    "steps": [{"step_id": "rerank", "action": "冻结候选后重放", "status": "partial"}],
    "metrics": [{"name": "correct", "value": 14, "baseline": 15, "unit": "questions", "direction": "higher"}]
  }
}
```

反馈记录是观察/回放/模拟记录；其蒸馏出的经验一律先记为 `hypothesis`，模型不能自动升级。
经验附带适用条件、排除条件、验证计划，以及可定位到事件和 payload hash 的原始 trace 引文。
引用不存在就拒绝写入。`GET /experience/{user_id}/{memory_id}` 返回状态、版本、证据链接和去重案例数。
相同任务/trace/环境的重复蒸馏追加事件链接，但不增加独立案例数；近似语义匹配不足以合并不同条件或结果。
`execution_kind=replay/simulation` 以及 `partial/unknown` 不增加真实 helpful/harmful 计数。
这些计数表示调用方归因的任务反馈，不表示因果效应或独立实验样本数。

`POST /experience/review` 接收 `user_id, memory_id, expected_version, status, reviewer, rationale, evidence`。
其中 `evidence` 是 `{event_id, quote, relation}` 列表，relation 为 `support/counterexample/context`。
允许 `hypothesis → validated → adopted`，也可转为 `rejected/stale`；后两者需先回到 hypothesis 才能重新验证。
升级必须引用当前用户已有事件中的准确引文，且至少一项支持记录具有匹配的任务、环境、代码、依赖、
模型、数据版本、约束和配置，以及 `environment_verified=true` 和 `verification_result=passed`。
这些是调用方提交、复核者确认的执行记录，不是服务自行执行工具后提供的证明；模拟记录不能授权升级。
不允许跳过验证或使用旧版本复核；每次变更写入已有 claim history。状态、证据和反馈随用户 purge 一起删除。
搜索正文显示经验状态和执行类别，拒绝/过期经验明确标记为不可直接采用的建议。

### 冻结快照对照回放

`experience_replay.py` 为已完成且通过审计的本地 PersonaMem-v2 HTTPS 实验创建 SQLite backup
（包含 WAL 中已提交的数据）、候选/CE/查询/题目快照及文件哈希。它固定的是候选与 CE 阶段输入，
不会重新抽取记忆或检索数据库；数据库副本作为来源记录保存。两组使用生产 cascade、复核和打包实现，
只切换 `LISTWISE_STAGED`，其余搜索设置沿用原始 trace。默认三轮，交替执行组别；可选运行回答评分。
两组 coarse/fine/listwise 准入不一致或任一组失败时，该对记为不可比较，不计入有效得分差。
不评分时得分字段为 null，不把候选保留率当作准确率。数据不写入线上反馈，也不自动升级任何经验。

在 `memory_system` 目录运行：

```bash
.venv/bin/python scripts/experience_replay.py freeze \
  --run runs/https-staged-off-20261021/legacy --output runs/experience-snapshot
.venv/bin/python scripts/experience_replay.py verify --snapshot runs/experience-snapshot
AML_FAKE=0 .venv/bin/python scripts/experience_replay.py compare \
  --snapshot runs/experience-snapshot --output runs/experience-paired \
  --execute --repeats 3 --answer
```

快照和输出目录拒绝覆盖；读取及运行结束均校验快照完整性。`--cases` 可指定以逗号分隔的
`conversation_id:qa_id`。结果含每组调用、预算、trace、证据包和可选评分，便于按失败阶段复核。
这是冻结上游的局部对照，不包括真实检索/入库耗时，不等价于端到端 HTTPS 效果或因果证明。

### 召回主题、来源配额与上下文标签

搜索支持三项可独立消融的改动，均记录在 `versions.settings` 中：

- `AML_RECALL_TOPIC_ANCHORS`（默认开启）：在原问题和个人前提之外补充选项的字面主题视图，
  每项不超过 240 字符，最多 `AML_RECALL_TOPIC_ANCHOR_LIMIT` 项（默认 4）。
  不替换已有查询，不为补充主题创建事实支持或新的必达需求，保留原本为空的通用建议槽位。
- `AML_SOURCE_RECALL_DIVERSITY`（默认开启）：在原文 FTS 的每查询截断之前优先分配独立原始来源；
  来源不足时再填充同来源的其他 AMU。已有权限、遗忘、历史和敏感信息过滤继续生效。
  此项改善原文载体分配，不是独立的原文向量索引。
- `AML_LISTWISE_CONTEXT_LABEL`（默认开启）：keyed listwise 输出增加 `context` 类别，表示有用的
  话题或背景材料；与 `useful/uncertain` 一样进入排序，并继续受最终容量约束。
  该标签只记录检索相关性，不设置个人事实、支持、资格或蕴含验证标记。
  trace 的 `candidate_relevance` 记录逐候选标签，协议版本为 `keyed_candidates_v2_context`。

从仓库根目录运行冻结入库／规划对照：

```bash
memory_system/.venv/bin/python memory_system/scripts/probe_recall_context.py \
  --run memory_system/runs/https-manual-20261021-161313-1262163 \
  --output memory_system/runs/recall-probe.json
memory_system/.venv/bin/python memory_system/scripts/evaluate_recall_context.py \
  --run memory_system/runs/https-manual-20261021-161313-1262163 \
  --output memory_system/runs/recall-live-control --arm baseline
memory_system/.venv/bin/python memory_system/scripts/evaluate_recall_context.py \
  --run memory_system/runs/https-manual-20261021-161313-1262163 \
  --output memory_system/runs/recall-live-treatment --arm optimized
```

离线 probe 不调用模型，分别比较查询主题、来源配额及两者组合；只衡量词法／原文通道的
参考载体命中。live evaluator 读取原实验 SQLite backup，在新目录创建副本，冻结原查询
规划，重新调用 embedding、排序与答题模型；支持 `anchors/diversity/context` 单项组别。
不修改原实验库或服务数据库。输出目录不可覆盖，参考答案和片段仅用于评分。
默认每组并发 2，耗时不能直接与历史单请求服务指标比较；单轮差异也不等于统计显著提升。
可用 `scripts/audit_recall_context.py --run <回放输出目录>` 独立重算每层覆盖，结果为
`coverage-audit.json`。参考片段按 CSV 行号、问题、答案和历史身份联合匹配，不能仅按问题文本
查找：数据中相同问题可能属于不同历史与不同参考片段。审计不调用模型，也不覆盖原始回答。

### fine 阶段的可选主题上下文

`AML_FINE_TOPIC_CONTEXT=1` 让 `option_topic` 查询继续参与 adaptive fine 的补充候选筛选。
需要同时启用 `AML_RECALL_TOPIC_ANCHORS=1` 和 `AML_FINE_EVIDENCE_ADAPTIVE=1` 才能产生效果。
最多接受 4 个、每个不超过 240 字符的主题，只匹配排序视图中可见的来源；不为主题创建
核心保留席位或个人事实支持。原有核心选择、额外候选上限、listwise 字节／token 准入保持原规则。
`cascade.fine.selection.adaptive.context_facets` 记录参与的主题，运行元数据记录开关。
默认关闭，冻结回放目前未显示完整参考回合覆盖提升，HTTPS 对照见
[召回迭代记录](RECALL_ITERATION_20261022.md)。

```bash
memory_system/.venv/bin/python memory_system/scripts/probe_fine_topic_context.py \
  --run memory_system/runs/recall-context-20261022/optimized \
  --output memory_system/runs/fine-topic-probe-new.json
```

此命令从仓库根目录执行，拒绝覆盖输出。它固定候选和已记录的 CE 分数，首先逐题复现
历史 fine 选择，再比较主题传递开关；不调用模型，不声称等同于最终证据召回或 HTTPS 成绩。

### listwise 提示预算遗漏的补审

`AML_LISTWISE_OVERFLOW_REVIEW=1` 在主排序之后，对 fine 中仅因提示字节上限未准入的候选
做一次原文用途审查。默认开启；最多 4 条候选，12,000-byte 提示上限，沿用剩余请求预算和
本轮排序截止时间。只有合法 useful 判断且已录依赖完整的候选可追加，失败或无效判断不追加。
最终包容量继续执行。`cascade.overflow` 单独记录提议、模型实际输入和追加项，不改变主
`cascade.listwise.candidate_ids/selected_ids` 的含义。

冻结同一入库／规划／时间锚点／使用计数，运行两组真实 HTTPS Search 和本地 reader：

```bash
memory_system/.venv/bin/python memory_system/scripts/evaluate_frozen_https.py \
  --run memory_system/runs/https-fine-topic-20261022/control \
  --output memory_system/runs/https-overflow-new
memory_system/.venv/bin/python memory_system/scripts/analyze_overflow_https.py \
  --run memory_system/runs/https-overflow-new
```

输出目录不可覆盖。两组使用同一源码快照，各自创建独立数据库和 TLS 服务；服务 fixture 不含
参考答案或片段。协议是冻结上游的 HTTPS Search 消融，不包含新 Add/planner 质量，不是官方
评分；单轮仍有模型随机性。审计报告新增 `overflow_input/overflow_selected` 阶段，详见
[迭代记录](RECALL_ITERATION_20261022.md)。

`AML_LISTWISE_ROLE_BOUND_REFS=1`（默认关闭）在 span 引用协议中，将 `evidence_kind` 与可引用
来源角色关联：用户类标签只能引用 user，assistant_context 只能引用 assistant；负判断不带
引用。原文和验证器不变，provider 若忽略 schema，仍由现有角色／引文／归属检查拒绝无效
判断。需要 `AML_LISTWISE_SPAN_REFS=1`，同时作用于删除复核及已开启的 overflow 补审。
新 schema/system 的大小计入原请求预算。可用 `evaluate_frozen_https.py` 的
`--toggle LISTWISE_ROLE_BOUND_REFS`，从 overflow 已开启的历史 run 创建两组相同上游的
HTTPS 对照；运行结果仍由 `analyze_overflow_https.py` 审计。

`AML_LISTWISE_TRANSFORM_CONTEXT=1`（默认关闭）为来源复核补充翻译／润色素材的用途说明。
只改变 system 指令：评估具体素材的检索相关性，不因任务是文字变换就自动排除，也不因此
认证个人事实。归属未明的相关素材沿用非个人上下文标签，完整原文、外层任务、schema 和
引文／归属验证器保持原规则；新增指令字节计入预算。`context_policy` 记录实际启用情况。
可从角色约束关闭、overflow 开启的历史运行，使用 `evaluate_frozen_https.py` 的
`--toggle LISTWISE_TRANSFORM_CONTEXT` 进行单开关 HTTPS 对照。

`AML_LISTWISE_OVERFLOW_EXTEND=1`（默认关闭）在 overflow 已开启时，先按原 12,000-byte／
最多 4 条规则生成提议，再保留其顺序，在总提示 16,000-byte 内最多追加 2 条完整候选。
仍只审查 fine 的提示字节遗漏，主输入候选不得重新进入；共享剩余 call/token/deadline，
只调用一次补审，主删除复核保持 12,000-byte，最终包容量不变。trace 中 `baseline_ids`、
`extra_ids`、`prompt_limit_bytes` 记录准入边界；追加候选仍需合法 useful 判断和完整依赖。
可用 `probe_listwise_overflow.py --extend` 固定输入诊断，再用
`evaluate_frozen_https.py --toggle LISTWISE_OVERFLOW_EXTEND` 进行真实 HTTPS 单因素对照。

`probe_coarse_capacity.py --run <已完成 HTTPS arm> --output <新 JSON>` 用固定候选重现
原 coarse 选择，再比较保留原提议后的容量追加（默认 4／8／16／24 条）。不调用 provider，
不生成新 CE 分数，只报告 coarse 参考覆盖上界和完整文档字节；输出拒绝覆盖，不能将
结果当作最终召回或实际 provider 成本。

`AML_COARSE_EVIDENCE_EXTEND=1`（默认开启）先执行原 coarse 选择，再从容量增加 8 的同一
选择器结果中追加最多 8 条新候选，原提议 ID 顺序保持前缀。trace 的 `baseline_ids`、
`extra_ids`、`baseline_limit` 与 `extra_limit` 明确区分两部分。新增来源继续通过已有
CE 大小／call／token／deadline 准入、fine 和 listwise；没有额外保障席位或放宽最终包。
可从其他扩展开关关闭的历史 arm 使用 `evaluate_frozen_https.py --toggle COARSE_EVIDENCE_EXTEND`
进行冻结上游的真实 HTTPS 对照；对比审计在 coarse 之前匹配输入，不排除干预引起的 fine
或主输入变化。实际收益与成本见迭代记录。

`probe_coarse_fine_core.py --run <coarse 扩展 arm> --output <新 JSON>` 固定真实 CE 分数，
先复现实际 fine ID、主提示输入 ID 和提示字节，再诊断限制新增 coarse 候选参与核心的数量。
`--new-core-limit` 支持 0／1／2／4／8，默认 0 表示仅原 coarse 池参与核心；完整池继续竞争
原有补充名额。支持复现核心配额开启或关闭的历史 arm，使用各题实际 token 余额复算主提示准入。
仅适用于 CE 完整、adaptive fine 开启且主题实验关闭的 trace；不修改在线实现，不调用模型。
输出包括 fine 和主输入覆盖，不能代替模型用途判断、最终 packet 或答题验证。

`AML_FINE_COARSE_CORE_QUOTA=1`（默认开启）在 coarse 扩展和 adaptive fine 都开启时，允许
新增 coarse 候选中 CE 最高的至多 1 条参与 fine 核心选择；原 coarse 候选照常竞争核心，
完整候选池继续竞争补充名额。它限制新来源挤占核心的数量，但不保证每条原核心都保留。
原池中的 CE 失败救援和融合头部预留仍按原池计算；没有 CE 分数的新来源不能凭此进入核心。
trace 记录 `restricted_core_pool_v1`、允许的核心候选池与新增核心名额上限。完整来源、
依赖、fine 总容量、主提示准入与最终包限制保持原配置。使用
`evaluate_frozen_https.py --toggle FINE_COARSE_CORE_QUOTA` 从 coarse 已扩展的 arm 创建
单开关 HTTPS 对照。`probe_coarse_fine_core.py --new-core-limit 1` 可固定真实分数复现 fine
及主提示准入，但不调用模型或验证最终答题。

`audit_https_stages.py` 同时单独记录助手参考文本的逐阶段传输：
`assistant_reference_transport` 为汇总，逐题明细在 `assistant_visible`。它不改变原用户
回合指标，也不把助手文本视为个人事实；长回复的完整覆盖不是充分证据判断。

`probe_review_order.py --run <已完成 arm> --output <新 JSON>` 默认只离线复现复核请求，
检查将目标／引文／类别放在 `useful` 之前的输出顺序提议：来源提示和 JSON 约束保持等价，
新增系统说明纳入请求预算。指定 `--live-qas 11,17 --stages main --first-batch-only --repeats 3`
才会调用真实 provider 做固定请求配对，保存全部候选的原始输出字段顺序和引文校验结果。
这些是开发诊断，不能当作全流程 HTTPS 结果；该脚本本身不改变生产选择规则。

`AML_FINE_WEAK_PROMOTION_ONE=1`（默认关闭）将 adaptive fine 核心中、CE 名次在原核心
截断线以外的弱预留晋升上限设为 1；默认 16 个核心名额时原上限为 2。弱预留包括纯
`option_topic_coverage`／`fused_head`，不限制已在截断线内的候选、强来源见证、CE 失败
救援或原有补充／依赖名额。仅在 adaptive fine 且有 CE 分数的分支生效。
`evaluate_frozen_https.py --toggle FINE_WEAK_PROMOTION_ONE` 可从既有 arm 创建单开关
HTTPS 对照；`analyze_overflow_https.py` 检查实际晋升数量及生效上限。

`probe_fine_soft_reservations.py --run <已完成 arm> --output <新 JSON>` 固定完整真实 CE
分数，先复现 fine ID、主输入 ID 与字节，再比较弱晋升上限和主提示优先顺序。分别报告
用户／助手参考传输，不调用 provider；其中主提示顺序变体仅为离线诊断，未加入生产。
离线 fine 覆盖改善还需验证主提示／overflow 准入和有效复核，不能直接视作最终召回。

`probe_source_merge.py --run <已完成 arm> --output <新 JSON>` 在私有内存副本上复现原文
召回通道，比较按新来源分配名额与原问题前缀预留；各变体仍使用原通道容量。必须逐题
精确复现原 source route。AMU FTS 的复现情况另外报告；跨通道合并指标固定使用历史
其他通道，不代表 graph／follow-up／fusion 会在真实 Search 中保持不变。参考来源 ID
只用于选择后的评分，脚本不调用 provider，也不修改线上召回规则。

`AML_SOURCE_RECALL_NOVELTY=1`（默认关闭）在 `SOURCE_RECALL_DIVERSITY` 已开启时，
跨查询合并原文检索结果优先给包含新来源的候选分配名额，已完整代表的来源集合后置并
补齐余位。通道上限不增加，原文、过滤和后续预算保持原规则；相同来源的不同候选仍
可能携带互补片段，所以必须验证最终传输。每个 source route 的 `source_merge` 记录
输入来源集合、原选择、新选择与有效策略。可用 `evaluate_frozen_https.py --toggle
SOURCE_RECALL_NOVELTY` 创建单开关 HTTPS，对照审计由 `analyze_overflow_https.py`
独立重算分配，并比较首轮合并输入／FTS／向量，允许后续候选池沿干预路径变化。

原文合并回放现在同时要求 FTS 精确复现。旧 trace 未记录部署级敏感来源开关时，须用
`--legacy-sensitive-recall on` 或 `off` 明确指定历史配置；不再固定关闭。新运行元数据
记录 `SENSITIVE_RECALL_ENABLED`，Search trace 同时记录请求及最终生效的
`include_sensitive`，以免离线过滤与实际运行不一致。

最新 `https-source-novelty-20261023` 完整 HTTPS：答题 29→35/56（51.8%→62.5%），
但最终用户参考覆盖 38→35/52、完整参考集合 32→30/52，token 增加 1.19%。Q46 新
来源进入主提示后被有效负向复核，未形成最终恢复；该开关继续默认关闭。详细漏斗、
反例和固定请求诊断见 `RECALL_ITERATION_20261022.md`，不把答题上升视作召回改善。

`probe_supplied_text.py --run <已完成 arm> --output <新 JSON>` 离线复现历史请求及
可解码判断，诊断供翻译／润色正文的非个人事实上下文提示。显式指定 `--live-qas`
才进行真实配对调用；计划先落盘、单次结果逐条保存。当前 36-call 诊断未恢复 Q47
目标，且出现无关上下文保留，故不采用该提示，脚本未修改生产校验或选择规则。

`AML_LISTWISE_ABBREVIATION_PARTS=1`（默认关闭）在 span-reference 复核中保留明确缩写
后续词，避免把 `U.S. automakers` 或 `5 p.m. since…` 切成两条不完整陈述。保留原文、
换行边界及未变片段 ID，角色／归属校验不变。`probe_abbreviation_parts.py` 可复现
历史请求、统计精确片段和预算变化；指定 `--live-qas` 才调用真实 provider。
`evaluate_frozen_https.py --toggle LISTWISE_ABBREVIATION_PARTS` 支持单变量 HTTPS；
旧数据缺少敏感来源配置时须显式指定 `--legacy-sensitive-recall on/off`，已有配置不能覆盖。

缩写分句完整试验 `https-abbreviation-parts-20261023`：答题 29→30/56（51.8%→53.6%），
最终参考覆盖 38→37/52、完整集合 32→30/52；未证实召回收益，保持默认关闭。
`audit_recovery_requests.py` 使用冻结源码核对实际复核提示、预算、精确引文和校验结果；
`audit_reference_changes.py` 在完整漏斗审计后跟踪逐题得失及相同输入下的模型波动。

本轮召回优化的 LLM 固定为 `gpt-4o-mini`。`evaluate_frozen_https.py` 在运行前核对
当前模型和历史来源 trace，服务启动与 reader 请求前再次检查；`analyze_overflow_https.py`
核对 reader 元数据、每条服务 trace 和服务端 LLM 调用日志，并输出 `model_audit`。
其他模型或缺失模型记录会报错，不会自动改配置；embedding／CE 不受此 LLM 名称约束。
`selftest_evaluation_model_guard.py` 验证拒绝路径。检查范围是本地配置和记录的请求名，
不保证外部代理内部实际路由。

`probe_review_subjects.py` 和 `probe_source_reviews.py` 的 60-call 配对诊断均未恢复
目标参考，不进入生产。`probe_review_framing.py --run <已完成 arm> --output <新 JSON>`
只离线复现请求，检验移除复核首句的先前排除结论；指定
`--live-qas 7,10,22,26 --repeats 3` 才进行真实配对。三者均检查固定模型，保留原有
引用验证，结果不能等同于完整 HTTPS 答题率。

复核首句诊断 60 calls 完成：有效正向 9→9，目标恢复 0→0；未采用。
最新完整 HTTPS 答题率仍为 30/56（53.6%），对照为 29/56（51.8%）；
这些局部复核诊断没有产生新的全流程分数。

`AML_LISTWISE_TOPIC_INTEREST=1`（默认关闭）在证据复核系统提示中区分具体主题问句与
选项中的个人前提。问句可以作为 `topic_interest` 原始背景，但不会变成所有权、习惯或
偏好的证明；原文、引用验证和事实标记保持原规则。`probe_topic_interest.py` 的 84-call
诊断稳定恢复 Q23 参考 3 次，但也增加了语义噪声，因此需要完整对照判断取舍。
可用 `evaluate_frozen_https.py --toggle LISTWISE_TOPIC_INTEREST` 运行，审计检查有效策略。
`audit_recovery_requests.py` 另记录重建后的完整请求、系统提示与 schema 哈希；比较系统
提示变更时不能只使用原文 prompt 哈希。

主题问句完整 HTTPS `https-topic-interest-20261023`：答题 32→35/56（57.1%→62.5%），
最终参考覆盖 39→37/52，完整参考集合均 32/52，用户／助手完整回合均 54／30。
没有新增参考题命中，并引入无关主题；`LISTWISE_TOPIC_INTEREST` 保持默认关闭。
调用 466→461，token 增加 0.64%，两组无运行错误、固定模型及完整审计通过。
`audit_review_additions.py` 导出非参考来源在内的全部正向变化，供人工检查精确引文与
实际问题的关系；形式有效不代表语义相关。完整报告见 `RECALL_ITERATION_20261022.md`。

`probe_review_presentation.py` 隔离候选成员、候选编号、来源顺序和来源编号；Q23 的
36-call 诊断显示来源顺序与主题提示存在交互，但原始提示下没有有效恢复，因此未采用
排序改动。`probe_review_connections.py` 的 114-call 关联说明诊断与 `probe_target_order.py`
的 12-call 目标顺序诊断也未恢复目标参考，均未进入生产。

`AML_PACKET_ADJACENT_CONTEXT=1`（默认关闭）已接入 Search 打包阶段；
`probe_adjacent_context.py --run <已完成 arm> --output <新 JSON>` 可离线试算核心规则。
对已保留的助手片段，检查合法 fine 候选是否只额外含其直接前序完整用户回合，
在原包剩余容量内补充最多两条原始用户上下文，不添加个人事实或覆盖标签。
最新 control 的离线参考覆盖 39→41/52、完整用户回合 54→57，增加 508 bytes，无原包
内容损失；仍需真实 HTTPS 验证，不能当作已上线召回收益。详见迭代记录与对应试算 JSON。
生产入口额外排除已选候选，重新执行时间过滤；补充项不继承原候选摘要、分数或事实标签。
`audit_adjacent_context.py` 用冻结代码重建完整补全过程，检查来源、前缀、digest 与容量；
漏斗增加 `packet_base` 阶段，以单独测量补全的参考传输收益。可用
`evaluate_frozen_https.py --toggle PACKET_ADJACENT_CONTEXT` 做真实对照，模型固定 gpt-4o-mini。

相邻上下文完整 HTTPS `https-adjacent-context-20261023` 已完成：对照 30/56（53.6%），
实验 33/56（58.9%）；两组最终参考覆盖均 39/52、完整参考集合均 32/52。
实验组实际新增上下文为 0，因此本轮分数差不能归因于新功能，开关继续默认关闭。
两组无运行错误，固定模型、冻结输入、来源与完整补全审计均通过。最新瓶颈仍是
主输入 46 题→主选择 38 题的参考丢失；详细结果见 `RECALL_ITERATION_20261022.md`。

随后进行了 156 次固定模型的回复关系／呈现格式诊断。最初 Q22 的恢复未在后续复测
中保持，因此未采用；本轮临时生产接线已撤回，最新完整 HTTPS 成功率仍为上段结果。
诊断脚本 `probe_conversation_links.py` 与 `probe_link_encoding.py` 仅用于实验。

`AML_PACKET_FINE_CONTEXT_RESERVE=1`（默认开启）从本次已过滤并已评分的 fine 候选中，
在原包之外保留最多两项、合计最多 4 KB 的可见原文上下文。要求有新的完整用户来源，
不携带生成摘要、个人事实或覆盖标签，不替换原包，也不表示模型已验证其相关性。
`probe_fine_context_reserve.py` 离线验证最新两组参考覆盖均 39→42/52；平均每题多出
约 1.8–2.0 KB。

最新完整 HTTPS `https-fine-context-reserve-20261023`：对照 30/56（53.6%），实验
33/56（58.9%）。实验组内部参考覆盖从补充前的 40/52 增至 42/52，完整参考集合
33→37/52，直接补回 Q23／Q24；来源、容量、模型与全部审计通过。实际回答目录原文字节
增加 20.34%，开关暂保持默认关闭，继续评估精度与成本；答题分差不作完全因果归因。
`probe_fine_context_reserve.py --base-packet --limit 1` 可离线评估较小配额，不产生新 HTTPS 分数。

`AML_PACKET_FINE_CONTEXT_5K=1`（默认开启，需同时启用原文保留）仅将补充字节上限改为
5,120；两项配额、原包、总预算、来源与事实边界不变。可通过
`evaluate_frozen_https.py --toggle PACKET_FINE_CONTEXT_5K` 比较 4 KiB／5 KiB。
`probe_context_source_structure.py` 为离线结构／容量诊断，不向生产引入删除助手片段的规则。

最新容量 HTTPS `https-fine-context-capacity-20261023` 已完成：4 KiB 对照 30/56（53.6%），
5 KiB 实验 34/56（60.7%），两组无运行错误。固定实验组实际模型输出，扩容直接使参考
覆盖 42→43/52，补回 Q26，原参考回合无损；回答目录原文字节增加 0.86%。完整来源、
模型、请求与容量审计通过，112 个实际 packet／reader catalogs 精确复现。
Q26 的新增来源未被模型显式引用，且多题存在相同包下答题变化，因此分差不作完整因果
归因。两个原文保留开关继续默认关闭，模型固定 gpt-4o-mini，详细归因见迭代报告。

`AML_PACKET_FINE_CONTEXT_LEXICAL=1`（默认开启，需同时启用原文保留）保留原两项，
仅在同一额外字节上限和原 packet 容量内追加最多一项用户原文词项匹配的候选。
所有可见来源完整交付，分数／事实标签不升级。实验使用两组均开启 5 KiB 保留的
`evaluate_frozen_https.py --toggle PACKET_FINE_CONTEXT_LEXICAL`，评估召回增益及输入成本。

本轮最终结果 `https-fine-context-lexical-recovered-20261023`：对照 33/56（58.9%），
实验 31/56（55.4%）。原对照 Q33 发生一次未送达的 HTTPS 断连，已在独立目录仅补测
该题；原试验及全部成功结果不变，排除 Q33 的原始配对 55 题为 32／30。
词项席位直接补回 Q31，原参考回合无损，且回答步骤引用了新增原文；但新增 52 项使
自身 reader catalog 原文字节增加 10.60%，整体答题未改善，开关继续默认关闭。
模型、来源、完整前缀、请求、覆盖、reader catalog 和补测归档审计均通过；详情见迭代报告。

后续 CE 诊断均未采用：完整用户消息评分在 6 题上未补回 Q5／30／47，主输入丢失
Q31；复用既有选项主题规则，在本轮实际触发的 15 题上完成 90 次真实配对 CE 调用，
完整 56 题 fine／主输入参考覆盖仍为 47／46，主输入完整用户参考回合 65→63。
Q47 目标排名 19→23；候选 CE tokens 增加 28.54%，没有新 HTTPS 答题分数。
产物 `current-ce-topics-diagnostic*` 位于上述恢复目录，模型继续固定 gpt-4o-mini。

`AML_PACKET_FINE_CONTEXT_FILL=1`（默认关闭，依赖 reserve 与 lexical）保留原两个
CE 席位和词项席位，再按 fine 顺序使用剩余原文字节预算；额外总字节上限、原 top_k
和总 packet 预算不变。六份快照 336 题离线复现，最新两份参考覆盖均 42→46、无参考
回合损失，原文输入约增加 10.6%。26 项来源／容量／模型相关测试通过；独立 HTTPS
`https-fine-context-fill-20261023` 已完成：对照 35/56（62.5%），实验 30/56（53.6%）。
最终参考覆盖 44→47，实验自身 fill 直接补回 Q7／10／54、原参考无损，reader catalog
原文字节增加 9.86%。112 个实际包和 catalog 精确复现，来源、请求、模型审计通过。
实验 Q15 有一次 CE 超时并部分恢复，排除此题仍是 35／30；预算调用与日志相差该次
取消请求，计费未知，未冒称端到端成本完整。新开关继续默认关闭，详情见迭代报告。

`AML_PACKET_SOURCE_QUERY_CONTEXT=1`（默认关闭）实验：对于已评分的 coarse 候选中未进
fine 的项，仅当它是原问题或原始选项主题的来源查询第 1，且处于该请求既有 fine
软预留 CE 名次窗口内，尝试优先保留最多一个完整用户来源单元。与原两项 CE＋一项
用户词面席位共用 5 KiB；需要 reserve／5K／lexical 开启且 fill 关闭，不改变正式
证据包前缀，不赋予事实支持标签。三份 56 题快照各新增 Q47、无参考回合损失；
106 项相关测试及 168 包生产回放通过，答案收益仍需真实 HTTPS 对照验证。

来源查询席位完整 HTTPS（`https-source-query-context-recovered-20261023`）：对照
31/56（55.4%），实验 34/56（60.7%）；Q27 阅读连接错误只用原证据重试一次且仍错。
固定自身上游输出，两组都仅新增 Q47 参考原文，无参考回合损失，阅读目录约增 0.23%。
Q47 虽答对且引用新增原文，其 D 获选来自范围判断，不能宣称已验证个人偏好；其他
分差混有随机性及服务错误。新开关继续默认关闭；用户指定的对照默认配置不变。

来源查询席位第二轮复验：对照 35/56（62.5%）、实验 33/56（58.9%）；Q17 阅读连接
错误仅用原证据包重试一次且仍错。固定上游后仍仅新增 Q47 原文，无参考回合损失，
输入约增 0.24%；整体答题改善未稳定复现，新开关保持关闭。两轮是同 56 题的重复
观测，不能合称独立新题。完整来源／运行错误／跨轮比较见
`runs/https-source-query-context-repeat-recovered-20261023`。


最新离线约束优先诊断（2026-10-26）：`runs/fine-direct-lifecycle-20261026/`完整117题
原接口精确复现；此前丢失的遗忘请求恢复，回归/验证参考覆盖仍为46题67回合、30题
43回合，均保留原接口参考。仍有3题10个旧卡片未完整保留；完整核查Q2新包后，未
找到原用户行动受限描述的等价保留，因此不采用该直接fine替换方案。新增语境未全量
语义审查，无新HTTPS答题结果或应用集成。单任务CPU50%、硬限512MiB、4MiB分块读取，
峰值197.94MiB、无内存限额事件、零provider调用。详见该目录`final-decision.json`和
`targeted-context-review.json`，迭代目标继续。


后续实验 `PACKET_ENDPOINT_FINE_COMPLETION` 默认关闭：保持整个最终包，再在原
32000-byte容量内补充完整fine单元。117题离线及当前运行时一致性通过，旧证据/约束
全部保留；回归参考39题57回合→46题67回合，验证27题39回合→30题43回合。reader
原文增加约75%/80%，不能据此宣称答题改善。完整234次隔离HTTPS已串行启动，见
`runs/https-endpoint-completion-20261026/state.json`；并发1、CPU75%、硬限1200MiB、
Swap0。公网默认未切换，最终采用须看两组完整答案、错误和成本。文件读取使用4MiB块。

最终包补充实验的回归队列已完整完成并通过运输审计：对照34/56、实验32/56，最终参考
覆盖40题/58回合→46/67；reader报告tokens增加29.24%，Search+reader增加12.79%。
固定实验组实际上游，补充自身增加10个参考回合且旧回合无损，但这不是独立回答的
反事实成绩；跨臂主审判断有波动。回归已未通过预设答题不退步门槛，开关继续关闭。
验证队列仍按原计划串行运行，完整联合结论待终态，详见同目录regression-findings.json。
分析器另修正内部判断统计：partial保留为部分引文被过滤，不混为异常；invalid/error/
unfinished另计，provider账本独立核对。当前scope出现内存high软限流，max/oom/oom_kill为0。

最终包完整fine补充的234次隔离HTTPS及全部审计已结束（2026-10-26）：回归34→32/56，
验证21→19/61，均未通过预登记的答题不退步门槛。最终用户参考覆盖分别40题/58回合
→46/67、27/39→30/43，无旧参考回合损失；reader记录tokens分别增加29.24%/34.08%。
明确拒绝采用，`AML_PACKET_ENDPOINT_FINE_COMPLETION`继续默认0。结论与逐题/逐回合/
账本证据见`runs/https-endpoint-completion-20261026/final-decision.json`和
`final-adoption-review.json`。234题零终止错误，但存在内部校验/修复错误；总记录
tokens16569856，不等同外部账单核对。原实验进程全部退出，公网443健康检查200。
下一轮从完整漏斗继续：验证coarse→fine仍缺25个参考回合；不继续扩大整个fine尾包。


同类型可选fine交换离线实验（2026-10-26）已完成117题精确基线回放：回归fine仅
Q47新增1参考回合，main两组均无增益；回归fine/main及验证fine还发生reader约束
丢失，按预登记拒绝，不接线或追加付费HTTPS。详见
`runs/fine-type-exchange-20261026/final-decision.json`和`outcome-analysis.json`。
单任务CPU0.5核、硬限512MiB，实际峰值205.71MiB，high/max/oom均0；采用4MiB
分块读取及SQLite fetchmany(256)。持续召回迭代目标仍进行中。


原main剩余容量追加（2026-10-26）完成117题离线验证：31题能在原预算内多容纳一项，
原输入/参考/约束无损，但main参考覆盖两组均无增益，按预登记拒绝，不追加付费HTTPS。
证据`runs/main-residual-admission-20261026/final-decision.json`。4项行为测试通过，
实际峰值167.46MiB、CPU0.5核、内存限额事件0；下一项只读库存区分候选排序与容量。


图元数据JSON空白压缩+coarse末端追加（2026-10-26）已完成117题回放与全量容量
盘点：追加机会由31题增至50题，但两组主输入参考回合仍无增益，原参考/约束无损。
33个缺口关联载体压缩后仍超24KB，不能仅改追加顺序解决。保持拒绝、不增加付费
HTTPS；见`runs/compact-graph-capacity-20261026/findings.json`。本轮所有阶段结束，
CPU0.5核、最高内存178.42MiB、内存限额事件0；持续召回迭代目标仍active。


可逆候选ID+coarse末端准入（2026-10-26）完成117题回放：114题能追加候选，原参考/
约束/图上下文保留，但main参考无增益，拒绝采用。后续诊断发现验证组5个缺口载体
已能单独装入，却在CE顺序中输给其他候选，追加首项后又全部放不下；需检查提名
优先级。历史原文CE输入并不全面匹配，尚未复用分数。证据
`runs/local-id-residual-20261026/followup-findings.json`。本轮峰值183.74MiB，三个
串行任务均已结束，无新增模型调用；持续迭代仍active。


2026-10-26 用户已要求停止试验并保留基线：持续迭代暂停；最新 screened-source
预检已停止，新增的三个来源提名脚本已撤回，仅在本地 runs 目录保留冻结记录。
本轮未调用付费模型、未接入运行时。此前实验实现保留为默认关闭的工具，最新 HTTPS
实验结果不予采用，基线服务继续运行。未经新指令不恢复试验。


2026-10-26 用户重新激活持续召回／HTTPS目标。以提交5f48069为起点恢复独立实验，
公网基线默认不变。被停止的screened-source预检保留取消状态；重新登记于
`runs/screened-source-preflight-resumed-20261026/plan.json`，不覆盖旧部分结果。
先完整重放117题、冻结候选及原文请求，再计算每题单个追加候选的召回上限；
预检无provider调用。CPU0.5核、384/512MiB、Swap0、重任务串行、4MiB分块读取。


2026-10-26 重新激活后的来源提名实验：完整117题预检及冻结评分均已结束。
预检每题单项追加的理想上限为验证3题/4回合、回归0；原文CE实际117次调用
（56+61），报告tokens139306，无失败/重试，独立请求正文/评分/账本审计通过。
实际主审输入回归仍40题56回合，验证28题40回合→29题42回合，仅Q340新增2回合；
这不是最终交付或答案改善。详见`runs/screened-source-ce-20261026/audit.json`。

完整审阅114次新增展示、82份不同语境；不少相关性偏弱，旧摘要也有归属概括风险
（例如Q29把照片中的狗写成用户的狗）。原文/角色/遗忘/推断标记保留，不据此授予
事实支持。评审只允许默认关闭接线和隔离HTTPS，不能视为采用结论。

`SOURCE_CE_LATE_ADMISSION` / `AML_SOURCE_CE_LATE_ADMISSION`默认0：原fine/main
完成后，仅在原24候选/24000字节/token预算内追加一个完整来源提名候选；无完整评分
或失败时保留原main与原未编码展示。显示ID/图压缩使用局部数据，无全局渲染patch；
额外CE单次禁重试，不改原阶段全局重试策略。两组117题原基线、全部来源请求与候选、
已冻结提名、主审请求逐字一致，见`runs/screened-source-runtime-20261026/`。
运行时相关82测试、级联接线1测试、HTTPS设置8测试通过，来源helper/执行器此前12测试。
预检峰值309.67MiB，原文CE峰值90.63MiB，运行时验证峰值235.64MiB，三scope内存限额
事件均0；不覆盖更早实验的资源事件。

完整234次HTTPS已预登记并启动于`runs/https-screened-source-20261026/`：回归和验证
各control→treatment串行、并发1、CPU0.75核、1024/1200MiB、Swap0；使用同FastAPI
/search实现的隔离TLS服务，冻结Add/planner/usage，非公网全流程压测。source评分在
真实Search预算/期限内执行，每组结束追加自身基线和来源准入审计。采用门槛包括
两完整队列正确率不退步、最终参考总体有增益且两组旧参考无损、完整错误/成本核对。
公网基线未切换。当前没有本轮完整HTTPS正确率，持续目标保持active。


2026-10-26 HTTPS运行期间补充终态汇总：`scripts/finalize_screened_source_https.py`
要求两组全量原始结果、终止的TLS进程、来源和漏斗审计通过后才生成联合结论。
逐参考回合核对增减，避免总量相等掩盖旧回合丢失；Search/reader用量分别读取
4MiB日志块，partial判断与invalid/error分开，未知失败用量不计为零成本。
3项边界测试通过，上一轮四个实验臂的Search+reader账本总量独立复核完全一致。
`runs/https-screened-source-20261026/finalizer-plan.json`固定最终分析源码，独立
等待现有systemd实验服务真实终态及资源门槛后执行；不会重启实验或重发provider。
本轮迭代仍在进行，未据部分正确率做采用判断。


2026-10-26 来源提名 HTTPS 的回归队列已完整结束并通过全部审计：control 38/56
（67.86%），treatment 33/56（58.93%），零终止错误。main 输入参考仍40题56回合；
最终58→60回合实际为新增Q7/26/54、丢失Q20，并非旧参考无损。无答题获胜题，
Q22/41/47/52/55由对变错；这五题的完整参考覆盖均无变化，判断路径含引文校验、
范围判断及复核差异，不能将分差全部归因于召回。Search+reader报告tokens
3765106→3832598（+1.79%）。完整9题变化档案及记录复核见
`runs/https-screened-source-20261026/regression-paired-review.md`。
回归已不满足答案不退步和旧最终参考无损门槛；默认仍关闭。联合终态见下文。


2026-10-26 来源提名 HTTPS 验证control已完整61题，答对17（27.87%），零终止错误；
client/stage/source审计均0。fine31题44回合→main28/40→最终27/39。
该控制臂记录见`runs/https-screened-source-20261026/validation-control-findings.json`。

等待期间完成独立来源组投影离线预检：
`runs/source-bundle-capacity-v3-20261026/findings.md`。原117题主提示字节与显示审计
精确匹配，保留所有来源及原main/图，按原CE/fused首个可行项至多追加一组。回归
main仍40题56回合，验证28/40→29/41，仅Q292草稿原文新增一回合；不据此推断
家族早发心脏病等个人事实，也没有最终召回/答题收益结论。两组选择先于参考评分固定。
v1结果及v2空来源卡片导出失败均保留，v3修正后8测试通过、全量prepare/score exit0。
116次新增、95份不同展示的全文及关联问题审查已完成；仅Q292审查了完整选项，
不代表全部题目的答案语义裁决。因存在无关来源及缺失语境，拒绝当前首个可行项
选择器接入运行时或付费测评，见同目录`context-review.json`。未接app；零模型调用。CPU25%、
192/256MiB、Swap0，峰值192.5MiB，high214、max/oom/oom_kill0；资源门槛等待生效，
4MiB块/SQLite每批256条。


2026-10-26 联合 HTTPS 最终结果（覆盖上述运行中快照）：234个请求全部完成，
实验、finalizer及验证汇总服务均已inactive/success；停止试验，保留基线。
权威报告为`runs/https-screened-source-20261026/final-decision.json`，结论
`reject_answer_or_reference_or_error_gate`，来源提名功能继续默认关闭。
回归control 38/56（67.86%）、treatment 33/56（58.93%）；验证control
17/61（27.87%）、treatment 15/61（24.59%）。两队列均未通过答案不退步及
旧最终参考无损门槛；验证最终参考39→36回合，丢失Q283两回合及Q298一回合。
零终止错误不代表内部无错误：验证control有一次Search调用费用未知，不能按零费用计算。
本轮使用冻结Add/planner/usage的隔离TLS与相同FastAPI `/search`实现，
不构成公网端到端基准。公网HTTPS服务保持运行，未重启或启用本轮实验功能。
