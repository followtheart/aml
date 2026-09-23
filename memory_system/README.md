# AML Memory System（M1–M3 实现 + ULM 生命周期）

AML（Agent Memory Leaderboard）参赛记忆系统的参考实现，对应
`../agent-memory-system-design.md` v0.4、通用设计 `../llm-memory-survey/memory-system-design.md`
（ULM）与 `../prompts/` 模板库。

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

三級排序預設為 **粗排 50 → Cross-Encoder 精排保留 12 → LLM 列表比較 10**，
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
计入总字节预算但不占普通证据 `top_k`，因此返回条数可能超过 `top_k`。
来源节录按相关性选取，正文优先于短润色请求，保留引文、邻近语境和消息角色。
預覽與最終節錄使用相同短查詢，按句子局部匹配密度排序，避免長篇 persona 靠散落詞彙占位。
同段的 `Marcus wrote:` 等人物歸屬與必要引文一起保留；最終裝包僅移除包內已可見的重複来源片段，保留引用。
只有來源集合、預覽及語義狀態相同的 episode 才在重排前合併。
trace 的 `query_specs` 記錄短查詢及選項對應，`expansion` 記錄擴展原因、直接命中數及配額。
新版 episode 索引带角色的原始对话，不再用“请求润色”等摘要替代正文。
抽取校验失败本身不再把普通原文标记成敏感；显式来源／模型隐私标签仍保留。
推断型记忆显示来源归属提示；主体、个人前提和遗忘范围由回答模型统一判断。

單選題使用 `verified-source-choice-v6-jev-support`：先對完整可見來源做一次 LLM 語意選句，
每個選項最多保留 3 條、每條最多 320 字元的原文引用，不以關鍵詞交集預先排除候選。
語意相關只是候選提示；後續仍獨立驗證引用歸屬、個人前提、遺忘約束及選項資格。
語意選句最多一次呼叫、20 秒、48,000 UTF-8 bytes 提示；個別無效引用會被拒絕，其他有效引用保留。
提供者失敗、輸出無法驗證或預算不足時回退關鍵詞匹配，詳情記於 `answer_witness_retrieval`，
呼叫記於 `answer_calls` 的 `eval.choice_witness`。`AML_CHOICE_SEMANTIC_WITNESSES=0` 可關閉；
`AML_FAKE=1` 使用明確標記的離線關鍵詞回退，不代表語意品質。
抽取與引用校驗之後，JEV 對所有已抽取前提（包含 unsupported）獨立判斷支持關係，
不接收初始支持標籤或標準答案。判斷分歧或信心不足時，現有 LLM 單次復核相關選項，
重新校驗原文引用，再進入完整選項語意驗證、約束與選擇。JEV 不直接修改選項資格。
使用 OpenRouter `typesafe/jev-1.13` 的 `POST /api/alpha/decisions`；設定 `OPENROUTER_API_KEY`
或 `AML_JEV_API_KEY` 即可啟用，與原有聊天提供者分開。`AML_CHOICE_JEV_SUPPORT=0` 可消融。
無 key、離線、超時、無效回應或預算不足均保留原判斷，原因記於 `answer_jev_support`。
回答通常需語意選句、前提支持、語意驗證三次 LLM 呼叫；啟用 JEV 時另加一次 Decisions，
分歧復核最多另加一次 LLM；有約束、其他復核、格式修復或多個合格選項時增加。
完整設定下獨立單選回答預算最多 11 次提供者呼叫；共享既有預算時不擴張上限。
JEV 整階段預設 20 秒，保留原始證據並限制請求大小；詳見 [JEV_SUPPORT.md](JEV_SUPPORT.md)。
其他選擇題仍使用 `direct_evidence_v2`。
Cross-Encoder 批次與 embedding 另計入 Search 的提供者呼叫預算。
`choice_alignment.py` 及其 replay 腳本保留作歷史離線診斷。

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
值外且由空白分隔的 `#` 行尾註解會被移除；引號內的 `#`、未分隔的字面 `#` 及 Windows 路徑反斜線會保留。

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

# 每次 LLM/Embedding 调用会输出 aml.metrics 日志：阶段、模型、
# attempt/retries、耗时和 prompt/completion/total tokens，不记录正文或密钥。

# 消融
python scripts/local_eval.py --data data/locomo_eval.json --convs 1 --limit 100 \
    --no-graph          # 去掉图谱 PPR 路
    # --no-governance  去掉治理（全部 ADD）
    # --no-rerank      去掉相关性过滤/重排
    # --no-keyexp      去掉查询理解扩展
```

`POST /feedback` 接受显式任务成败与轨迹，蒸馏最多 3 条程序性记忆；普通
对话不会推断任务结果。`DELETE /memory/{user_id}` 执行合规硬删除并返回不含
正文的 receipt。敏感记忆默认不召回，只有服务端开关与请求 opt-in 同时开启
才可访问。完整正文调试日志默认关闭。

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
