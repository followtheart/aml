# AML Memory System（M1–M3 实现 + ULM 生命周期）

AML（Agent Memory Leaderboard）参赛记忆系统的参考实现，对应
`../agent-memory-system-design.md` v0.3、通用设计 `../llm-memory-survey/memory-system-design.md`
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
  graph.py           实体-AMU 二部图 + PPR（HippoRAG 式单步多跳）
  add_pipeline.py    切分→MemCell 抽取(episode+facts+triples)→新颖度门控→治理→多索引→场景巩固→滚动摘要
  search_pipeline.py 查询理解(锚定用户最新记忆时间)→七路召回(含场景→情景)→RRF→小 R 重排
                     →充分性验证/迭代召回/弃权→前瞻时效过滤→top_k
  main.py            FastAPI: /add /search /health（Bearer 鉴权）
scripts/
  selftest_contract.py   契约自测 11 项（幂等/回显/隔离/422/health…）
  selftest_ulm.py        ULM 生命周期自测（切分/场景/门控/热度/遗忘/验证器/前瞻）
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

回答上下文默认总上限为 24,000 字符，单条上限为 2,400 字符，分别通过
`AML_ANSWER_CONTEXT_MAX_CHARS`、`AML_ANSWER_CONTEXT_ITEM_MAX_CHARS` 配置。
预算覆盖所有记忆类型、证据正文及分隔符，按检索顺序保留，截断处标注
`[truncated]`；问题、选项和任务指令不被截断。这是字符预算而非精确 token
预算；较长任务可按需调大，以免丢失证据。搜索结果的 `content` 仅拼接有正文
的证据、时间戳和角色，来源 ID 等元数据保留在 `sources` 字段中。
`aml.context` 日志记录输入和实际上下文字符数，修改配置后需重启进程。

遇到服务端限流（HTTP 429 / `RateLimitError`）时，所有模型调用默认额外重试
6 次，异步等待约 15、30、60、120、120、120 秒（带随机抖动）。若服务端
返回 `Retry-After` 秒数或 HTTP 日期，至少等待该时长。日志中的
`retry_delay_s` 显示实际等待时间。可通过 `AML_RATE_LIMIT_RETRIES`、
`AML_RATE_LIMIT_BACKOFF_SECONDS`、`AML_RATE_LIMIT_MAX_BACKOFF_SECONDS` 调整；
修改后需重启评测/服务进程。持续限流或配额不足仍会在重试耗尽后报错，
本地评测结果应检查 `error_stage` / `error_type`，避免将调用失败当作答错。
离线重试测试：`python scripts/selftest_metrics.py`。

主动节流：同一进程、事件循环内，同类同模型调用串行执行。默认最近 60 秒
最多发起 30 次请求（含失败尝试）；已返回的累计 token 达到 60,000 后，
后续请求等待旧用量退出窗口。分别通过 `AML_PROVIDER_RPM` 和
`AML_PROVIDER_SOFT_TPM` 配置，0 禁用对应阈值；日志显示 `provider_throttle`。
这是本地保守阈值，不代表服务商实际额度，也不预估下一次请求的 token；
请按账户额度留余量调整。其他进程/账户共享用量以及服务商的其他限流仍可能
触发 429，届时继续使用等待重试。修改后重启评测/服务进程。

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

## 验证状态（2026-09-01）

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
