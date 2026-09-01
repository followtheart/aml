# AML Memory System（M1–M3 实现）

AML（Agent Memory Leaderboard）参赛记忆系统的参考实现，对应
`../agent-memory-system-design.md` v0.2.1 与 `../prompts/` 模板库。

## 架构

```
app/
  config.py          环境变量配置（模型/密钥/路径/阈值）
  schemas.py         Add/Search 契约的 pydantic 模型
  llm.py             LiteLLM 统一抽象 + 离线 FakeLLM（AML_FAKE=1）
  embeddings.py      LiteLLM embedding + 哈希 Fake embedding
  store.py           SQLite 存储（AMU + FTS5 + triples + 幂等账本）
  graph.py           实体-AMU 二部图 + PPR（HippoRAG 式单步多跳）
  add_pipeline.py    抽取→治理(ADD/UPDATE/SUPERSEDE/NOOP)→多索引→滚动摘要
  search_pipeline.py 查询理解→六路召回→RRF→过滤重排→时间区间前缀→top_k
  main.py            FastAPI: /add /search /health（Bearer 鉴权）
scripts/
  selftest_contract.py   契约自测 11 项（幂等/回显/隔离/422/health…）
  convert_locomo.py      locomo10.json → 评测格式（1542 QA）
  local_eval.py          本地代理评测 + 消融开关
data/
  sample_eval.json       内置小样例
  locomo10.json          LoCoMo 原始数据（已下载）
  locomo_eval.json       转换后的评测数据
```

## 快速开始

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
