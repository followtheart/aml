# aml — Agent Memory Leaderboard 参赛记忆系统

面向 [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) 的 LLM agent
记忆系统：平台侧只需对接同步 **Add / Search** 两个接口，Answer/Eval 由平台统一执行。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `agent-memory-system-design.md` | 设计方案 v0.2.1（平台契约、AMU 数据模型、管线、维度映射） |
| `agent-memory-literature.md` | 前沿文献清单（2024–2026，33 篇） |
| `papers/notes/` | 33 篇逐篇精读笔记（模块与提示词提炼，索引见其中 README） |
| `prompts/` | 可调用提示词模板库（Add×3 / Search×2 / 本地评测×2） |
| `memory_system/` | M1–M3 参考实现（FastAPI + SQLite + LiteLLM），详见其 README |
| `download_papers.py` | 论文批量下载脚本（PDF 不入库，可自行重新拉取） |

## 快速验证

```bash
cd memory_system
pip install -r requirements.txt
AML_FAKE=1 AML_API_KEY=testkey python scripts/selftest_contract.py   # 契约自测 11 项
```

## 关键设计

- **AMU 原子记忆单元**：双时间轴（event_time / valid_from-to）+ 版本链软废止
- **Add 侧重 / Search 侧轻**：LLM 调用前置写入侧，检索侧仅查询理解 + 轻量重排
- **多路召回**：向量（主力）+ BM25 + 图谱 PPR 多跳 + 时序切片 + 画像/规则直通
- **LiteLLM 抽象**：`AML_LLM_MODEL` 默认 `gpt-4o-mini`（平台第一期规则），可配置

## 合规

评测数据仅用于当前任务、30 天内删除；Search 只返回记忆证据、不生成答案；
user_id 存储层行级隔离。
