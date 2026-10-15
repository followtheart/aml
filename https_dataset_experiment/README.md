# HTTPS 数据集实验（平台镜像）

仅依赖 Python 3 标准库，按 [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/zh-cn/docs) 的流程 **Add → Search → Answer → Judge** 调用已部署服务，默认校验证书与域名、禁止重定向。客户端只需要服务接口 token；Answer/Judge 模型可选，另行配置。

默认行为与平台一致：

- **Add**：每个 session 一次请求；超过 20 条消息或 2000 词时按消息/句子边界分块（`--max-messages/--max-words`）。消息只带 `role/content/timestamp`。408/409/425/429/5xx/524 以**相同 `request_id` 与 payload** 重试，最多 32 次，尊重 `Retry-After`（≤60 s）。
- **Search**：只发送 `{query, options?, user_id, top_k=100}`。`reference_time / as_of / min_revision` 等非契约字段默认不发送，加 `--extended` 才会带上（用于与旧实验对比）。408/425/429/5xx 有界退避重试，409 与其他 4xx 直接失败。
- **Answer**：按返回顺序取 `data[].content` 的 token 前缀（默认 117,760，与平台窗口一致，token 以字符估算），与问题、选项一起交给 `--answer-model`。
- **Judge**：选择题本地比对字母；开放题用 `--judge-model` 做 CORRECT/WRONG 判定；rubric 题逐条打分。没有模型时仍输出 `gold_hit_rate`（标准答案是否出现在返回内容中，忽略方括号头部与 "unknown" 类占位答案），作为无成本的召回代理。

从仓库根目录执行小样本实验（真实服务会调用已配置的模型，产生相应费用）：

```bash
python3 https_dataset_experiment/run.py \
  --token-file /root/aml-runtime/api-token \
  --data memory_system/data/sample_eval.json \
  --convs 1 --limit 5
```

接入 Answer/Judge（任意 OpenAI 兼容端点；`AML_EVAL_API_KEY` 或 `OPENAI_API_KEY`，`--eval-api-base` 或 `AML_EVAL_API_BASE`）：

```bash
export AML_EVAL_API_KEY=...
python3 https_dataset_experiment/run.py --token-file /root/aml-runtime/api-token \
  --data memory_system/data/locomo_eval.json --convs 1 --limit 10 \
  --answer-model gpt-4o-mini --judge-model gpt-4o-mini
```

先检查数据或服务：

```bash
python3 https_dataset_experiment/run.py --dry-run          # 校验数据并统计分块后的 Add 请求数
python3 https_dataset_experiment/run.py --health-only --timeout 15
```

支持 JSON 数组和逐行 JSONL，每条记录采用项目 `memory_system/scripts/prepare_dataset.py` 转换后的结构：

```json
{"conversation_id":"demo","sessions":[[{"role":"user","content":"我的猫叫 Milo","timestamp":1677628800000}]],
 "qa":[{"question":"我的猫叫什么？","answer":"Milo","category":"single-hop"}]}
```

QA 可带 `options/gold_labels/qa_type`（选择题）、`rubrics`（rubric 题）、`scoring`、`category`、`question_date`、`unanswerable`。`--convs` 控制会话记录数，`--limit` 控制每条记录的问题数，`--add-concurrency/--search-concurrency` 控制并发（平台把并发上限视为上界而非保证速率）。写入失败会跳过该记录的全部问题；查询失败会记录后继续，最终返回非零退出码。

结果位于 `https_dataset_experiment/runs/<run_id>/`，可用 `--output` 修改：

- `events.jsonl`：每次 Add/Search 的响应、重试次数、耗时；每题的标准答案、预测、分数、`gold_hit`，以及服务端诊断（`evidence_status`、`search_degraded`、`rerank_status`、预算与来源、CE 批次数、listwise 调用数）。
- `summary.json`：成功/失败数、`evidence_status` 分布、各阶段耗时 P50/P95、重试总数、总体与按 `category` / 能力字母的准确率与 `gold_hit_rate`、评测模型 token 用量。

能力字母（A 事实回忆、B 组合推理、C 时序、D 治理、E 个性化、G 上下文执行、H 安全隐私）由 `category` 启发式映射，**不是官方映射**；可用 `--capability-map map.json` 覆盖。本地 Answer/Judge 是平台 Answer 模型的代理，不代表官方分数。实验记忆保留在服务端，结果可能包含数据集原文，输出目录已加入本目录的 Git 忽略规则。
