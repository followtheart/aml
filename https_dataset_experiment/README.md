# HTTPS 数据集检索实验

仅依赖 Python 3 标准库，通过 HTTPS 调用现有服务的 `/health`、`/add` 和 `/search`。默认校验证书与域名，禁止重定向。客户端只需要服务接口 token，不需要模型供应商密钥。

从仓库根目录执行小样本实验（真实服务会调用已配置的模型，产生相应费用）：

```bash
python3 https_dataset_experiment/run.py \
  --token-file /root/aml-runtime/api-token \
  --data memory_system/data/sample_eval.json \
  --convs 1 --limit 5
```

远程机器可复制本目录，指定本地数据集，并通过 `AML_API_KEY` 环境变量或 `--token-file` 传入服务 token。`--url` 默认 `https://www.dustman.com.cn`。私钥仅在服务端使用。

先检查数据或服务：

```bash
python3 https_dataset_experiment/run.py --dry-run
python3 https_dataset_experiment/run.py --health-only --timeout 15
```

运行已转换的 LoCoMo 或 PersonaMem 数据：

```bash
python3 https_dataset_experiment/run.py \
  --token-file /root/aml-runtime/api-token \
  --data memory_system/data/locomo_eval.json --convs 1 --limit 10

python3 https_dataset_experiment/run.py \
  --token-file /root/aml-runtime/api-token \
  --data memory_system/data/prepared/personamem-v2-32k.jsonl \
  --convs 1 --limit 5 --chunk-messages 20 --timeout 600
```

支持 JSON 数组和逐行 JSONL，每条记录采用项目转换后的结构：

```json
{"conversation_id":"demo","sessions":[[{"role":"user","content":"我的猫叫 Milo"}]],"qa":[{"question":"我的猫叫什么？","answer":"Milo"}]}
```

原始数据需先使用项目 `memory_system/scripts/prepare_dataset.py` 转换。消息角色必须为 `user` 或 `assistant`；保留消息时间戳及来源字段。QA 的 `options`、`question_date`、`as_of` 会传给检索接口，标准答案只保存在实验记录中。

`--convs` 控制会话记录数，`--limit` 控制每条记录的问题数。每次使用独立用户 ID，按 session 顺序分批写入，再使用 `min_revision` 逐题检索。写入失败会跳过该记录的全部问题；查询失败会记录后继续，最终返回非零退出码。请求不自动重试，超时后服务端可能仍在处理。

结果位于 `https_dataset_experiment/runs/<run_id>/`，可用 `--output` 修改：

- `events.jsonl`：写入响应、问题、标准答案、检索完整响应、耗时及错误，逐条落盘。
- `summary.json`：成功/失败数量、非空检索数量及总耗时。

这些是写入与检索实验指标，不是答案准确率或官方评测分数。服务当前没有问答生成接口，本脚本不额外调用模型进行回答或裁判。实验记忆保留在服务端，结果可能包含数据集原文，输出目录已加入本目录的 Git 忽略规则。
