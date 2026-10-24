# 兼容 HTTPS 服务

`memory_connection.app.https_app:app` 使用本目录的双侧联想、召回和原文精判，实现旧服务的 HTTPS 调用契约。应用不导入 `memory_system`，不读取其配置或数据库。`app.main:app` 原生接口仍可独立使用。

当前部署状态（2026-10-24）：`memory-connection-https.service` 已停止并取消开机启动；`memory-system-https.service` 已恢复并启用开机启动。`https://www.dustman.com.cn:443` 当前提供旧系统接口。下述部署参数和验证结果记录本服务此前的配置；代码与独立数据库保留，启动前需处理同端口服务冲突。

## 接口

除 `/health` 外均使用 `Authorization: Bearer <MC_API_KEY>`。

| 方法 | 路径 | 行为 |
|---|---|---|
| GET | `/health` | `{"status":"ok"}` |
| POST | `/add` | 原格式 `request_id/user_id/session_id/messages`，返回提交版本及 epoch |
| POST | `/search` | 原格式 `user_id/query/top_k`，可选 options、时间、历史、敏感内容、版本及证据预算 |
| DELETE | `/memory/{user_id}` | 删除事件、幂等记录、反馈及经验，返回 receipt_id、deleted_at、row_counts |
| POST | `/feedback` | 幂等保存执行反馈，返回 feedback_event_id、memory_ids |
| GET | `/feedback/{user_id}/{event_id}` | 查询当前用户的原始执行反馈 |
| GET | `/experience/{user_id}/{memory_id}` | 查询经验及其版本、状态、引用 |
| POST | `/experience/review` | 引用反馈原文进行版本检查及状态转换 |
| POST | `/answer` | 保留新系统的有来源回答扩展 |
| DELETE | `/users/{user_id}` | 保留新系统的删除别名 |

请求/响应模型独立保存在 `app/https_schemas.py`。`top_k` 必填，范围 1–100；外部写入角色为 user/assistant。source_event_id、trust_scope 等来源字段保存在新数据库。原契约的 nullable options/include_history/min_revision 均可使用。搜索响应保留 data、来源、版本、证据状态等字段；未计算的旧系统 packet_hash 返回 null，不生成虚假摘要。`evidence_token_budget` 延续旧接口的 UTF-8 字节预算含义，按单次请求设置。

接口兼容不代表算法完全一致：

- 写入和检索执行新系统的联想与原文精判；仍受其 512000 字节写入上限及检索/证据预算约束。
- `distill=false` 仅保存反馈；`distill=true` 将原始任务、结果和执行 trace 索引为 workflow 假设，不调用旧系统的经验蒸馏器，不生成额外策略事实。
- 经验从 hypothesis 开始，升级必须有同任务/环境的已验证原文反馈；simulation 不允许升级。review 使用版本检查；rejected/stale 不参与召回。保留引用及人工审核历史，但不复刻旧系统的 helpful/harmful 统计和经验合并算法。
- 新旧数据库独立，旧服务已有记忆未迁移。删除和审核只影响新系统的数据。

每个响应包含 `X-Memory-System: memory_connection` 和服务器生成的 `X-Request-ID`。访问日志仅记录路由模板、状态码、耗时等元数据，每 10 MiB 轮转、保留 5 份，不记录 token 或正文。

## 启动

```bash
python3 -m venv memory_connection/.venv
memory_connection/.venv/bin/python -m pip install -r memory_connection/requirements.txt
export MC_API_KEY='your-service-token'
# 正式模式还需设置 MC_MODEL / MC_PROVIDER_API_KEY / MC_API_BASE
# 以及 MC_EMBED_MODEL / MC_EMBED_API_KEY / MC_EMBED_API_BASE / MC_EMBED_DIMENSION。
export SSL_CERTFILE=/path/to/certificate-chain.pem
export SSL_KEYFILE=/path/to/private.key
bash memory_connection/scripts/start_https.sh
```

默认 `0.0.0.0:443`，可用 HOST/PORT/PYTHON 覆盖；脚本不加载 `.env`，无凭证不会自动退回 fake 模式。MC_FAKE=true 仅用于离线管线检查，不能通过真实召回冒烟测试。

部署模板：`deploy/memory-connection-https.service`。本机使用：

- 地址：`https://www.dustman.com.cn:443`，沿用原证书和服务 token。
- systemd：`memory-connection-https.service`，当前已停止且禁用开机启动；模板配置失败重启。
- 独立环境：`memory_connection/.venv`。
- 数据库：`/root/aml/memory_connection/https.db`。
- 配置：`/root/memory-connection-runtime/service.env`，权限 0600。
- 访问日志：`/root/memory-connection-runtime/access.jsonl`。
- 进程日志：`journalctl -u memory-connection-https.service`。
- 切换前进程参数和环境备份：`/root/memory-connection-runtime/previous-service.json`，权限 0600。

配置中的 chat 为 GPT-4o-mini，embedding 为 text-embedding-v4 / 256 维，rerank 为 qwen3-rerank；使用现有供应商凭证。运行时不再依赖旧目录。服务控制：

```bash
systemctl status memory-connection-https.service
systemctl restart memory-connection-https.service
journalctl -u memory-connection-https.service -n 50 --no-pager
```

## 验证

```bash
memory_connection/.venv/bin/python -m unittest memory_connection.tests.test_https -v
# 设置与服务一致的 MC_API_KEY；默认校验证书及主机名。
memory_connection/.venv/bin/python -m memory_connection.scripts.smoke_https \
  --url https://www.dustman.com.cn --connect-host 127.0.0.1
```

冒烟程序检查 HTTPS 证书、401 鉴权、幂等写入、真实模型检索、反馈及删除，finally 清理独立测试用户。`--connect-host` 仅覆盖连接地址，TLS 仍校验 URL 的域名。原客户端也可继续使用 AML_API_KEY 调用同一域名和端口。

全套 99 项测试包括历史对照测试，其中 `test_benchmark.py` 需要旧系统的可选依赖；新服务只依赖本目录 requirements.txt。部署时全套测试在已有完整依赖的解释器上通过，新兼容接口测试也在独立环境通过。

## 按旧系统口径进行数据集评测

`memory_connection.scripts.evaluate_https` 读取指定历史运行的完整请求和结果，校验数据哈希、题目身份及完整会话，再将原 Add/Search 请求重放到新 HTTPS 服务。仅替换隔离用户、会话和请求 ID；使用历史冻结的客户端回答器及已记录配置。回答和评分在客户端执行，标准答案不发送给记忆服务。

运行时设置 MC_API_KEY 和客户端 AML_LLM_API_KEY / AML_LLM_API_BASE，使用具备旧回答器依赖的 Python：

```bash
PYTHONDONTWRITEBYTECODE=1 AML_FAKE=0 AML_LLM_MODEL=gpt-4o-mini \
  AML_EMBED_MODEL=dashscope/text-embedding-v4 AML_CE_MODEL=qwen3-rerank \
  memory_system/.venv/bin/python -m memory_connection.scripts.evaluate_https \
  --baseline memory_system/runs/https-validation-baseline-20261024 \
  --output memory_connection/runs/https-validation-new \
  --url https://www.dustman.com.cn --connect-host 127.0.0.1 \
  --server-db /root/aml/memory_connection/https.db

python3 -m memory_connection.scripts.analyze_https memory_connection/runs/https-validation-new
```

`--server-db` 仅用于只读审计新数据库中原文是否逐字入库；Add/Search 一律通过 HTTPS。默认并发为 2，单用户写入按原分块顺序提交。结果目录包含请求/响应、逐题答案、延迟、客户端模型用量、原文审计、清理回执及历史对比；目录权限 0700，并已加入 Git 忽略。结束时调用 DELETE 清理本次测试用户。生产服务的模型用量未通过接口暴露，因此只报告客户端 reader 的 token 用量及估算费用，不将其称为全链路费用。

历史分数对比包含重新入库与检索差异，不能视为仅切换传输层的严格因果实验，也不是官方榜单分数。

## 空结果诊断与修复

HTTPS 搜索在原有 `coverage_manifest` 内返回 `metrics`、`degradation`、按原因汇总的 `verification_errors` 和 `empty_reason`，不增加顶层契约字段，也不返回内部提示词或调试正文。候选数、已验证数、重排调用数及文档分片数可用于区分无匹配、模型/预算失败和确实无相关证据。

长候选重排使用保留角色、时间与原文字符范围的分片；按最高片段分数排序原事件，最终精判仍使用完整原文。选择题为各选项增加明确标记为待核实假设的检索视角，并在重排与精判时检查选项前提。精判最多 16 个候选，默认最多 8 个有界批次，不改变单批字节上限及总截止时间。

模型过量返回句子引用时，只有全部引用均逐字有效、属于同一事件，并可无损合并到最多 8 段的情况才保留；不会截断引用列表或放宽来源校验。无相关证据、跨用户查询以及验证失败仍允许返回空结果。

完整复测可以使用 `evaluate_https --archive-index` 保存该次隔离用户的索引事件，便于固定输入的后续诊断；不会复制其他生产用户的数据。

`evaluate_https --reuse-index <已归档运行目录> --server-source <隔离服务源码目录>` 支持对已恢复到隔离服务的索引做 HTTPS Search/Answer 复测，不重复 Add；manifest 明确记录 `fresh_ingestion=false`。此模式只应指向准备好的隔离数据库，运行结束仍会删除测试用户。
