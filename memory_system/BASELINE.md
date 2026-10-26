# 当前恢复的 HTTPS 基线

2026-10-27 按用户要求恢复并保存基线状态。源码对应提交
`a430d57c305dbf76763da10a5902c7ae764cb880`，与下面历史测评登记的 839 份源码文件逐一核对一致。
恢复时运行时代码已与该版本一致；未提交的实验草稿与文档改动已归档。

| 历史测评范围 | 基线正确题数 | 正确率 |
| --- | --- | --- |
| 回归集 | 38/56 | 67.86% |
| 验证集 | 17/61 | 27.87% |
| 合计 | 55/117 | 47.01% |

这里的“接近 70%”对应 56 题回归集的历史成绩。历史分数不保证再次调用模型时完全复现。

测评记录：

- `runs/https-screened-source-20261026/final-decision.json`
- `runs/https-screened-source-20261026/regression/control/summary.json`
- `runs/https-screened-source-20261026/validation/control/summary.json`
- `runs/https-screened-source-20261026/regression/plan.json`（源码摘要清单）

保留 24,000 字节主审预算和既有 fine 语境预留配置。来源后置准入、联合来源排序以及其他后续实验均未启用到当前基线。
恢复后 Git 工作区干净，公网 HTTPS 服务健康检查返回 `{"status":"ok"}`。

恢复前草稿备份：`runs/baseline-restore-20261027-004224/`。
后续刷新同口径最高正确率时，按仓库 `AGENTS.md` 立即保存对应代码提交。
