# 重排修復與選中策略：single_pass_v5

2026-09-18 完成。範圍為 J 階段的批次評分可靠性與評分後選中；沒有重跑 Add、修改現用 `.env` 或覆寫實驗日誌。

## 根因與修復

- ROOT_CAUSE：scores schema 未限定批次長度；1–2 個候選的尾批曾回傳 4 個分數。單批例外直接傳出 TaskGroup，取消其他批次，最後全局 RRF 回退；選中階段又把任何重排錯誤都當成全局降級。TaskGroup 的例外取消行為符合 [Python 官方文件](https://docs.python.org/3/library/asyncio-task.html#task-groups)，問題在於本專案把可隔離的單批錯誤交給整組處理。
- CATEGORY：LOGIC_ERROR。
- FIX_PLAN：限定輸出長度與平衡小尾批；隔離失敗並有界修復；保留成功評分；同分時使用可見來源資訊安排選中次序。
- NEXT_ACTION：RETRY_TASK，已在授權範圍內完成程式修復與驗證。
- CONFIDENCE：HIGH，指協定、回退及選中行為；不代表真實模型準確率已提高。

## 實作行為

| 範圍 | 新行為 |
|---|---|
| 輸出契約 | 每批 schema 的 `minItems`、`maxItems` 等於候選數；保留本地長度、型別、有限值及 0–1 範圍校驗 |
| 提示詞 | 明確要求評分記憶候選，不評分答案選項；移除固定長度陣列範例 |
| 尾批 | 在原有候選數／bytes 上限內與前批重新平衡；不增加批次、不刪減引用、不改候選順序 |
| 格式恢復 | 首輪所有批次完成後，格式錯誤最多額外修復 1 批；可用 `AML_RERANK_REPAIR_MAX_CALLS` 調整，0 可關閉 |
| 預算 | 修復仍受 Search 呼叫數、token、重排截止時間約束；一般連線錯誤不額外重試 |
| 部分成功 | 保留成功分數，失敗批次標為 `batch_failed`；普通未評分項不填包，使用者規則仍受保護 |
| 完全失敗 | 只有完全沒有可用評分才全局 RRF 回退；不混合模型相關性分數與 RRF 排序分數 |
| 選中同分 | 模型分數優先；同分時優先有可見使用者原文的個人線索，再優先尚未覆蓋的來源，最後保留原次序 |
| 同源記憶 | 重複來源延後，不因同源就刪除不同陳述；不提升零分或未評分項 |
| 可觀測性 | `rerank_status` 為 ok／recovered／partial／fallback／not_run；保存每次嘗試、selected_ids、tie_policy 與來源特徵 |

部分成功沿用原相關性閾值及弱證據上限，分別記為 `partial_relevance`、`partial_weak_fallback`；評測把 partial 與 fallback 都標為 `search_degraded=true`。格式恢復後全部評分完成則標為 recovered，先前錯誤保留在批次 attempts 中。

來源標記取自重排實際使用的片段，而非僅看記憶類型。短潤色請求及只有 assistant 正文的候選不會因 preference 標籤取得使用者原文優先權。來源角色只是同分排序訊號，不證明內容真實或屬於使用者本人。

## 新鮮驗證證據

- STATUS：VERIFIED。
- CLAIM_TYPE：FIX／TEST_OR_BUILD。
- CLAIM：單批錯誤不再抹去其他成功評分；格式恢復受限；同分選中使用可見來源並保留原分數及規則保護。
- EVIDENCE：21 組檢查全部 exit 0，含 288 項 unittest 與 16 項 API contract，共 **304 項**。
- GAPS：未呼叫真實評分／回答模型，不能據此宣稱新的答案準確率或人工噪聲率。

新增 17 項檢查覆蓋：實際 1／2→4 分數錯誤、schema 長度、尾批平衡、位置對齊、局部失敗、截止時間、外部取消、呼叫預算、包裝後的格式例外、全零評分、同分使用者來源、來源多樣性、同源不同陳述、零分不提升、規則保護及 partial／recovered 回應與證據包。

測試均在 `AML_FAKE=1` 下執行，並關閉實驗 debug 日誌輸出；有關 provider 的測試使用 mock。記憶日誌、搜尋日誌及 26 題結果檔的 SHA-256 與本次修復前一致。

## 固定舊分數的選中重放

使用 v4 本次 26 題的候選與既有分數，只重放選中與分批。沒有重新取得模型分數、裝包或回答。

| 指標 | 原 v4 | 套用新策略 |
|---|---:|---:|
| 既定人工原文線索的承載項被選中 | 18 題 | 19 題 |
| Q22 使用者黑白電影提問被選中 | 否 | 是 |
| Q13 批次大小 | 24／24／24／2 | 24／24／13／13 |
| Q21 批次大小 | 24／24／24／2 | 24／24／13／13 |
| Q24 批次大小 | 24／24／24／1 | 24／24／12／13 |

選中集合改變的題目是 Q10、Q18、Q22。Q22 恢復的 AMU 是 `amu_6d7963da98ec49eb`，來源為 S3:17 的使用者老黑白電影提問。全部三個已知出錯尾批在此次分批重放中都避免了 1–2 項微小批次；模型是否還會輸出錯誤仍須實測。

**18→19 是來源線索在「選中」階段的覆蓋，不能解讀為答對題數。** 同分偏好仍可能選中無關的使用者提問；例如重放也改變了 Q10／Q18 的弱證據組成，不能只因來自使用者就判定更相關。模型給資產 projects、湖水細菌等噪聲高分的問題，也不會由同分策略自動修復。

Q14 的漫展證據在重排池外，本輪沒有調整入池策略，故此問題仍需獨立處理。來源主體抽取、畫像錯誤與回答端的證據遵循也仍屬其他階段。

## 檔案與重放

- 核心：[search_pipeline.py](C:/Users/aapoo/Desktop/aml/memory_system/app/search_pipeline.py)、[llm.py](C:/Users/aapoo/Desktop/aml/memory_system/app/llm.py)。
- 回歸：[selftest_rerank_selection.py](C:/Users/aapoo/Desktop/aml/memory_system/scripts/selftest_rerank_selection.py)、[selftest_retrieval_quality.py](C:/Users/aapoo/Desktop/aml/memory_system/scripts/selftest_retrieval_quality.py)。
- [驗證彙總](C:/Users/aapoo/Desktop/aml/memory_system/runs/rerank-selection-checks/summary.json)。
- [完整選中重放](C:/Users/aapoo/Desktop/aml/memory_system/runs/rerank-selection-checks/replay.json)。

重放命令（專案根目錄）：

```powershell
.\memory_system\.venv\Scripts\python.exe memory_system\scripts\replay_selection.py
```

下一次真實實驗應固定同一記憶快照、另存結果與日誌，分別比較重排錯誤率、選中來源線索覆蓋、人工相關性與答案準確率。
