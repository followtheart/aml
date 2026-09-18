# Fine 精排修復驗收：graph_cascade_v9

本次修復前輪分析的六項選擇、覆蓋、查詢與逾時回退缺陷，並補上逐候選選擇原因。沿用既有 Rerank API 與 `qwen3-rerank` 設定，沒有呼叫外部模型或重跑付費評測。

## 實作與行為

| 原缺陷 | 現在的行為 | 主要檔案 |
|---|---|---|
| 0.1 分桶讓低分新來源擠掉高分互補證據 | 移除分桶；普通名額依原始分數排序，只有可見原文已完全被選中項涵蓋時才延後 | `app/evidence_selection.py` |
| 詞彙命中直接取得硬保留 | `_coverage_ids` 仍作軟診斷；硬保留改用 `_supported_coverage_ids`，必須有可定位的 user 原文見證，檢查主體、否定、歷史、引述及假設範圍 | `app/search_coverage.py` |
| CE query 拼入所有完整建議答案 | 保留原問題，僅加入可安全分離的選項前提與有限子問題；明確標記前提尚待驗證，保留否定與時間限定 | `app/rerank_query.py` |
| 以過期來源 ID 判斷新穎性，同源重述重複占位 | 使用實際 `_rank_text` 中可見的來源片段；部分重疊不等於冗餘；不同時態、狀態、極性與額外圖依賴不合併 | `app/evidence_selection.py` |
| fine 與 listwise 各自重新計算保留名額 | fine 選中的保留候選 ID 傳至 listwise，取消 `count//2` 二次削減；名額或 prompt/token 預算不足時記錄 `reservation_missing` | `app/cascade_rerank.py` |
| 逾時只救 coarse 前兩項 | 原時限和剩餘額度內小批重試尚未得分的逾時候選；未恢復項按查詢分面取得有界救援機會 | `app/cascade_rerank.py` |

`search_pipeline.py` 在原子證據單元準備後，用實際可見來源建立見證，再傳入級聯。日誌新增各階段選擇原因、保留需求、代表候選、可見來源、CE 名次、逾時重試與恢復 ID；執行版本改為 `graph_cascade_v9`。

硬保留見證仍是保守的詞彙與來源範圍判定，**不是完整語義蘊含驗證**。未取得硬保留的候選仍能按普通分數入選，不因此直接刪除。

## 預算與設定

新增預設值已寫入程式及 `.env.example` 註解：

```dotenv
AML_CE_RETRY_BATCHES=2
AML_CE_RETRY_BATCH_SIZE=4
```

重試次數限制為 0–2，每批設定限制為 1–8，且仍受原 `AML_CE_BATCH_SIZE` 限制。設定次數為 0 可停用。重試共用原 CE 階段期限、請求數與 token 額度，保留原有 listwise 呼叫及 token 空間；不重送已成功候選。已涵蓋內建 `TimeoutError` 與真實 HTTP `ReadTimeout`。

fine 與 listwise 仍有明確容量上限；本次沒有靠提高總 budget 或放大 Top-K 掩蓋選擇缺陷。現有 `.env` 不需要增加上述設定便會採用預設值。

## 新鮮驗證

在儲存庫根目錄執行：

```powershell
.\memory_system\.venv\Scripts\python.exe memory_system/scripts/run_selftests.py
.\memory_system\.venv\Scripts\python.exe memory_system/runs/fine-repair/replay.py
git diff --check
```

- 全部 **30／30 組通過**：440 個單元測試及 16 個 HTTP 合約檢查，共 456 項，exit 0。
- 新增選擇 17 項、逾時／救援 4 項、覆蓋／查詢 31 項測試。涵蓋真實 HTTP transport 的逾時型別，但使用 mock transport，不連外。
- RED 證據包含原分桶與來源選擇失敗、逾時未重試、香料來源未救援、第三人引述／虛構背景／跨謂詞否定錯接。各對應修復皆已 GREEN。
- 兩個獨立審查範圍（selector／cascade、coverage／query）均取得 `APPROVED`；最新時態／狀態相容性補充也已複審。
- `git diff --check` exit 0；原始四份輸入的 SHA-256 在重放前後相同。

驗證材料：[完整測試輸出](C:/Users/aapoo/Desktop/aml/memory_system/runs/fine-repair/full-selftests-final.txt)、[選擇 RED](C:/Users/aapoo/Desktop/aml/memory_system/runs/fine-repair/selection-red.txt)、[HTTP 逾時 RED](C:/Users/aapoo/Desktop/aml/memory_system/runs/fine-repair/http-timeout-red.txt)、[覆蓋邊界 RED](C:/Users/aapoo/Desktop/aml/memory_system/runs/fine-repair/coverage-review-red.txt)。`runs/` 是本地驗收產物目錄，不納入版本控制。

## 26 題固定分數重放

固定原 v8 的 coarse 候選、來源正文與 CE 分數，重新計算新版 coverage、fine 與 listwise 名額。觀測對象仍是前輪標定的 42 組來源線索，包含弱線索、否定與第三人材料，不等於 42 組 Gold 支持。

| 關口 | 舊版 | 修復後 |
|---|---:|---:|
| fine 的來源線索／覆蓋題數 | 30／19 | 31／20 |
| 再過 listwise 10 個名額的來源線索／覆蓋題數 | 27／19 | 30／19 |
| 平均 CE query UTF-8 bytes | 1,554.81 | 400.54 |

fine 新恢復 Q1/S1:12（學生困境）、Q10/S6:5（Milan 原文）、Q24/S2:14（香料原文）；同時失去 Q10/S11:11、Q20/S4:9。Q1 原本只靠泛詞命中的 Facebook 候選不再取得硬支持，學生來源取得可定位的 `option:3` 見證。

這個重放**沒有重新計算 CE 分數、沒有模擬新版 prompt 字節裁剪、沒有重新執行 LLM 排序或回答**；數字只驗證固定候選下的名額行為，不能推算原先 9／26 答題率會提高多少。新查詢的實際排序品質仍需後續受控評測。

重放材料：[程式](C:/Users/aapoo/Desktop/aml/memory_system/runs/fine-repair/replay.py)、[逐題結果及程式／輸入指紋](C:/Users/aapoo/Desktop/aml/memory_system/runs/fine-repair/replay.json)。尚未進入同一原子單元的一般多跳圖路徑，仍沒有新增的全局路徑聯合選擇器；本次只保護既有單元與依賴，沒有宣稱解決所有圖路徑組合問題。

## Verification Result

- STATUS: VERIFIED
- CLAIM_TYPE: FIX
- CLAIM: 六項 fine 缺陷的程式修復、離線回歸與固定分數重放已完成。
- EVIDENCE: 30／30 suites、456 項檢查通過；兩個分工範圍獨立審查通過；26 題重放與原始輸入 hash 校驗通過。
- GAPS: 未進行外部模型重評，不包含端到端答題率提升或完整語義蘊含保證。
