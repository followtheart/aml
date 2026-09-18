# Search pipeline 九項修正與驗證

2026-09-18，搜尋策略版本：`bounded_search_v6`。
本次針對既有搜尋鏈路修正，不改動部署 `.env` 或正式記憶資料。

| 項目 | 實作後行為 | 主要位置 |
|---|---|---|
| 1. 單輪規劃無法利用新發現實體，查詢上限擠掉選項 | 保留原問題、全部選項和最多三個子問題；多跳查詢可用首輪可見證據補查一次，預設最多兩條，帶種子與需求 ID，並預留重排預算 | `retrieval_queries.py`、`search_pipeline._followup`、`04b_query_followup.txt` |
| 2. 以 AMU 數判定直接召回足夠，向量失敗拖垮搜尋 | 按每條查詢匹配的獨立來源事件計數；同一來源拆出的多個 AMU 不重複計數，無關聯結來源不湊數；詞面召回先執行，embedding 失敗明確降級 | `search_pipeline._lexical_recall`、`_recall`、`_expand_recall` |
| 3. 圖預過濾切斷路徑，場景只服務第一條查詢 | 保留最多三跳的連接路徑與種子，限制 120 條三元組；區分正反向及關係詞權重；場景在所有查詢與場景排名間輪流分配配額 | `graph.py`、`search_pipeline._expand_recall` |
| 4. 同源多路反覆投票 | 詞面族、擴展族各取最大貢獻；融合採最佳族貢獻加最多兩族各 15% 貢獻，保留投票診斷 | `search_pipeline._rrf`、`_route` |
| 5. 跨批分數漂移、0.5 突變、只在完全同分時去重 | 多批共用最多兩個真實候選作校準，維持已知零分；正常門檻固定為 0.15，相近分數區間內考慮覆蓋與來源多樣性 | `search_pipeline._rerank_batches`、`_calibrate_scores`、`_selection_order` |
| 6. 部分重排成功反而丟掉失敗批次 | 保留成功分數，失敗批次以 `score=null`、`score_kind=unscored` 有界回退；預設上限八條，不把 RRF 值當相關性；零分不復活 | `search_pipeline._filter_rerank`、`_select_evidence`、`evidence_packet.pack` |
| 7. 重排預覽與交付來源／依賴不一致 | 先解析來源、依賴版本及連接種子，建立固定證據單元，再送重排與裝包；僅可去除包內已可見的重複來源片段；正文中的來源分隔符不再導致截斷 | `evidence_units.py`、`search_pipeline._prepare_candidates`、`evidence_packet.py` |
| 8. 長文件整條貪婪裝包，沒有缺口報告 | 使用完整原文句段與 offsets；必要引用無法放入時明確排除；裝包考慮查詢覆蓋、成本、相關性與剩餘需求，允許後續可容納候選遞補；覆蓋從實際可見文字重算 | `evidence_units.py`、`search_coverage.py`、`evidence_packet.pack` |
| 9. 複製整個資料庫、反覆解碼向量及全量排序、阻塞事件迴圈 | 普通檔案搜尋改唯讀 WAL 快照；按資料庫／使用者／向量空間／epoch／向量代數快取，精確分塊 top-k，只補齊命中正文；近似搜尋需明確啟用；背景執行與協作式取消，清理後再次檢查刪除 epoch | `provenance.py`、`store.py`、`vector_index.py`、`local_work.py`、`budget.py` |

證據包 hash 為 v4，新增覆蓋 ID 與分數類型；既有 v1/v2/v3 仍可讀取。
`coverage_manifest` 新增查詢需求的 `packed`／`candidate_only`／`missing` 狀態，以及校準、補查、向量搜尋及降級診斷。
有已知降級且無可交付證據時，`evidence_status=incomplete`；正常空結果才用 `not_found`。

驗證結果：**23/23 組自測通過，共 347 項**（331 項 unittest 加 16 項 API 契約檢查）。
其中新增 `selftest_search_repairs.py` 19 項及 `selftest_search_storage.py` 24 項。
自測使用獨立子程序、暫存資料庫及 Fake 模型；不讀部署 `.env`，不寫正式 debug log。
完整輸出：[selftests.txt](runs/search-repairs-20260918/selftests.txt)。

獨立審查曾重現並修正：第六選項消失、同源多 AMU 誤判覆蓋、無關來源湊數、跨批比例漂移、
失敗批次消失、長文丟失、補查未保留連接種子、節錄沿用失效覆蓋、正文分隔符截斷、
取消時過早關閉快照、跨使用者快照讀取，以及 purge 期間舊索引重新寫入快取／舊證據交付。
既有來源引用、敏感權限、遺忘、版本、API、Add 與評測契約回歸亦通過。
`compileall` 與 `git diff --check` 已通過；Windows 換行以 `cr-at-eol` 檢查。

重現命令（倉庫根目錄）：

```powershell
.\memory_system\.venv\Scripts\python.exe memory_system/scripts/run_selftests.py
.\memory_system\.venv\Scripts\python.exe -m compileall -q memory_system/app memory_system/scripts
git -c core.safecrlf=false -c core.whitespace=cr-at-eol diff --check
.\memory_system\.venv\Scripts\python.exe memory_system/scripts/benchmark_search_vectors.py
```

向量檢查使用固定種子、64 維正規化隨機向量、20 條查詢、top-10，近似候選額度 8,192：

| 向量數 | 實際搜尋模式 | 平均 Recall@10（相對精確 top-10） | 每條查詢最多評分數 |
|---|---|---:|---:|
| 10,000 | 精確 | 1.000 | 10,000 |
| 25,000 | 明確啟用近似 | 0.825 | 8,192 |
| 100,000 | 明確啟用近似 | 0.545 | 8,192 |

因此 **`AML_VECTOR_APPROXIMATE` 預設為 0**；合成資料顯示的漏召回不適合未經評估就改為預設。
測試中的大池精確搜尋也已與 NumPy top-k 對照。完整數據與機器耗時：[vectors.json](runs/search-repairs-20260918/vectors.json)。
這些數據不代表真實記憶資料的召回率、端到端延遲或回答正確率。

| 新設定 | 預設 | 用途 |
|---|---:|---|
| `AML_SEARCH_FOLLOWUP_QUERIES` | 2 | 每次補查的查詢數，0 關閉補查 |
| `AML_SEARCH_FOLLOWUP_SECONDS` | 8 | 補查規劃的秒數上限 |
| `AML_RERANK_CALIBRATION_ANCHORS` | 2 | 共用校準候選數，範圍 0–2 |
| `AML_SEARCH_ITEM_MAX_BYTES` | 6000 | 單一證據單元上限 |
| `AML_SEARCH_LOCAL_CONCURRENCY` | 4 | 背景工作執行緒数 |
| `AML_VECTOR_CACHE_BYTES` | 134217728 | 索引快取估計額度，0 關閉快取 |
| `AML_VECTOR_CHUNK_SIZE` | 2048 | 向量評分分塊大小 |
| `AML_VECTOR_APPROXIMATE` | 0 | 是否允許近似向量搜尋 |
| `AML_VECTOR_EXACT_LIMIT` | 20000 | 近似模式下仍使用精確搜尋的合格候選數門檻 |
| `AML_VECTOR_CANDIDATE_LIMIT` | 8192 | 近似模式評分候選額度，至少為請求 k |

仍需保留的限制：

- 本次沒有執行真實 LLM／LoCoMo 品質評測，不能据自測宣稱回答準確率提升。共用錨點只校準簡單尺度差異，分數不是機率。
- 覆蓋標記反映詞面命中或有種子依據的補查，不保證因果鏈正確、選項為真或證據充分；回答端仍需判斷。
- 打包為有界啟發式，不保證全域最優。無法在預算內容納完整必要引文或依賴的單元會排除並回報。
- 預設精確向量計算、合格 ID 查詢、首次索引及歷史 `as_of` 投影仍隨資料量增長。快取額度是索引記憶體估計值，並非整個程序的記憶體硬上限。
- 取消需先停止背景工作再清理快照；本地函式或原生運算完成目前區塊前，清理可能延後，不能把設定秒數視為絕對返回時刻。
- `evidence_token_budget` 沿用 UTF-8 bytes 保守上界，不是供應商 tokenizer 的精確 token 數。
