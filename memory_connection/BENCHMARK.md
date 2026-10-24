# 真實模型五組對照實驗

此入口只在評測層調用兩個系統；`memory_connection/app` 沒有匯入舊系統。

已完成 2026-10-24 真實模型先導實驗，結果與限制見 [REAL_BENCHMARK_REPORT.md](REAL_BENCHMARK_REPORT.md)。

## 固定條件

- 原始資料為本機 `memory_system/data/locomo_eval.json`，對照 `locomo10.json` 的 dia_id，逐字校驗消息對齊。
- 固定種子 20261023，抽兩段完整會話，每段 12 題。選題不查看答案／證據標註，歷史不截斷。
- C00、C01 共用未擴展索引；C10、C11 共用已擴展索引。寫入只做兩次，但每組的邏輯寫入成本應各自計入所屬索引建置成本。
- 每題五組啟動順序固定種子打亂；本輪在尚未評分前設定 `--query-concurrency 5`，五組平行查詢。延遲屬於此共同負載下的實測，不是隔離單請求最佳延遲。
- 所有組共用相同 chat／embedding 模型、top_k=10、最終 24,000 UTF-8 bytes 整項證據預算，以及舊系統的共同 Answer/Judge 提示詞。標準答案只進 Judge，不進 Add、Search 或 Answer。
- 舊系統保留原有圖與 reranker；因此四組內的比較是擴展消融，與舊系統的比較則是整體架構比較。
- 這次品質實驗將兩系統 Search 總期限統一為 120 秒；新系統查詢擴展 30 秒、精判 80 秒。這與新系統預設 12 秒服務延遲不同，不能直接代表預設部署效果。
- 新系統 Add 最多 4 個請求並行，舊系統同使用者 Add 串行；報告同時保留呼叫數、token 與牆鐘時間，不能把寫入時間差全歸因於演算法。

## 執行

```powershell
python -m memory_connection.scripts.real_benchmark --config memory_connection/.env --output memory_connection/runs/my-preflight
python -u -m memory_connection.scripts.real_benchmark --config memory_connection/.env --output memory_connection/runs/my-run --prices memory_connection/benchmark-prices-usd.json --query-concurrency 5 --run
```

每次新實驗的輸出目錄必須不存在，避免混合不同 run；網路中斷可使用相同命令加 `--resume`，程式會核對抽樣與配置，重播冪等 Add 並跳過已完成答題。每次續跑的程式雜湊另外保存。程式不接受 Fake 做正式執行；無憑證只寫預檢狀態，不呼叫外部模型。設定檔支援既有 AML_ 與新的 MC_ 變數；金鑰不寫入 manifest。敏感來源、原始 evidence 與逐題結果只存於已忽略的本機 runs/。

產物包括 manifest（資料／程式雜湊與抽樣）、status、calls（每次外部模型用量）、ingestions（索引建置成本）、results（逐題分數／延遲）、evidence（實際輸出與 trace）、summary（組別統計與配對差異）。

離線重算成本、寫入成本攤提及聯想使用情況：

```powershell
python -m memory_connection.scripts.analyze_benchmark memory_connection/runs/my-run
```

## 寫入前校準

第一輪連線遇到 ConnectError，已加入最多三次有界傳輸重試及 HTTP 連線重用。原文索引的成功提交與失敗呼叫均保留。

第一批聯想輸出暴露模型字元位置計算錯誤：引文逐字存在，但 start/end 不正確。原校驗會將所有聯想丟棄。在任何題目評分前，加入「同一來源、精確引文、唯一出現」的確定性位置校正，同時用於聯想錨點與精判引文；不存在或多處出現的引文仍拒絕。重新建立 C1 索引，原 C0 索引可原樣復用。此前中止的 C1 建置成本不混入正式消融指標，但保留為實驗開發成本。

## 指標口徑

- `gold_quote_recall`：已標註來源的完整原句在最終共同證據包中可見的比例。這是保守的來源原句覆蓋代理；短節錄可能語義充分卻無法完整匹配，不能稱作語義召回率。沒有 gold 或 ID 無法映射的題目從分母排除。
- `all_gold_quotes`：必要標註原句是否全部可見；不表示未標註的證據沒有用。
- 準確率：共同 Answer/Judge 模型二元判分，並記錄回答／評分失敗。Search 出錯的題目操作性分數記零，原始 Judge 分數另存。這是本機代理分數，不是官方排行榜成績。
- 延遲：Search 的端到端 p50/p95；Answer/Judge 時間另列。Add 以完整索引建置牆鐘時間與模型呼叫數列出。
- 成本：每次呼叫實際回報的 input/output tokens，乘以定價快照。缺回報、失敗或缺單價標記不完整，不當作零費用。舊日誌的零用量不能區分缺值，因此保守視為未知。
- 準確率差異以 conversation 為 bootstrap 單位，避免把同段歷史的題目當成完全獨立樣本。僅兩段會話的區間很不穩定，只能作探索性結果。
- 既有 LoCoMo 轉換器將原始類別 3、4 的名稱對調。分析器驗證原始 QA 對齊後，以原始數字 ID 列出類別，報表命名採 3=open-domain、4=single-hop；保存結果中的舊名稱保留於 `categories_dataset_labels`。此修正只影響分組名稱，不改動抽樣或分數。
- 離線分析器會核對結果與 evidence 是否一一對應，分別記錄供應商錯誤、精判錯誤、候選／精判池／接受／最終來源召回，以及雙側相對兩個單側的獨有 gold 來源。live 分析可能讀到不同時間的快照；正式結論必須等 `status=complete` 且完整性檢查通過。

## 定價快照

`benchmark-prices-usd.json` 使用美元／百萬 tokens，查閱日期 2026-10-23：

- GPT-4o-mini：輸入 0.15、輸出 0.60。[OpenRouter 模型頁](https://openrouter.ai/openai/gpt-4o-mini/providers)
- text-embedding-v4 北京端點：輸入 0.072。[阿里雲官方模型頁](https://www.alibabacloud.com/help/en/model-studio/text-embedding-v4)
- qwen3-rerank 北京端點：輸入 0.069。[阿里雲官方模型頁](https://www.alibabacloud.com/help/en/model-studio/qwen3-rerank)

這是公開標價估算，不是帳單；未套用快取折扣、免費額度、促銷或帳戶匯率。端點地區若改變須重新取得對應價格。
