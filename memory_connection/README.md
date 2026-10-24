# memory_connection：雙側聯想記憶系統

目前狀態：**獨立 v5 研究原型，92 項測試通過，已完成新的 12 段 LongMemEval 完整歷史、96 次搜尋與 192 筆交叉評測**。逐答覆核後雙側準確率由 6/12 升到 9/12，但僅查詢擴展也為 9/12；仍未證明雙側優於最佳單側。原始 Judge 有 24 筆與覆核不同，兩套結果均保留。新保留條件沒有觸發，精判誤刪前提仍存在。詳見 [最新驗證報告](GROUNDING_V5_REPORT.md)，不將測試通過或小樣本改善視為普遍品質保證。

統一符號：**A 是已儲存的原始事件，B 是本次搜尋的原始需求**。A₀=A、B₀=B；下標大於 0 的節點是衍生的聯想視角。

核心流程：

```text
寫入：原始事件 A → 一次有界擴展 → A₀、A₁…Aₘ → 持久化與索引
搜尋：原始需求 B → 一次有界擴展 → B₀、B₁…Bₙ
召回：各 Bⱼ 查詢所有已索引 Aᵢ → 聚合回原始事件 A → 候選連接
精判：比較原始 A 與原始 B → 相關／部分相關／反證／無關／不確定
輸出：有來源的原始記憶證據，並保留可追溯的連接路徑
```

**聯想是檢索入口，不是新增事實；擴展相似也不是原始事件相關的證明。**

文件入口：

- [v5 日期、前提與回答驗證](GROUNDING_V5_REPORT.md)：原始評分與逐答覆核、四組消融、交叉歸因、證據流失及成本。
- [歷史 v4 日期與前提結果](REASONING_V4_REPORT.md)：四組準確率、來源流失、超時敏感度、延遲成本與失敗案例。
- [目前 v5 設計](GROUNDING_V5.md)：來源日期錨定、時間上下界、條件式線索補足及有來源回答器。
- [歷史 v4 設計](REASONING_V4.md)：原始來源日期、互補前提指引及單一來源保留槽。
- [已撤回的 v3 實驗](REASONING_V3_REPORT.md)：較大幅度修改在未測會話退步，保留完整負面結果。
- [完整設計](DESIGN.md)：資料模型、擴展策略、演算法、介面與生命週期。
- [評測與驗收](EVALUATION.md)：四組消融、品質與成本指標、故障案例。
- [實作計畫](IMPLEMENTATION.md)：模組劃分、階段交付與待決策事項。
- [真實模型實驗規範](BENCHMARK.md)：五組對照、模型配置、續跑與成本口徑。
- [候選排序與原文精判優化](OPTIMIZATION.md)：來源多樣性、原文重排、引用編號與凍結的配對實驗。
- [未測會話優化報告](OPTIMIZATION_REPORT.md)：36 題、288 筆前後對照；三組準確率提高，雙側擴展準確率持平。

設計參考本倉庫現有 [契約](../memory_system/app/schemas.py)、[搜尋流程](../memory_system/app/search_pipeline.py)、[目前架構](../memory_system/README.md) 與 [既有問題分析](../memory_system/RETRIEVAL_ANALYSIS.md)。歷史分析用於辨認風險，不代表新設計已取得效能提升。

## 安裝與驗證

在倉庫根目錄執行，Python 3.11 以上：

```powershell
python -m pip install -r memory_connection/requirements.txt
python -m unittest discover -s memory_connection/tests -v
python -m memory_connection.scripts.ablation_smoke
```

消融腳本使用可控的正交向量與裁決 fixture，預期 C00／C10／C01 沒有命中，C11 有一條雙側橋接命中。這是刻意構造的流程測試，不是實際資料集的品質結論。完整驗證紀錄見 [VALIDATION.md](VALIDATION.md)。

## 啟動服務

設定環境變數後，從倉庫根目錄啟動獨立程序：

```powershell
$env:MC_API_KEY = 'local-service-key'
$env:MC_FAKE = 'true'
python -m uvicorn memory_connection.app.main:app --host 127.0.0.1 --port 8011
```

此模式可檢查 HTTP、持久化與詞面召回，但精判一律為 uncertain，因此不返回已驗證證據。正式語義模式設 `MC_FAKE=false`，並設定 `MC_API_BASE`、`MC_PROVIDER_API_KEY`、`MC_MODEL`、`MC_EMBED_MODEL`、`MC_EMBED_DIMENSION`，參考 [.env.example](.env.example)。服務不自動讀取 `.env`；原生向量維度必須與配置一致。

正式 provider 使用 OpenAI 相容的 `/chat/completions` JSON object 與 `/embeddings` API；其他供應商可實作 `Provider` 協定。原文精判使用伺服器產生的引用編號，由伺服器還原原文位置，再經 Pydantic、來源歸屬與逐字引文校驗；模型不需計算字元位置。外部 provider 直接回傳位置時，仍只對同一來源中精確且唯一出現的引文重新定位。可選 `MC_RERANK_*` 配置對原始候選重排；未配置時仍執行來源多樣性排序。API 適配器有 MockTransport 測試，並已通過真實 chat／embedding／rerank 連線驗證。

HTTP 端點：

- `GET /health`：程序狀態。
- `POST /add`：`request_id, user_id, session_id, messages`；成功即完成 SQLite 原子提交。
- `POST /search`：`user_id, query, top_k`；可選 `debug=true` 取得雙側連接及精判 trace。
- `POST /answer`：與搜尋相同的輸入；搜尋後讀取原始證據，回傳答案、來源引用及缺失前提。引用校驗不等於答案語義正確；答案不寫回記憶。
- `DELETE /users/{user_id}`：刪除該使用者的事件、內嵌來源與視角、冪等紀錄，推進 epoch。

除 health 外均要求 `Authorization: Bearer <MC_API_KEY>`。此金鑰是可信平台服務金鑰，可操作所有 user_id；不應當成對外多租戶終端使用者的身分驗證機制。

例如搜尋：

```powershell
$headers = @{ Authorization = "Bearer $env:MC_API_KEY" }
$add = @{ request_id='r1'; user_id='u1'; session_id='s1'; messages=@(@{role='user'; content='我對蝦過敏。'}) } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8011/add -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($add))
$search = @{ user_id='u1'; query='聚餐有蝦餃，安排時要注意什麼？'; top_k=5; debug=$true } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8011/search -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($search))
```

## 本版實作邊界

- 完全不 import `memory_system`，僅使用 `MC_` 配置；預設 DB 在本目錄，embedding 指紋不一致會拒絕開啟。
- SQLite 將來源、事件與視角／向量保存為同一事件 JSON，方便原子發布；向量精確掃描與 BM25 在租戶篩選後執行，尚未接入 ANN／FTS 索引。此基線適合小資料驗證，不宣稱大規模效能。
- 以完整消息為事件，最多帶本次 Add 的前兩則消息作獨立歸屬的上下文。尚未實作跨 Add 指代解析、長文件語義切分；無法容納的精判語境會明確標記 overflow。
- 同 session 的明確 message_id 重送新版本會關閉舊版區間，`valid_from` 必須遞增。沒有明確 ID 的不同 Add 不自動判定事實修訂。valid_from 是版本有效時間，timestamp 是來源陳述時間；不自動推測事件時間。
- `as_of` 按有效區間過濾；`include_history` 可返回歷史。搜尋參考時間優先使用請求，其次可見來源的最新 timestamp，再使用現在時間。
- 聯想文字會去除完全／空白／大小寫重複；尚未做語義近重複合併。精判最多兩批、16 個事件；圖關係與多事件組級精判列入後續階段。
- 原始來源附加可讀 UTC 時間及有限英文相對日期註解，缺失或歧義不捏造；已支援有限年月區間計算，但不保證模型正確綁定事件與日期。多跳提示詞保留部分前提，但真實實驗仍有時間約束被誤刪的案例。
- 舊獨有來源槽預設停用（`MC_EXCLUSIVE_BRIDGE_SLOTS=0`）。新策略只補足查詢明示且前 16 名缺少的實體／引號短語／年份，要求原文重排前 32 名且不損失既有線索，最多替換末兩名（`MC_COMPLEMENTARY_SLOTS=2`）；四組同樣適用。此策略不保證找到未明示的多跳前提，也不保證實際準確率提升。
- 部分證據、裁決失敗、超預算或裁剪均標記 incomplete；聯想與精判內容不持久化成新事實。診斷 trace 只隨回應返回，不落庫；目前沒有查詢快取。
- 聯想生成失敗時保存原文並記錄失敗狀態；尚未加入背景重建佇列。沒有憑證時明確報錯，不自動退回 Fake。
- 記錄各階段耗時、比較數、候選與模型呼叫數；已完成真實模型四組消融與舊系統對照的先導實驗，見 [真實模型報告](REAL_BENCHMARK_REPORT.md) 與 [重現方法](BENCHMARK.md)。24 題、120 筆結果顯示雙側擴展提高候選覆蓋，但未提高端到端準確率；全量資料與多輪穩健性驗證仍待進行。
- 候選排序及原文精判優化後，以三段未參與先導實驗的會話重跑四組前後配對，36 題共 288 筆。C00／C10／C01 準確率分別提高 5.56／11.11／16.67 個百分點；C11 的最終來源召回由 65.28% 升至 73.84%，準確率仍為 63.89%。詳見 [優化驗證報告](OPTIMIZATION_REPORT.md)；不宣稱雙側擴展已優於單側擴展。
- 歷史 v4 對照 v2 的 C11 最終來源覆蓋為 80.56% → 83.33%，但雙方無重排降級的 21 題為 82.54% → 80.95%；準確率均持平。新樣本不同，不能與歷史總分直接比較。LoCoMo10 全部會話已使用；本輪已轉用首次測試的 LongMemEval，跨資料集總分不能直接比較。
