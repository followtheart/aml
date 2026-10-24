# 實作計畫

使用者已授權按設計實作。M0–M2 的獨立基線已完成，實際模組與可執行命令見 README；以下保留完整階段規劃。M3 已完成兩段完整歷史、24 題的真實模型先導消融，結果見 REAL_BENCHMARK_REPORT.md；全量、多輪驗證及 M4 大規模能力仍待執行。

## 1. 預計目錄

```text
memory_connection/
  app/
    schemas.py           Add/Search、事件、視角、連接、裁決契約
    config.py            MC_ 設定、上限與模型版本
    providers.py         模型／embedding 可替換介面與呼叫計量
    store.py             SQLite 原文、事件、版本與冪等交易
    event_builder.py     來源切分、上下文及精確來源範圍
    expansion.py         寫入／搜尋兩種擴展及去重
    index.py             向量／全文索引介面與租戶過濾
    recall.py            視角匹配、事件聚合、連接 trace
    verification.py      原始 A/B 精判與引用校驗
    packing.py           有界證據輸出、衝突與組完整性
    service.py           Add/Search 編排與期限
    main.py              FastAPI 介面與鑑權
  prompts/               三個任務的版本化提示詞
  tests/                 契約、行為、隔離及失敗路徑
  scripts/               消融、效能測試、診斷與來源遷移
```

## 2. 交付順序

### M0：契約與原始證據層

建立獨立套件、資料庫 schema、來源切分、revision／冪等／刪除語義，以及不帶衍生視角的 C00 流程。驗收來源可追溯、角色保留與租戶隔離。

### M1：雙側擴展與可重放召回

實作兩種擴展、錨點校驗、批次 embedding、全文與小資料精確向量索引，開關可切 C00／C10／C01／C11。事件聚合必須在路徑配額之前，輸出完整連接 trace。

### M2：原始事件精判與服務

實作原文精判、五種裁決、來源校驗、期限、部分失敗、證據打包及 Add/Search HTTP 介面。接入真正模型前，先用可控 provider fixtures 驗證拒絕誤連接、反證保留與失敗標記。Fake 只能驗證流程，不能報語義品質。

### M3：真實模型消融與成本基線

固定資料、模型、提示詞與 code revision，跑四組消融與 legacy 基線。依 EVALUATION.md 產生來源層級報告、失敗分類及品質／成本比較，決定是否採用 C11。

### M4：資料量擴展與可選能力

依容量測試接入可過濾租戶的 ANN；再按漏召回案例決定是否需要有來源的圖關係、一跳擴展與多事件組級精判。外部索引引入時同時完成 outbox、tombstone 與版本切換。

## 3. 舊系統復用原則

可參考既有請求契約、供應商適配經驗、來源範圍處理及評測格式；經介面測試後再抽取公共元件。避免直接 import 舊搜尋管線及其全域設定，讓新系統的效果可以獨立歸因。

遷移從 messages／原始來源重建事件與視角；保留舊 source ID 對照與角色資訊。先在獨立庫重播，驗證來源數、權限、遺忘狀態與時間版本，再做影子搜尋比較。不能從已刪除或無授權的歷史輸出恢復資料。

## 4. 待實作前評審的決策

預設採 Python＋FastAPI＋SQLite，延續倉庫使用方式；正式 embedding、生成模型、資料容量與預算由部署配置決定。設計不依賴特定模型，但相似度品質與延遲需要真實測量。

目前實作沿用此規劃；`index.py` 的小資料精確檢索暫合併於 `recall.py`，SQLite 以事件 JSON 保存來源與視角，未建立外部索引。實作差異與限制在 README 明列；驗證紀錄見 VALIDATION.md。
