# ULM 落地與驗收清單

依據：[`memory-system-design.md`](../llm-memory-survey/memory-system-design.md)。使用者已明確要求依該設計實作；此清單追蹤程式碼落地，不取代或重新要求設計審批。

## 工作項目

- [x] 1.1 來源、雙時間版本與刪除一致性
  - _Boundary:_ `app/store.py`、`app/add_pipeline.py`、`app/integrity.py`、`app/provenance.py`、Add schema、對應回歸測試。
  - _Design:_ §2.1、§2.5、§3.1–3.5、§8。
  - 驗收：延遲告知與追溯纠錯可按認知時間查詢（`selftest_versioned_storage.py`：world_change/correction/as_of 投影）；更新保留版本（`claim_versions` 版本鏈）；來源可追蹤（`source_messages`/`amu_sources`/`source_identity`）；刪除世代不重設、進行中的寫入不可復活資料（`deletion_epochs`、`selftest_lifecycle_safety.py`）；Add 返回寫入版本（`/add` 回傳 `write_revision`/`scope_epoch`，`selftest_contract.py` 新增斷言）。
  - 本次修正：無時間戳消息不再合成單調時間、不再以牆鐘充當 `valid_from`/錨點（§2.1 缺少錨點返回未知）；移除 `_with_synthetic_timestamps` 死代碼與 `AML_SYNTHETIC_TIME*` 配置。
- [x] 1.2 有預算的最終證據契約與按需檢索
  - _Depends:_ 1.1。
  - _Boundary:_ Search schema、`app/search_pipeline.py`、`app/answer_context.py`、`app/graph.py`、`app/llm.py`、`app/budget.py`、`app/evidence_packet.py`、設定、對應測試。
  - _Design:_ §5.1–5.6、§9.4。
  - 驗收：實際注入包先完成裁切與来源組裝再驗證（`_pack_evidence` → `_verify`，`selftest_evidence_packet.py` 斷言驗證器所見即最終包）；四種狀態 complete/partial/conflicting/not_found；內容雜湊 `packet_hash`；冷層回退（`selftest_ulm.py`）；明確時間錨點（`reference_time`，缺失時為接收牆鐘而非最新記憶時間）；刪除後不返回舊候選（`selftest_lifecycle_safety.py`）；圖擴展與場景路按意圖/缺口啟動並受整體預算約束（`budget.scope`）。
  - 本次修正：Search 對最終證據包記錄一次實際使用（`record_recall` 最終 AMU、不增場景訪問、不推 revision），候選命中與多輪重試不增熱（§4.4/§4.5）。
- [x] 1.3 派生画像、生命週期與經驗治理
  - _Depends:_ 1.1、1.2。
  - _Boundary:_ `app/profile.py`、`app/scenes.py`、`app/experience.py`、相關 Store/Feedback schema、提示詞、對應測試。
  - _Design:_ §2.2–2.4、§4、§6、§8。
  - 驗收：敏感性繼承（`register_dependencies`/`link_sources` 取最嚴）；同源支持去重（事件級去重，跨會話轉述不增獨立支持）；推斷標記（`epistemic_status`）；熱度與可信度分離（Heat 只調度）；過期計畫可歷史查詢（`selftest_ulm.py` 前瞻過濾）；反馈幂等與有歸因計數（`feedback_events` 賬本，`selftest_experience.py`）；技能驗證綁定環境/代碼/依賴版本（`experience.py` verified 派生）。
- [x] 1.4 整合、生命週期回放與完整驗收
  - _Depends:_ 1.1、1.2、1.3。
  - _Boundary:_ API/評測整合、離線測試執行器、既有測試契約更新、生命週期回放、README/STORAGE/環境設定說明。
  - _Design:_ §9、§11。
  - 驗收：全部 14 個離線測試套件通過（`scripts/run_selftests.py`，含 FastAPI TestClient HTTP 契約煙霧測試 16 項）；`python -m compileall app scripts` 與 `git diff --check` 通過；文檔已同步（STORAGE.md 錨點語義、.env.example、IMPROVEMENT_PLAN.md）；可選參數/激活後端、RL、MATTS、多智能體擴展保持預設關閉，未宣稱已實作。

## 驗證方式

- 使用 `memory_system/.venv/Scripts/python.exe`，Fake 模式，隔離 `.env` 和臨時資料庫；不呼叫付費模型、不修改現有 `memory.db`。
- 測試：既有 `scripts/selftest_*.py` 加新增行為回歸，14/14 套件通過。
- 建置檢查：Python `compileall`、`git diff --check`。
- Smoke：既有 FastAPI TestClient 契約測試，另驗證新狀態/版本欄位與 purge。
- 可選參數/激活後端、RL、MATTS 和多智能體擴展依設計預設關閉；不能宣稱已實作或已取得真實模型收益。

## 實作備註

- 開始時已有使用者尚未提交的設計文檔修改；不覆寫、不納入程式碼提交。
- 舊測試已知未通過項已按新契約修正（修正測試而非恢復已捨棄行為）：
  - `selftest_evidence`：`_segments` mock 改為索引段格式（episode 分段假設）。
  - `selftest_profile`：無時間戳消息保持無時間，`latest_time` 為 None（取代合成時間延續的舊預期）。
  - `selftest_ulm`：同一觀察事件的跨會話轉述不增獨立支持（support 保持單一、特質維持 transient）；場景路測試改用 multi_hop 意圖（§5.2 按需啟用）。
  - `selftest_search`：來源正文隨證據返回，修正夾具與斷言不一致。
  - `selftest_evidence_packet`：最終包內記憶記一次實際使用（強度提升），未返回候選不增熱。
