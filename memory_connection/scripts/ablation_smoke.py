"""Reproducible orthogonal-vector experiment; NOT a semantic benchmark."""
import asyncio
import json
from memory_connection.app.config import Settings
from memory_connection.app.schemas import AddRequest, SearchRequest
from memory_connection.app.service import MemoryService
from memory_connection.app.store import Store
from memory_connection.tests.test_pipeline import FixtureProvider


async def main():
    rows = []
    for memory, query in ((False, False), (True, False), (False, True), (True, True)):
        provider = FixtureProvider()
        store = Store(":memory:", provider.space_id)
        try:
            service = MemoryService(store, provider, Settings(memory_expansion=memory, query_expansion=query))
            await service.add(AddRequest(request_id="r", user_id="u", session_id="s",
                                         messages=[{"role": "user", "content": "甲源"}]))
            result = await service.search(SearchRequest(user_id="u", query="乙需", debug=True))
            rows.append({"mode": f"C{int(memory)}{int(query)}", "candidates": len(result["trace"]["candidates"]),
                         "returned": len(result["data"]), "verification_status": result["verification_status"],
                         "bridges": [p for c in result["trace"]["candidates"] for p in c["paths"]],
                         "verifier_received_expansions": "共同橋接" in json.dumps(provider.seen, ensure_ascii=False)})
        finally:
            store.close()
    print(json.dumps({"kind": "deterministic_fixture_not_quality_benchmark", "results": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
