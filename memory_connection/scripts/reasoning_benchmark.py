"""Paired before/after four-arm evaluation on frozen, previously unused histories."""
import argparse
import asyncio
import copy
import importlib
import importlib.util
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from .benchmark_metrics import prepare, file_hash, common_packet, quote_metrics, call_cost
from .real_benchmark import load_env, configuration, CONTEXT, LegacyUsage, append, save
from memory_connection.app.providers import HTTPProvider
from memory_connection.app.service import MemoryService
from memory_connection.app.store import Store
from memory_connection.app.schemas import AddRequest, SearchRequest, View
from memory_connection.app.expansion import clean_expansion


ARMS = ("C00", "C10", "C01", "C11")


def settings_from_file(path):
    load_env(path)
    os.environ["AML_ANSWER_CONTEXT_MAX_CHARS"] = "24000"
    os.environ["AML_ANSWER_CONTEXT_ITEM_MAX_CHARS"] = "24000"
    settings, legacy = configuration()
    return settings.model_copy(update={
        "rerank_model": os.environ.get("MC_RERANK_MODEL") or legacy.CE_MODEL,
        "rerank_api_url": os.environ.get("MC_RERANK_API_URL") or legacy.CE_API_URL,
        "rerank_api_key": os.environ.get("MC_RERANK_API_KEY") or legacy.CE_API_KEY,
        "rerank_api_format": os.environ.get("MC_RERANK_API_FORMAT") or legacy.CE_API_FORMAT,
        "search_seconds": 120, "expansion_seconds": 30, "recall_seconds": 30,
        "rerank_seconds": 20, "verify_seconds": 80, "add_seconds": 180, "evidence_bytes": 24000})


def safe_settings(settings):
    data = settings.model_dump(exclude={"api_key", "provider_api_key", "embed_api_key", "rerank_api_key",
                                        "api_base", "embed_api_base", "rerank_api_url"})
    data.update(chat_host=urlsplit(settings.api_base).hostname,
                embed_host=urlsplit(settings.embed_api_base or settings.api_base).hostname,
                rerank_host=urlsplit(settings.rerank_api_url).hostname)
    return data


def hashes():
    result = {str(p): file_hash(p) for root in ("memory_connection/app", "memory_connection/prompts",
                                            "memory_system/app", "prompts")
            for p in Path(root).rglob("*") if p.suffix in {".py", ".txt"}}
    for name in ("optimization_benchmark.py", "reasoning_benchmark.py", "benchmark_metrics.py", "real_benchmark.py"):
        path=Path("memory_connection/scripts")/name
        result[str(path)]=file_hash(path)
    return result


def before_modules(root):
    snapshot = root/"before"
    expected = json.loads((root/"before-hashes.json").read_text())
    if any(file_hash(snapshot/p) != sha for p,sha in expected.items()):
        raise ValueError("frozen before implementation changed")
    spec = importlib.util.spec_from_file_location("mc_before", snapshot/"app/__init__.py",
                                                submodule_search_locations=[str(snapshot/"app")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["mc_before"] = module
    spec.loader.exec_module(module)
    return {name: importlib.import_module("mc_before."+name) for name in ("config", "service", "providers")}


class PreparedQueryProvider:
    """Same generated query views and vectors in every matched comparison."""
    def __init__(self, delegate, prepared):
        self.delegate, self.prepared = delegate, prepared

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    async def expand_query(self, query):
        if query != self.prepared["query"]:
            raise ValueError("frozen query mismatch")
        return copy.deepcopy(self.prepared["expansion"])

    async def embed(self, texts):
        return [self.prepared["vectors"][text] for text in texts]


async def preparation_call(provider, operation):
    # Shared, bounded preprocessing retries. Every failed attempt stays in usage;
    # each component's measured time includes its retries for all paired arms.
    for attempt in range(3):
        try:
            return await asyncio.wait_for(operation(),30)
        except (TimeoutError,):
            await provider.aclose()
            if attempt==2:raise
            await asyncio.sleep(1)


async def prepare_query(root, conv, qa, provider):
    folder = root/"query-preparations"
    folder.mkdir(exist_ok=True)
    path = folder/f"{conv['conversation_id']}-{qa['qa_id']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    stamps = [m["timestamp"] for session in conv["sessions"] for m in session if m.get("timestamp") is not None]
    query = {"query": qa["question"], "options": [], "reference_time": datetime.fromtimestamp(max(stamps)/1000,timezone.utc).isoformat(), "as_of": None}
    context = {"arm":"query_shared", "phase":"query_prepare", "conversation_id":conv["conversation_id"], "qa_id":qa["qa_id"]}
    CONTEXT.set({**context, "component":"original_embed"})
    started = time.perf_counter()
    original = (await preparation_call(provider,lambda: provider.embed([qa["question"]])))[0]
    original_seconds = time.perf_counter()-started
    CONTEXT.set({**context, "component":"query_expansion"})
    started = time.perf_counter()
    expansion = await preparation_call(provider,lambda: provider.expand_query(query))
    views = clean_expansion(expansion,provider.settings.query_views,original=qa["question"])
    expansion_seconds = time.perf_counter()-started
    CONTEXT.set({**context, "component":"derived_embed"})
    started = time.perf_counter()
    derived = await preparation_call(provider,lambda: provider.embed([v.text for v in views])) if views else []
    embed_seconds = time.perf_counter()-started
    result = {"query":query,"expansion":expansion,"vectors":{qa["question"]:original, **{v.text:vec for v,vec in zip(views,derived)}},
              "original_seconds":original_seconds,"expanded_seconds":original_seconds+expansion_seconds+embed_seconds}
    save(path,result)
    return result


async def run(args, settings, plan, manifest):
    root = Path(args.output)
    calls = [json.loads(l) for l in (root/"calls.jsonl").read_text().splitlines()] if (root/"calls.jsonl").exists() else []
    def sink(record):
        row = {**CONTEXT.get(),**record};calls.append(row);append(root/"calls.jsonl",row)
    prices=json.loads(Path("memory_connection/benchmark-prices-usd.json").read_text())
    handler=LegacyUsage(sink);logger=logging.getLogger("aml.metrics");logger.setLevel(logging.INFO);logger.addHandler(handler)
    provider=HTTPProvider(settings,sink)
    stores={}
    try:
        for conv in plan:
            for family in ("C0","C1"):
                stores[(conv["conversation_id"],family)]=Store(str(root/f"{conv['conversation_id']}-{family}.db"),provider.space_id)
        if args.phase in {"all","ingest"}:
            started=time.perf_counter();gate=asyncio.Semaphore(8);done=0
            payloads=[(conv,family,si,start,session[start:start+6]) for conv in plan for family in ("C0","C1")
                      for si,session in enumerate(conv["sessions"]) for start in range(0,len(session),6)]
            save(root/"status.json",{"status":"ingesting","total_batches":len(payloads),"completed_batches":0})
            async def ingest(payload):
                nonlocal done
                conv,family,si,start,messages=payload;cid=conv["conversation_id"]
                async with gate:
                    CONTEXT.set({"arm":family,"phase":"add","conversation_id":cid})
                    service=MemoryService(stores[(cid,family)],provider,settings.model_copy(update={"memory_expansion":family=="C1"}))
                    response=await service.add(AddRequest(request_id=f"{cid}-{si}-{start}",user_id=cid,session_id=str(si),messages=messages))
                    if response["degradation"]:append(root/"ingestion-degradation.jsonl",{"conversation_id":cid,"family":family,"request_id":f"{cid}-{si}-{start}","degradation":response["degradation"]})
                    done+=1
                    save(root/"status.json",{"status":"ingesting","total_batches":len(payloads),"completed_batches":done})
                    if done%20==0:print(f"ingested {done}/{len(payloads)}",flush=True)
            outcomes=await asyncio.gather(*(ingest(p) for p in payloads),return_exceptions=True)
            error=next((x for x in outcomes if isinstance(x,BaseException)),None)
            if error:raise error
            append(root/"ingestion-runs.jsonl",{"seconds":time.perf_counter()-started,"batches":len(payloads)})
            save(root/"status.json",{"status":"ingestion_complete"})
        if args.phase not in {"all","evaluate"}:return
        # All histories must be complete before any held-out score is inspected.
        for conv in plan:
            for family in ("C0","C1"):
                count=len(stores[(conv["conversation_id"],family)].snapshot(SearchRequest(user_id=conv["conversation_id"],query="integrity"))[2])
                if count!=sum(map(len,conv["sessions"])):raise ValueError("incomplete history")
        from memory_system.app import eval_scoring
        old=before_modules(root)
        before_settings=old["config"].Settings(**{k:v for k,v in settings.model_dump().items() if k in old["config"].Settings.model_fields})
        before_provider=old["providers"].HTTPProvider(before_settings,sink)
        rows=[json.loads(l) for l in (root/"results.jsonl").read_text(encoding="utf-8").splitlines()] if (root/"results.jsonl").exists() else []
        completed={(r["arm"],r["conversation_id"],r["qa_id"]) for r in rows}
        try:
            for conv in plan:
                cid=conv["conversation_id"]
                for qa in conv["qa"]:
                    prepared=await prepare_query(root,conv,qa,provider)
                    tasks=[(version,arm) for version in ("before","after") for arm in ARMS]
                    random.Random(f"20261025:{cid}:{qa['qa_id']}").shuffle(tasks)
                    async def evaluate(version,arm):
                        label=version+"/"+arm
                        if (label,cid,qa["qa_id"]) in completed:return
                        CONTEXT.set({"arm":label,"phase":"search","conversation_id":cid,"qa_id":qa["qa_id"]})
                        cls=MemoryService if version=="after" else old["service"].MemoryService
                        cfg=settings if version=="after" else before_settings
                        delegate=provider if version=="after" else before_provider
                        service=cls(stores[(cid,arm[:2])],PreparedQueryProvider(delegate,prepared),cfg.model_copy(update={"memory_expansion":arm[1]=="1","query_expansion":arm[2]=="1"}))
                        started=time.perf_counter();errors=[];response={}
                        try:response=await service.search(SearchRequest(user_id=cid,query=qa["question"],top_k=10,debug=True))
                        except Exception as exc:errors.append({"phase":"search","type":type(exc).__name__})
                        observed=time.perf_counter()-started
                        common_seconds=prepared["expanded_seconds" if arm[2]=="1" else "original_seconds"]
                        packet=common_packet(response.get("data",[]),24000)
                        CONTEXT.set({"arm":label,"phase":"answer_judge","conversation_id":cid,"qa_id":qa["qa_id"]})
                        started=time.perf_counter()
                        prediction,score,diagnostics=await eval_scoring.evaluate(qa,[{"id":str(i),"content":json.dumps(p,ensure_ascii=False)} for i,p in enumerate(packet)])
                        if diagnostics.get("error_stage"):errors.append({"phase":diagnostics["error_stage"],"type":diagnostics.get("error_type")})
                        row={"arm":label,"version":version,"expansion_arm":arm,"conversation_id":cid,"qa_id":qa["qa_id"],
                             "question":qa["question"],"gold":qa["answer"],"category":qa["category"],"prediction":prediction,
                             "score":0.0 if errors else score,"raw_judge_score":score,"errors":errors,
                             "search_seconds":observed+common_seconds,"search_seconds_observed":observed,"common_query_seconds":common_seconds,
                             "answer_judge_seconds":time.perf_counter()-started,"retrieval":quote_metrics(packet,qa,conv["source_texts"]),
                             "degradation":response.get("degradation",[]),"verification_errors":response.get("verification_errors",[])}
                        rows.append(row);append(root/"results.jsonl",row)
                        append(root/"evidence.jsonl",{"arm":label,"conversation_id":cid,"qa_id":qa["qa_id"],"packet":packet,"response":response})
                        save(root/"status.json",{"status":"evaluating","completed_results":len(rows),"expected_results":manifest["expected_results"]})
                        print(f"evaluated {cid}/{qa['qa_id']} {label} score={score} observed={observed:.2f}s",flush=True)
                    await asyncio.gather(*(evaluate(v,a) for v,a in tasks))
        finally:await before_provider.aclose()
        save(root/"status.json",{"status":"complete","completed_results":len(rows),"expected_results":manifest["expected_results"]})
    except BaseException as exc:
        save(root/"status.json",{"status":"interrupted","error_type":type(exc).__name__})
        raise
    finally:
        for store in stores.values():store.close()
        await provider.aclose();logger.removeHandler(handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="memory_connection/.env")
    parser.add_argument("--output",default="memory_connection/runs/optimization-v3-holdout-20261024")
    parser.add_argument("--phase",choices=["plan","all","ingest","evaluate"],default="plan")
    args=parser.parse_args();root=Path(args.output);root.mkdir(exist_ok=True)
    settings=settings_from_file(args.config)
    if not all((settings.provider_api_key,settings.embed_api_key,settings.rerank_api_key,settings.rerank_api_url)):
        raise ValueError("real chat, embedding and reranker configuration required")
    prior_runs=("real-pilot-v2-20261024", "optimization-holdout-20261024")
    exclude=sorted({c["conversation_id"] for name in prior_runs
                    for c in json.loads((Path("memory_connection/runs")/name/"manifest.json").read_text())["selected"]})
    plan=prepare("memory_system/data/locomo_eval.json","memory_system/data/locomo10.json",3,12,20261025,exclude=exclude)
    manifest={"protocol":"paired-heldout-four-arm-v3","seed":20261025,"excluded_conversations":exclude,
              "dataset_hash":file_hash("memory_system/data/locomo_eval.json"),"raw_hash":file_hash("memory_system/data/locomo10.json"),
              "selected":[{"conversation_id":c["conversation_id"],"messages":sum(map(len,c["sessions"])),"qa_ids":[q["qa_id"] for q in c["qa"]]} for c in plan],
              "expected_results":8*sum(len(c["qa"]) for c in plan),"settings":safe_settings(settings),"source_hashes":hashes(),
              "preparation_attempt_limit":3,"add_concurrency":8,"query_concurrency":8,"top_k":10,"evidence_bytes":24000,
              "query_control":"Identical pre-generated query views and vectors across before/after and expansion arms; actual prep recorded once; latency adds common measured prep to each observed search."}
    before_modules(root)
    if (root/"manifest.json").exists():
        previous=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
        if previous!=manifest:raise ValueError("frozen experiment configuration or source changed")
    else:
        save(root/"manifest.json",manifest);save(root/"plan.json",plan)
        save(root/"prices.snapshot.json",json.loads(Path("memory_connection/benchmark-prices-usd.json").read_text()))
    print(json.dumps({"selected":manifest["selected"],"expected_results":manifest["expected_results"]},ensure_ascii=False),flush=True)
    if args.phase!="plan":asyncio.run(run(args,settings,plan,manifest))


if __name__=="__main__":main()
