"""Offline paired analysis; never calls a model or tunes the frozen experiment."""
import argparse
import collections
import json
import random
import sqlite3
import statistics
import importlib.util
from pathlib import Path
from .benchmark_metrics import call_cost, percentile, file_hash
from memory_connection.app.event_builder import valid_spans
from memory_connection.app.schemas import Span
from memory_connection.app.temporal import source_annotations


def rows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def paired(left, right, by_key):
    pairs=[(r,by_key[(right,cid,qid)]) for (arm,cid,qid),r in by_key.items()
           if arm==left and (right,cid,qid) in by_key]
    if not pairs:return {"pairs":0}
    groups=collections.defaultdict(list)
    for a,b in pairs:groups[a["conversation_id"]].append(a["score"]-b["score"])
    rng=random.Random(20261024)
    estimates=[statistics.mean(d for cid in rng.choices(list(groups),k=len(groups)) for d in groups[cid]) for _ in range(4000)] if len(groups)>1 else []
    return {"pairs":len(pairs),"accuracy_delta":statistics.mean(a["score"]-b["score"] for a,b in pairs),
            "wins":sum(a["score"]>b["score"] for a,b in pairs),"losses":sum(a["score"]<b["score"] for a,b in pairs),
            "ties":sum(a["score"]==b["score"] for a,b in pairs),
            "conversation_deltas":{cid:statistics.mean(ds) for cid,ds in groups.items()},
            "cluster_bootstrap_95":[percentile(estimates,.025),percentile(estimates,.975)] if estimates else None,
            "exploratory_only":len(groups)<10}


def analyze(root):
    annotations=source_annotations
    archived=root/'frozen-sources/memory_connection/app/temporal.py'
    if archived.exists():
        spec=importlib.util.spec_from_file_location('archived_temporal',archived)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        annotations=module.source_annotations
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    plan=json.loads((root/"plan.json").read_text(encoding="utf-8"))
    prices=json.loads((root/"prices.snapshot.json").read_text())
    results,calls,evidence=rows(root/"results.jsonl"),rows(root/"calls.jsonl"),rows(root/"evidence.jsonl")
    by_key={(r["arm"],r["conversation_id"],r["qa_id"]):r for r in results}
    ekeys={(r["arm"],r["conversation_id"],r["qa_id"]) for r in evidence}
    expected={(v+"/"+a,c["conversation_id"],q["qa_id"]) for c in plan for q in c["qa"]
              for v in ("before","after") for a in ("C00","C10","C01","C11")}
    report={"status":json.loads((root/"status.json").read_text()),
            "integrity":{"expected":len(expected),"results":len(results),"evidence":len(evidence),
                "duplicate_results":len(results)-len(by_key),"duplicate_evidence":len(evidence)-len(ekeys),
                "missing":sorted(expected-set(by_key)),"unexpected":sorted(set(by_key)-expected),
                "results_without_evidence":sorted(set(by_key)-ekeys),"evidence_without_results":sorted(ekeys-set(by_key)),
                "source_audit_target":"archived_snapshot" if (root/"frozen-sources").is_dir() else "working_sources",
                "changed_frozen_sources":[p for p,sha in manifest["source_hashes"].items()
                    if file_hash(root/"frozen-sources"/p if (root/"frozen-sources").is_dir() else p)!=sha],
                "changed_before_snapshot":[p for p,sha in json.loads((root/"before-hashes.json").read_text()).items()
                    if file_hash(root/"before"/p)!=sha],
                "dataset_changed":file_hash("memory_system/data/locomo_eval.json")!=manifest["dataset_hash"],
                "raw_dataset_changed":file_hash("memory_system/data/locomo10.json")!=manifest["raw_hash"],
                "heldout_overlap":sorted(set(manifest["excluded_conversations"])&{c["conversation_id"] for c in plan})},
            "actual_calls":len(calls),"actual_cost":call_cost(calls,prices),
            "unreported_or_failed_calls":sum(not c.get("usage_reported") or c.get("status")!="ok" for c in calls),
            "failed_calls_by_phase":dict(collections.Counter(c.get("phase") for c in calls if c.get("status")!="ok")),
            "resumptions":rows(root/"resumptions.jsonl"),
            "index_families":{},"arms":{},"paired_versions":{},"dual_vs_single":{}}
    for family in ("C0","C1"):
        selected=[c for c in calls if c["phase"]=="add" and c["arm"]==family]
        report["index_families"][family]={"calls":len(selected),"cost":call_cost(selected,prices)}
    lookup={};index_counts={};expansion=collections.Counter()
    for path in root.glob("*-C[01].db"):
        conn=sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro",uri=True)
        try:events=[json.loads(p) for (p,) in conn.execute("SELECT payload FROM events")]
        finally:conn.close()
        lookup.update({e["event_id"]:e for e in events});index_counts[path.stem]=len(events)
        if path.stem.endswith("C1"):
            expansion["events"]+=len(events)
            expansion["derived_views"]+=sum(len(e["views"])-1 for e in events)
            expansion["zero_derived"]+=sum(len(e["views"])==1 for e in events)
            expansion["failed"]+=sum(e["expansion_status"]=="failed" for e in events)
    report["index_event_counts"],report["expansion_audit"]=index_counts,dict(expansion)
    gold={(c["conversation_id"],q["qa_id"]):set(q.get("evidence",[])) for c in plan for q in c["qa"]}
    source_texts={c["conversation_id"]:c["source_texts"] for c in plan}
    raw={r["sample_id"]:[q for q in r["qa"] if q.get("answer") is not None] for r in json.loads(Path("memory_system/data/locomo10.json").read_text(encoding="utf-8"))}
    categories={1:"multi-hop",2:"temporal",3:"open-domain",4:"single-hop",5:"adversarial"}
    def source_ids(events):return {s["message_id"] for e in events for s in e["sources"]}
    diagnostics={};event_sets={};citation_errors=[];annotation_errors=[];returned=0
    for item in evidence:
        key=(item["arm"],item["conversation_id"],item["qa_id"])
        if key not in by_key:continue
        cid,qid=key[1:];g=gold[(cid,qid)];response=item["response"];trace=response.get("trace",{})
        candidates=trace.get("candidates",[]);decisions=trace.get("decisions",[])
        eligible=bool(g) and g.issubset(source_texts[cid])
        candidate_events=[lookup[c["event_id"]] for c in candidates]
        pool=candidate_events[:16]
        reserved=[lookup[c["event_id"]] for c in candidates[:16] if c.get("bridge_reserved")]
        accepted=[lookup[d["event_id"]] for d in decisions if d["verdict"] in {"relevant","partial","contradicting"}]
        found={s["message_id"] for p in item["packet"] for s in p.get("sources",[])}
        def cited_ids(event,spans):
            selected={s["source_id"] for s in spans}
            return {s["message_id"] for s in event["sources"] if s["source_id"] in selected}
        accepted_cited=set().union(*(cited_ids(lookup[d["event_id"]],d.get("evidence_spans",[]))
            for d in decisions if d["verdict"] in {"relevant","partial","contradicting"}))
        visible_hits=[h for h in response.get("data",[]) if any(
            h["content"]==p["content"] and h["sources"]==p["sources"] for p in item["packet"])]
        final_cited=set().union(*(cited_ids(lookup[h["id"]],h.get("evidence_spans",[])) for h in visible_hits))
        def recall(ids):return len(g&ids)/len(g) if eligible else None
        original=raw[cid][int(qid)]
        if original["question"]!=by_key[key]["question"]:raise ValueError("QA category alignment failed")
        diagnostic={"arm":key[0],"conversation_id":cid,"qa_id":qid,"raw_category":original["category"],
            "category":categories[original["category"]],"gold_ids":sorted(g),"final_gold_ids":sorted(g&found),
            "candidate_recall":recall(source_ids(candidate_events)),"pool_recall":recall(source_ids(pool)),
            "accepted_recall":recall(source_ids(accepted)),"final_source_recall":recall(found),
            "accepted_cited_source_recall":recall(accepted_cited),"final_cited_source_recall":recall(final_cited),
            "reserved_event_count":len(reserved),"reserved_gold_sources":sorted(g&source_ids(reserved)),
            "candidate_count":len(candidates),"query_views":trace.get("query_views",[]),
            "returned":len(item["packet"]),"errors":response.get("verification_errors",[]),
            "ranking_mode":response.get("metrics",{}).get("ranking_mode"),"metrics":response.get("metrics",{}),
            "score":by_key[key]["score"]}
        diagnostics[key]=diagnostic
        event_sets[key]={"candidates":{e["event_id"] for e in candidate_events},
                         "candidate_sources":source_ids(candidate_events),"final_sources":found,
                         "coarse_order":[c["event_id"] for c in sorted(candidates,key=lambda c:c.get("coarse_rank") or candidates.index(c)+1)]}
        for hit in response.get("data",[]):
            returned+=1;event=lookup[hit["id"]]
            if hit["content"]!=event["text"] or hit["sources"]!=event["sources"] or not valid_spans([Span(**s) for s in hit["evidence_spans"]],event):
                citation_errors.append([*key,hit["id"]])
            if "source_annotations" in hit and hit["source_annotations"]!=annotations(event["sources"]):
                annotation_errors.append([*key,hit["id"]])
    report["integrity"]["returned_original_items"]=returned
    report["integrity"]["invalid_original_or_citation"]=citation_errors
    report["integrity"]["invalid_source_annotations"]=annotation_errors
    report["integrity"]["different_query_views"]=[]
    report["integrity"]["different_coarse_candidates"]=[]
    for (arm,cid,qid),detail in diagnostics.items():
        if not arm.startswith("after/"):continue
        other=(arm.replace("after/","before/"),cid,qid)
        if other not in diagnostics:continue
        if detail["query_views"]!=diagnostics[other]["query_views"]:
            report["integrity"]["different_query_views"].append([arm,cid,qid])
        if event_sets[(arm,cid,qid)]["coarse_order"]!=event_sets[other]["coarse_order"]:
            report["integrity"]["different_coarse_candidates"].append([arm,cid,qid])
    for arm in sorted({r["arm"] for r in results}):
        selected=[r for r in results if r["arm"]==arm];ds=[d for k,d in diagnostics.items() if k[0]==arm]
        usage=[c for c in calls if c.get("arm")==arm and c["phase"]=="search"]
        evaluated={(r["conversation_id"],r["qa_id"]) for r in selected}
        common=[c for c in calls if c["phase"]=="query_prepare" and (c["conversation_id"],c["qa_id"]) in evaluated
                and (arm[-1]=="1" or c.get("component")=="original_embed")]
        family=arm.split("/")[1][:2]
        index=[c for c in calls if c["phase"]=="add" and c["arm"]==family]
        logical=call_cost(usage+common+index,prices)
        report["arms"][arm]={"questions":len(selected),"accuracy":statistics.mean(r["score"] for r in selected),
            "pipeline":{metric:statistics.mean(d[metric] for d in ds if d[metric] is not None) for metric in
                        ("candidate_recall","pool_recall","accepted_recall","final_source_recall",
                         "accepted_cited_source_recall","final_cited_source_recall","candidate_count","returned") if any(d[metric] is not None for d in ds)},
            "full_quote_recall":statistics.mean(r["retrieval"]["gold_quote_recall"] for r in selected if r["retrieval"]["eligible"]),
            "search_p50_s":percentile([r["search_seconds"] for r in selected],.5),"search_p95_s":percentile([r["search_seconds"] for r in selected],.95),
            "observed_stages_p50_s":percentile([r["search_seconds_observed"] for r in selected],.5),
            "search_cost_including_shared_preparation":call_cost(usage+common,prices),"add_plus_search_cost":logical,
            "logical_cost_per_query":logical["priced_reported_usage_cost"]/len(selected) if logical["priced_reported_usage_cost"] is not None else None,
            "answer_judge_cost":call_cost([c for c in calls if c["arm"]==arm and c["phase"]=="answer_judge"],prices),
            "provider_error_calls":sum(c.get("status")!="ok" for c in usage),
            "top_level_error_questions":sum(bool(r["errors"]) for r in selected),
            "degraded_questions":sum(bool(r["degradation"]) for r in selected),
            "verification_error_questions":sum(bool(r["verification_errors"]) for r in selected),
            "verification_error_reasons":dict(collections.Counter(e["reason"] for r in selected for e in r["verification_errors"])),
            "ranking_modes":dict(collections.Counter(d["ranking_mode"] for d in ds)),
            "reserved_questions":sum(d["reserved_event_count"]>0 for d in ds),
            "reserved_events":sum(d["reserved_event_count"] for d in ds),
            "reserved_gold_questions":sum(bool(d["reserved_gold_sources"]) for d in ds),
            "categories":{str(cat):{"name":categories[cat],"questions":len(group),"accuracy":statistics.mean(d["score"] for d in group)}
                          for cat in sorted({d["raw_category"] for d in ds}) if (group:=[d for d in ds if d["raw_category"]==cat])}}
    for arm in ("C00","C10","C01","C11"):
        report["paired_versions"][arm]=paired("after/"+arm,"before/"+arm,by_key)
    for version in ("before","after"):
        for arm in ("C00","C10","C01"):
            report["dual_vs_single"][version+"/C11-"+arm]=paired(version+"/C11",version+"/"+arm,by_key)
    report["unique_dual_gold"]={}
    for version in ("before","after"):
        novel=[]
        for (label,cid,qid),sets in event_sets.items():
            if label!=version+"/C11":continue
            single=[event_sets.get((version+"/"+arm,cid,qid)) for arm in ("C10","C01")]
            if any(s is None for s in single):continue
            g=gold[(cid,qid)]
            novel.append({"conversation_id":cid,"qa_id":qid,
                "candidate":sorted(g&(sets["candidate_sources"]-single[0]["candidate_sources"]-single[1]["candidate_sources"])),
                "final":sorted(g&(sets["final_sources"]-single[0]["final_sources"]-single[1]["final_sources"]))})
        report["unique_dual_gold"][version]=novel
    report["question_pipeline"]=list(diagnostics.values())
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run");args=parser.parse_args();root=Path(args.run)
    report=analyze(root);(root/"analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"question_pipeline","unique_dual_gold"}},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
