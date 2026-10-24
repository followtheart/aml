import asyncio
import time
import uuid
from datetime import datetime, timezone
from .config import Settings
from .event_builder import build_events, original_packet
from .expansion import original_view, clean_expansion
from .providers import normalize_vectors
from .schemas import AddRequest, SearchRequest, View
from .recall import recall
from .verification import verify_candidates, byte_size
from .packing import pack
from .ranking import rank_candidates, single_side_sources, reserve_exclusive, cover_query_anchors


class MemoryService:
    def __init__(self, store, provider, settings=None):
        self.store = store
        self.provider = provider
        self.settings = settings or Settings()

    async def answer(self, req: SearchRequest):
        from .answering import answer
        result=await self.search(req)
        if result.get('verification_status')=='invalidated':return result
        started=time.monotonic()
        try:
            result['answer']=await asyncio.wait_for(answer(self.provider,req.query,result.get('data',[]),
                req.reference_time,self.settings.evidence_bytes),self.settings.answer_seconds)
        except Exception as exc:
            result['answer']={'answer':'Unknown','reference_status':'failed','error_type':type(exc).__name__}
            result.setdefault('degradation',[]).append({'stage':'answer','error_type':type(exc).__name__})
        result.setdefault('metrics',{})['answer_ms']=round((time.monotonic()-started)*1000,3)
        if self.store.state(req.user_id)!=(result['read_revision'],result['read_epoch']):
            return {'data':[],'answer':{'answer':'Unknown','reference_status':'invalidated'},
                    'verification_status':'invalidated','degradation':[{'stage':'snapshot','reason':'concurrent_write_or_purge'}]}
        return result

    async def add(self, req: AddRequest):
        replay = self.store.replay(req)
        if replay:
            return replay
        epoch = self.store.state(req.user_id)[1]
        deadline = time.monotonic() + self.settings.add_seconds
        events, degraded = build_events(req), []
        if byte_size(req.model_dump()) > 512000:
            raise ValueError("Add exceeds 512000-byte source budget")
        for event in events:
            views = [original_view(event)]
            event["expansion_status"] = "disabled"
            if self.settings.memory_expansion and self.settings.memory_views:
                try:
                    if byte_size(original_packet(event)) > self.settings.batch_bytes:
                        raise ValueError("memory expansion context exceeds byte budget")
                    raw = await asyncio.wait_for(self.provider.expand_memory(original_packet(event)),
                                                 timeout=max(0, deadline-time.monotonic()))
                    views += clean_expansion(raw, self.settings.memory_views, event, event["text"])
                    event["expansion_status"] = "ok"
                except Exception as exc:
                    event["expansion_status"] = "failed"
                    degraded.append({"event_id": event["event_id"], "stage": "memory_expansion", "error_type": type(exc).__name__})
            event["views"] = [{**view.model_dump(), "generator_version": "expand_memory_v1"} for view in views]
        # Batch across events, retaining atomic publication and exact view order.
        pending_views = [view for event in events for view in event["views"]]
        vectors = await asyncio.wait_for(self.provider.embed([v["text"] for v in pending_views]),
                                         timeout=max(0, deadline-time.monotonic()))
        vectors = normalize_vectors(vectors, len(pending_views), self.provider.dimension)
        for view, vector in zip(pending_views, vectors):
            view["vector"] = vector
        return self.store.commit(req, events, epoch, degraded)

    async def search(self, req: SearchRequest):
        started = time.monotonic()
        deadline = started + self.settings.search_seconds
        revision, epoch, events = self.store.snapshot(req)
        timestamps = [s["timestamp"] for e in events for s in e["sources"] if s.get("timestamp") is not None]
        reference = (datetime.fromtimestamp(max(timestamps)/1000, timezone.utc).isoformat()
                     if timestamps else datetime.now(timezone.utc).isoformat())
        query = {"query": req.query, "options": req.options,
                 "reference_time": req.reference_time or reference, "as_of": req.as_of}
        qviews = [View(kind="original", text=req.query)]
        degraded, metrics = [], {"query_expansion_calls": 0, "verification_calls": 0}
        expansion_started = time.monotonic()
        if self.settings.query_expansion and self.settings.query_views:
            try:
                metrics["query_expansion_calls"] = 1
                raw = await asyncio.wait_for(self.provider.expand_query(query),
                    timeout=max(0, min(self.settings.expansion_seconds, deadline-time.monotonic())))
                qviews += clean_expansion(raw, self.settings.query_views, original=req.query)
            except Exception as exc:
                degraded.append({"stage": "query_expansion", "error_type": type(exc).__name__})
        metrics["query_expansion_ms"] = round((time.monotonic()-expansion_started)*1000, 3)
        vectors = None
        recall_started = time.monotonic()
        recall_deadline = min(deadline, time.monotonic() + self.settings.recall_seconds)
        try:
            vectors = await asyncio.wait_for(self.provider.embed([v.text for v in qviews]),
                                             timeout=max(0, recall_deadline-time.monotonic()))
            vectors = normalize_vectors(vectors, len(qviews), self.provider.dimension)
        except Exception as exc:
            degraded.append({"stage": "query_embedding", "error_type": type(exc).__name__})
        try:
            candidates, recall_metrics = await asyncio.wait_for(
                asyncio.to_thread(recall, events, qviews, vectors, self.settings, recall_deadline),
                timeout=max(0, recall_deadline-time.monotonic()))
            metrics.update(recall_metrics)
        except TimeoutError:
            candidates = []
            degraded.append({"stage": "recall", "error_type": "TimeoutError"})
        single_sources = None
        if (self.settings.exclusive_bridge_slots and self.settings.memory_expansion and
                self.settings.query_expansion and len(qviews)>1 and vectors is not None):
            try:
                single_sources, compared = await single_side_sources(events,qviews,vectors,self.settings,recall_deadline)
                metrics["counterfactual_view_comparisons"] = compared
            except Exception as exc:
                degraded.append({"stage":"counterfactual_recall","error_type":type(exc).__name__})
        metrics["recall_ms"] = round((time.monotonic()-recall_started)*1000, 3)
        ranking_started = time.monotonic()
        candidates, ranking_metrics, ranking_errors = await rank_candidates(
            self.provider, query, candidates, self.settings,
            min(deadline, time.monotonic()+self.settings.rerank_seconds))
        metrics.update(ranking_metrics)
        metrics["ranking_ms"] = round((time.monotonic()-ranking_started)*1000, 3)
        degraded.extend(ranking_errors)
        reserved=[]
        if single_sources is not None:
            candidates,reserved=reserve_exclusive(candidates,single_sources,self.settings.verify_limit,self.settings.exclusive_bridge_slots)
        elif self.settings.complementary_slots:
            candidates,reserved=cover_query_anchors(candidates,req.query,self.settings.verify_limit,self.settings.complementary_slots)
        metrics["bridge_reserved_events"]=reserved
        verification_started = time.monotonic()
        decisions, errors, calls = await verify_candidates(self.provider, query, candidates, self.settings,
            min(deadline, time.monotonic() + self.settings.verify_seconds))
        metrics["verification_calls"] = calls
        metrics["verification_ms"] = round((time.monotonic()-verification_started)*1000, 3)
        metrics["verify_truncated"] = max(0, len(candidates)-self.settings.verify_limit)
        data, omitted, conflicting = pack(candidates[:self.settings.verify_limit], decisions, req.top_k, self.settings.evidence_bytes)
        metrics["elapsed_ms"] = round((time.monotonic()-started)*1000, 3)
        incomplete = bool(degraded or errors or metrics["verify_truncated"] or omitted or
                          metrics.get("candidate_truncated", 0) or any(d["verdict"] == "partial" for d in decisions.values()))
        verification_status = "not_run" if not candidates else (
            "partial" if errors and decisions else "failed" if errors else
            "partial" if metrics["verify_truncated"] else "complete")
        result = {"search_id": uuid.uuid4().hex, "data": data, "read_revision": revision, "read_epoch": epoch,
                  "evidence_status": "incomplete" if incomplete else "conflicting" if conflicting else "retrieved" if data else "not_found",
                  "verification_status": verification_status, "conflicting": conflicting,
                  "degradation": degraded, "verification_errors": errors, "omitted": omitted, "metrics": metrics}
        if req.debug:
            result["trace"] = {"query_views": [v.model_dump() for v in qviews],
                               "candidates": [{"event_id": c["event"]["event_id"], "score": c["score"], "paths": c["paths"],
                                               "coarse_rank": c.get("coarse_rank"), "semantic_score": c.get("semantic_score"),
                                               "pre_reservation_rank": c.get("pre_reservation_rank"),
                                               "bridge_reserved": c.get("bridge_reserved",False),
                                               "exclusive_source_count": c.get("exclusive_source_count",0),
                                               "covered_query_anchors": c.get("covered_query_anchors",[]),
                                               "selection_gain": c.get("selection_gain")} for c in candidates],
                               "decisions": list(decisions.values()), "errors": errors}
        # Any intervening write or purge invalidates the snapshot, including debug text.
        if self.store.state(req.user_id) != (revision, epoch):
            return {"search_id": result["search_id"], "data": [], "read_revision": revision, "read_epoch": epoch,
                    "evidence_status": "incomplete", "verification_status": "invalidated",
                    "degradation": [{"stage": "snapshot", "reason": "concurrent_write_or_purge"}]}
        return result
