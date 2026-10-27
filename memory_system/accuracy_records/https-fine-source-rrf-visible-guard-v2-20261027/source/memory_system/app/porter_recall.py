"""Optional lexical recall with an unchanged original coarse pool.

Porter indices exist only in a request-local copy of the pinned read snapshot.
Original text, eligibility joins and the persistent database are unchanged.
"""
import copy
from contextlib import contextmanager
import json
import time

from . import budget, config

LIMIT = 20


@contextmanager
def lexical_view(st, user_id):
    from .store import Store
    st._check_user(user_id)
    work = Store(':memory:')
    try:
        # Preserve the full indexed corpus: filtering it to one user changes
        # BM25 document frequencies. All retrieval still uses production joins.
        with st._lock:
            st.conn.backup(work.conn, pages=128, progress=lambda *unused: budget.check())
        work.read_user_id = user_id

        def interrupt():
            try:
                budget.check()
                return 0
            except TimeoutError:
                return 1

        work.conn.set_progress_handler(interrupt, 1000)
        for name, columns in (
            ('amu_fts', 'amu_id UNINDEXED, user_id UNINDEXED, content, retrieval_key'),
            ('source_fts', 'request_id UNINDEXED, message_index UNINDEXED, user_id UNINDEXED, content'),
        ):
            budget.check()
            backup = name + '_original_rows'
            work.conn.execute(f'CREATE TEMP TABLE {backup} AS SELECT rowid AS original_rowid,* FROM {name}')
            work.conn.execute(f'DROP TABLE {name}')
            work.conn.execute(f"CREATE VIRTUAL TABLE {name} USING fts5({columns}, tokenize='porter unicode61')")
            names = ','.join(c.strip().split()[0] for c in columns.split(','))
            work.conn.execute(f'INSERT INTO {name}(rowid,{names}) SELECT original_rowid,{names} FROM {backup} ORDER BY original_rowid')
            work.conn.execute(f'DROP TABLE {backup}')
        work.conn.commit()
        work.conn.execute('PRAGMA query_only=ON')
        budget.check()
        yield work
    finally:
        work.conn.set_progress_handler(None, 0)
        work.conn.close()


def extend_coarse(coarse, metadata, supplemental):
    """Keep original candidates and baseline eligibility; bound additional CE work."""
    seen = {c['id'] for c in coarse}
    extra = []
    for candidate in supplemental:
        if candidate['id'] not in seen and len(extra) < LIMIT:
            seen.add(candidate['id'])
            extra.append(candidate)
    revised = dict(metadata, selected_ids=[c['id'] for c in coarse + extra],
                   output_count=len(coarse) + len(extra), supplemental_ids=[c['id'] for c in extra],
                   supplemental_limit=LIMIT, limit=metadata['limit'] + LIMIT)
    return coarse + extra, revised


def _merge_relations(plan, proposal, new_ids):
    for name in ('_fusion_edges', '_fusion_triples'):
        merged = list(plan.get(name, []))
        seen = {json.dumps(r, sort_keys=True) for r in merged}
        for relation in proposal.get(name, []):
            involved = {relation.get(k) for k in ('source', 'target', 'amu_id')}
            key = json.dumps(relation, sort_keys=True)
            if involved & new_ids and key not in seen:
                merged.append(relation)
                seen.add(key)
        plan[name] = merged


def augment(st, req, plan, routes, fused, candidates):
    """Nominate new lexical candidates after unchanged recall and follow-up."""
    if not config.RECALL_PORTER_SUPPLEMENT:
        return fused, candidates
    if not (config.SOURCE_RECALL_DIVERSITY and not config.SOURCE_RECALL_NOVELTY
            and config.COARSE_EVIDENCE_EXTEND and config.FINE_COARSE_CORE_QUOTA
            and config.FINE_EVIDENCE_ADAPTIVE):
        plan['_porter_recall'] = dict(status='incompatible_settings')
        return fused, candidates
    from . import cascade_rerank, search_pipeline as search
    started = time.monotonic()
    # Mutate the request plan only after all optional computation succeeds.
    proposal = copy.deepcopy(plan)
    specs = [q for q in plan['_query_specs'] if q['origin'] != 'grounded_followup']
    try:
        with lexical_view(st, req.user_id) as lexical:
            fts, sources = search._lexical_recall(lexical, req, proposal, specs)
        known = {c['id'] for route in routes for c in route}
        new_route = search._merge_query_results(
            [[c for c in group if c['id'] not in known] for group in (fts, sources)], LIMIT)
        new_route = [c for c in new_route if search._profile_not_expired(
            c, plan['_anchor'], plan['_include_history'])]
        nominated = {c['id'] for c in new_route}
        proposed_routes = copy.deepcopy(routes)
        proposal['_round'] = 1
        search._route(proposal, proposed_routes, 'lexical_porter', new_route, ' | '.join(s['text'] for s in specs))
        new_fused = search._fuse_candidates(st, req, proposal, proposed_routes)
        prepared = search._foresight_filter(search._prepare_candidates(st, req, proposal, new_fused), proposal)
        ordinary = [c for c in prepared if c.get('view_status') != 'stale' and not cascade_rerank.protected(c)]
        shortlist, _ = cascade_rerank.coarse_shortlist(ordinary, proposal.get('_coverage_requirements', []))
        extra = [c for c in shortlist if c['id'] in nominated]
        original_ids = {c['id'] for c in candidates}
        if any(c['id'] in original_ids for c in extra):
            raise ValueError('Supplement overlaps the original prepared pool')
        extra_ids = {c['id'] for c in extra}
        info = dict(status='ok', limit=LIMIT, nominated_ids=[c['id'] for c in new_route],
            admitted_ids=[c['id'] for c in extra], tokenizer='porter unicode61',
            original_coarse_preserved=True, fusion=proposal.get('_graph_fusion', {}),
            omissions=proposal.get('_unit_omitted', []),
            lexical_ids=dict(full_text=[c['id'] for c in fts], source_text=[c['id'] for c in sources]))
    except Exception as exc:
        budget.check()
        plan['_porter_recall'] = dict(status='error', error=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
        plan.setdefault('_recall_errors', []).append(dict(channel='lexical_porter', error=type(exc).__name__))
        return fused, candidates
    info['elapsed_seconds'] = time.monotonic() - started
    plan['_porter_recall'] = info
    plan['_porter_supplement_ids'] = [c['id'] for c in extra]
    plan.setdefault('_routes', []).append(proposal['_routes'][-1])
    _merge_relations(plan, proposal, extra_ids)
    # The old graph is unchanged; the separately computed new nodes retain
    # their own recorded scores. This is explicitly not a single joint ranking.
    plan['_graph_fusion'] = dict(plan.get('_graph_fusion', {}), supplemental_mode='retained_original_coarse',
                                supplemental_ids=[c['id'] for c in extra])
    return fused + [c for c in new_fused if c['id'] in extra_ids], candidates + extra
