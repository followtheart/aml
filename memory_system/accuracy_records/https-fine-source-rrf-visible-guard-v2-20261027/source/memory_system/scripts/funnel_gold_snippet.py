"""Funnel audit driven by PersonaMem-v2 gold metadata (no cue-word heuristics).

For every QA the raw CSV names the gold `preference` and the exact
`related_conversation_snippet` (the user turns that revealed it). The funnel
then asks, stage by stage, whether that evidence survived:

  S0 raw      snippet user turn(s) were ingested as source messages
  S1 memory   a non-episode memory cites one of those turns
  S2 pool     such a memory (or an episode covering the turn) is in `fused`
  S3 packet   it is in the returned packet / answer source catalog
  S4 support  gold option has a claim with a valid citation
  S5 eligible gold letter is in answer_eligible_options
  S6 select   gold eligible but another option chosen

Standard library only. Usage:
  python scripts/funnel_gold_snippet.py --run RUN_DIR [--all] [--json OUT]
  python scripts/funnel_gold_snippet.py --compare RUN_A RUN_B
"""
import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CSV = HERE / 'data/raw/personamem-v2.csv'
PREPARED = HERE / 'data/prepared/personamem-v2-32k.jsonl'
csv.field_size_limit(sys.maxsize)


def load(p):
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]


def norm(s):
    return ' '.join((s or '').split()).casefold()


def gold_rows():
    rows = {}
    with CSV.open(encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                question = ast.literal_eval(r['user_query'])['content']
            except Exception:
                continue
            rows[norm(question)] = r
    return rows


def snippet_user_turns(row):
    try:
        snip = json.loads(row['related_conversation_snippet'])
    except Exception:
        try:
            snip = ast.literal_eval(row['related_conversation_snippet'])
        except Exception:
            return []
    return [norm(s['content']) for s in snip if s.get('role') == 'user' and s.get('content')]


def contains_turn(text, turns):
    t = norm(text)
    return any(u[:120] in t or t[:120] in u for u in turns if u)


def audit(run_dir, include_correct=False):
    base = run_dir / 'completed'
    memory = load(base / 'logs/memory-debug.jsonl')
    searches = {s['search_id']: s for s in load(base / 'logs/search-debug.jsonl')}
    results = load(base / 'data/results/personamem-v2-32k.jsonl')
    qa = {(d['conversation_id'], q['id']): q for d in load(PREPARED) for q in d['qa']}
    gold = gold_rows()

    sources, memories = {}, {}
    for e in memory:
        for s in e.get('source_messages', []):
            sources[(s['request_id'], s['message_index'])] = s
        for m in e.get('memories', []):
            memories[m['id']] = m

    rows = []
    for r in results:
        if r['score'] >= 1 and not include_correct:
            continue
        q = qa[(r['conversation_id'], r['qa_id'])]
        g = gold.get(norm(q['question']))
        if g is None:
            continue
        turns = snippet_user_turns(g)
        gold_letter = q['gold_labels'][0]
        s = searches.get(r['search_id'], {})

        raw_keys = {k for k, v in sources.items() if v['role'] == 'user' and contains_turn(v['content'], turns)}
        mem_hits = [mid for mid, m in memories.items() if m.get('type') != 'episode' and any(
            (z.get('request_id'), z.get('message_index')) in raw_keys for z in (m.get('sources') or []))]
        episode_hits = [mid for mid, m in memories.items() if m.get('type') == 'episode'
                        and contains_turn(m.get('content', ''), turns)]
        fused_ids = {f['id'] for f in s.get('fused', [])}
        included = set(s.get('coverage_manifest', {}).get('included_ids', []))
        catalog = r.get('answer_source_catalog', [])
        catalog_hit = [c['id'] for c in catalog if c.get('role') == 'user' and contains_turn(c.get('text', ''), turns)]
        gold_entry = next((e for e in r.get('choice_alignment', []) if e['letter'] == gold_letter), {})
        gold_cites = [c for cl in gold_entry.get('claims', []) for c in cl.get('citations', []) if c.get('valid')]
        pred_entry = next((e for e in r.get('choice_alignment', []) if e['letter'] == r['prediction']), {})

        row = dict(
            qa=r['qa_id'], scenario=g['conversation_scenario'], pref_type=g['pref_type'],
            preference=g['preference'][:70], gold=gold_letter, pred=r['prediction'], score=r['score'],
            raw=len(raw_keys), mem=len(mem_hits), episodes=len(episode_hits),
            pool=len([m for m in mem_hits + episode_hits if m in fused_ids]),
            packet=len([m for m in mem_hits + episode_hits if m in included]),
            catalog=len(catalog_hit), gold_status=gold_entry.get('status'), gold_cites=len(gold_cites),
            pred_status=pred_entry.get('status'),
            eligible=''.join(r.get('answer_eligible_options', [])),
            selection=r.get('answer_selection'),
            mem_contents=[memories[m]['content'][:90] for m in mem_hits[:4]],
        )
        if row['raw'] == 0:
            row['stage'] = 'S0 snippet turn not ingested'
        elif row['mem'] == 0:
            row['stage'] = 'S1 no fact extracted from snippet turn'
        elif row['pool'] == 0 and row['catalog'] == 0:
            row['stage'] = 'S2 recall missed'
        elif row['packet'] == 0 and row['catalog'] == 0:
            row['stage'] = 'S3 packet/rerank dropped'
        elif row['gold_cites'] == 0:
            row['stage'] = 'S4 gold option unsupported by judge'
        elif gold_letter not in row['eligible']:
            row['stage'] = 'S5 gold ineligible'
        elif r['score'] < 1:
            row['stage'] = 'S6 wrong selection among eligible'
        else:
            row['stage'] = 'OK'
        rows.append(row)
    return rows


def print_rows(rows):
    for row in rows:
        print(f"QA{row['qa']:>3} ok={row['score']:.0f} {row['scenario']:<20} gold={row['gold']} pred={row['pred']} | "
              f"raw={row['raw']} mem={row['mem']} ep={row['episodes']} pool={row['pool']} pkt={row['packet']} cat={row['catalog']} "
              f"gold[{row['gold_status']}/{row['gold_cites']}c] pred[{row['pred_status']}] elig={row['eligible']:<4} "
              f"sel={row['selection']} -> {row['stage']}")
        print(f"       pref: {row['preference']}")
        for c in row['mem_contents']:
            print(f"       mem: {c}")
    print('\nSTAGE SUMMARY:', dict(Counter(r['stage'] for r in rows)))
    by_scn = Counter((r['scenario'], r['score'] >= 1) for r in rows)
    scns = sorted({r['scenario'] for r in rows})
    print('BY SCENARIO (correct/total):', {s: f"{by_scn[(s, True)]}/{by_scn[(s, True)] + by_scn[(s, False)]}" for s in scns})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--json', type=Path)
    ap.add_argument('--compare', nargs=2, type=Path, metavar=('RUN_A', 'RUN_B'))
    args = ap.parse_args()
    if args.compare:
        a = {r['qa']: r for r in audit(args.compare[0], True)}
        b = {r['qa']: r for r in audit(args.compare[1], True)}
        print(f"{'QA':>3} {'scenario':<20} {'A':>3} {'B':>3}  stage A -> stage B")
        for k in sorted(a, key=int):
            ra, rb = a[k], b.get(k, {})
            flag = '' if ra['score'] == rb.get('score') else (' +' if rb.get('score', 0) > ra['score'] else ' -')
            print(f"{k:>3} {ra['scenario']:<20} {ra['score']:>3.0f} {rb.get('score', float('nan')):>3.0f}{flag:<2} "
                  f"{ra['stage']} -> {rb.get('stage')}")
        sa, sb = sum(r['score'] for r in a.values()), sum(r['score'] for r in b.values())
        print(f"\nTOTAL A={sa:.0f}/{len(a)}  B={sb:.0f}/{len(b)}")
        return
    rows = audit(args.run or HERE / 'runs/latest', args.all)
    print_rows(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
