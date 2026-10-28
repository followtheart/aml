"""Classify unpaired indexed user messages without inventing cross-request links."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def main(args):
    plan_path = args.run/'exchange-vector-diagnostic/plan.json'
    plan = json.loads(plan_path.read_text())
    rows = []
    with sqlite3.connect((args.run/'frozen-memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory = sqlite3.Row
        for si, members in plan['exchange_sources'].items():
            if len(members) == 2:
                continue
            source = members[0]
            following = db.execute('SELECT * FROM source_messages WHERE request_id=? AND message_index=?',
                (source['request_id'],source['message_index']+1)).fetchone()
            last = db.execute('SELECT max(message_index) FROM source_messages WHERE request_id=?',
                (source['request_id'],)).fetchone()[0]
            if following is None:
                reason = 'end_of_request' if source['message_index'] == last else 'missing_next_message'
                linked = []
            elif following['role'] != 'assistant':
                reason, linked = 'next_message_not_assistant', []
            else:
                linked = [x[0] for x in db.execute('''SELECT a.id FROM amu a JOIN amu_sources l ON l.amu_id=a.id
                    WHERE l.request_id=? AND l.message_index=? AND a.user_id=?
                    AND a.valid_to IS NULL AND a.sensitivity!='suppressed' AND a.view_status='ready'
                    AND a.resolution_status!='retracted' AND a.tier!='cold' ORDER BY a.id''',
                    (following['request_id'],following['message_index'],source['user_id']))]
                reason = 'no_eligible_assistant_carrier' if not linked else 'no_shared_eligible_carrier'
                if set(linked) & set(source['amu_ids']):
                    raise ValueError('Unexplained missing indexed exchange')
            rows.append(dict(source_index=int(si),request_id=source['request_id'],message_index=source['message_index'],
                user_id=source['user_id'],reason=reason,user_amu_ids=source['amu_ids'],
                assistant_amu_ids=linked,source_content_sha256=hashlib.sha256(source['content'].encode()).hexdigest()))
    result = dict(sources=len(plan['sources']),paired_sources=len(plan['sources'])-len(rows),
        unpaired_sources=len(rows),reasons=dict(Counter(r['reason'] for r in rows)),rows=rows,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='No cross-request temporal adjacency is inferred from opaque request IDs or ingestion order.')
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
