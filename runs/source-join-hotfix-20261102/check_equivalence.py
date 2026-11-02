"""Compare exact source-search results under eligibility/diversity combinations."""
import itertools
import json
from pathlib import Path
from unittest.mock import patch

from benchmark import legacy_method, store, config
from app import schemas

ROOT = Path(__file__).resolve().parent
st = store.Store(':memory:')
try:
    for user in ['u', 'v']:
        st.save_messages(schemas.AddRequest(request_id=user, user_id=user, session_id='s', messages=[
            schemas.Message(role='user', content='OrchidLedger 冲浪 observation '+str(i % 3),
                            sensitivity='sensitive' if i == 7 else 'suppressed' if i == 11 else 'normal')
            for i in range(16)]))
        for i in range(32):
            aid = st.insert_amu(user_id=user, session_id='s', content=f'Claim {i}',
                                sensitivity='sensitive' if i % 9 == 0 else 'suppressed' if i % 13 == 0 else 'normal')
            mid = f'{user}-memory-{i:02}'
            st.conn.execute('UPDATE amu SET id=? WHERE id=?', (mid, aid))
            st.link_sources(mid, user, sorted({0 if i < 6 else i % 16, (i + 3) % 16}))
            if i % 5 == 0:
                st.conn.execute("UPDATE amu SET valid_to='2025-01-01' WHERE id=?", (mid,))
            if i % 7 == 0:
                st.conn.execute("UPDATE amu SET tier='cold' WHERE id=?", (mid,))
            if i % 11 == 0:
                st.conn.execute("UPDATE amu SET view_status='stale' WHERE id=?", (mid,))
            if i % 17 == 0:
                st.conn.execute("UPDATE amu SET resolution_status='retracted' WHERE id=?", (mid,))
    # Shared observations and malformed cross-user links exercise the joins.
    st.conn.execute("UPDATE source_messages SET source_event_id='shared-event' WHERE message_index IN (1,2,3)")
    st.conn.execute("INSERT INTO amu_sources VALUES ('u-memory-01','v',4)")
    st.conn.commit()
    before = legacy_method()
    count = 0
    for user, query, k, diverse, history, sensitive, cold in itertools.product(
            ['u', 'v', 'missing-user'], ['OrchidLedger', '冲浪'], [1, 2, 5, 50],
            [False, True], [False, True], [False, True], [False, True]):
        with patch.object(config, 'SOURCE_RECALL_DIVERSITY', diverse):
            kwargs = dict(include_history=history, include_sensitive=sensitive, include_cold=cold)
            expected = before(st, user, query, k, **kwargs)
            actual = st.source_search(user, query, k, **kwargs)
            assert actual == expected, (user, query, k, diverse, kwargs)
            count += 1
    report = {'comparisons': count, 'exact_matches': count,
              'compared': 'entire returned dictionaries, including order, scores and content',
              'dimensions': ['user isolation', 'English MATCH', 'Chinese LIKE', 'k=1/2/5/50',
                             'source diversity', 'history', 'sensitive', 'cold',
                             'suppressed/stale/retracted records', 'shared sources', 'cross-user corrupt link'],
              'passed': True}
    (ROOT / 'equivalence.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
finally:
    st.conn.close()
