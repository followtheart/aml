"""Transport diagnostics must remain independent of reader eligibility."""
from audit_https_funnel import classify, snippet_transport_stage


def main():
    row = dict(error_stage=None, observed_add=1, observed_search=True,
               gold='A', eligible=['A'], score=0, snippet_turns=1,
               prepared_hits=1, raw_hits=1, memory_hits=1, pool_hits=1,
               packet_hits=0, catalog_hits=[])
    assert classify(row) == 'S6:selection'
    assert snippet_transport_stage(row) == 'S3:packet'
    assert snippet_transport_stage(dict(row, score=1)) == 'S3:packet'
    assert snippet_transport_stage(dict(row, eligible=[])) == 'S3:packet'
    assert snippet_transport_stage(dict(row, constraint_catalog_hits=['r0'])) == 'delivered:constraint_catalog'
    # A card count alone does not show that the target request was delivered.
    assert snippet_transport_stage(dict(row, constraint_catalog_size=3)) == 'S3:packet'
    # A reconstructed or expanded source can arrive despite absent AMU links.
    assert snippet_transport_stage(dict(row, memory_hits=0, pool_hits=0,
                                        packet_hits=1, catalog_hits=['s0'])) == 'delivered:source_catalog'
    stages = [('catalog_hits', [], 'S3:source_catalog'),
              ('packet_hits', 0, 'S3:packet'), ('pool_hits', 0, 'S2:recall'),
              ('memory_hits', 0, 'S1:memory'), ('raw_hits', 0, 'S0:ingestion'),
              ('prepared_hits', 0, 'S0:snippet_absent_from_dataset')]
    row.update(packet_hits=1, catalog_hits=['s0'])
    for key, value, expected in stages:
        row[key] = value
        assert snippet_transport_stage(row) == expected
    assert snippet_transport_stage(dict(row, snippet_turns=0)) == 'unobserved:gold_snippet'
    assert snippet_transport_stage(dict(row, observed_search=False)) == 'unobserved:server_logs'
    print('PASS: independent transport, later delivery, missing evidence and legacy classification')


if __name__ == '__main__':
    main()
