"""Use a separate partial-CE audit without overwriting a failed strict audit.

All frozen final-summary checks remain; only the selection-audit filename and
its terminal exit-code filename change. Preserve original failures as evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    root=args.run.resolve();plan=json.loads((root/'plan.json').read_text())
    relative='memory_system/scripts/summarize_new_coarse_https.py'
    script=root/'source'/relative
    assert sha(script)==plan['source_files'][relative]
    source=script.read_text()
    substitutions=[("'selection-admission-audit.json'","'selection-admission-partial-audit.json'"),
                   ("for stage in ('funnel','runtime','selection'):","for stage in ('funnel','runtime','selection-partial'):")]
    for old,new in substitutions:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    env=dict(__name__='frozen_partial_ce_summary',__file__=str(script))
    exec(compile(source,'<frozen-summary-with-partial-audit>','exec'),env)
    result=env['summarize'](root)
    audit=json.loads((root/'selection-admission-partial-audit.json').read_text())
    assert audit['partial_ce_support']['provider_calls']==0
    result['partial_ce_audit']=dict(details=audit['partial_ce_support'],
        original_strict_exit_code=(root/'final-selection.exit-code').read_text().strip(),
        strict_failure_log_sha256=sha(root/'final-selection-audit.log'),
        wrapper_sha256=sha(Path(__file__)),substitutions=substitutions)
    result['inputs'][str(Path(__file__).resolve())]=sha(Path(__file__))
    with (root/'final-evaluation.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','direct_selection','answer_loss_details','nomination_paths')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args())
