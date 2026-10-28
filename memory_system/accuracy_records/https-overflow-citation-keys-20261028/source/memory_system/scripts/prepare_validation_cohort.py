"""Select the first complete prepared histories absent from saved scored results.

Selection uses dataset order and observed conversation IDs, never answer labels,
scores or reference snippets. Absence is limited to the enumerated local result
files; it does not establish that the histories have never been inspected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(args):
    prepared_bytes=args.prepared.read_bytes()
    lines=[line for line in prepared_bytes.splitlines() if line.strip()]
    conversations=[json.loads(line) for line in lines]
    ids=[c['conversation_id'] for c in conversations]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate prepared histories')
    command=['rg','--files','-uuu',str(args.runs),'-g','*.jsonl']
    paths=sorted(Path(p) for p in subprocess.check_output(command,text=True).splitlines()
        if '/data/results/' in p or Path(p).name=='results.jsonl')
    observed=set();inventory=[];malformed=[]
    for path in paths:
        data=path.read_bytes();count=0;file_ids=set()
        for index,line in enumerate(data.splitlines(),1):
            if not line.strip():continue
            try:row=json.loads(line)
            except (json.JSONDecodeError,UnicodeDecodeError):
                # An interrupted record still counts as exposure when its ID
                # is recoverable. Otherwise refuse a claim of non-exposure.
                match=re.search(rb'"conversation_id"\s*:\s*"([^"\\]+)"',line)
                if not match:raise ValueError(f'Unattributed malformed result: {path}:{index}')
                cid=match[1].decode();observed.add(cid);file_ids.add(cid)
                malformed.append(dict(path=str(path),line=index,conversation_id=cid));continue
            if isinstance(row,dict) and isinstance(row.get('conversation_id'),str) and 'score' in row:
                observed.add(row['conversation_id']);file_ids.add(row['conversation_id']);count+=1
        inventory.append(dict(path=str(path.resolve()),sha256=digest(data),scored_rows=count,conversation_ids=sorted(file_ids)))
    selected=[i for i,cid in enumerate(ids) if cid not in observed][:args.conversations]
    if len(selected)!=args.conversations:raise ValueError('Not enough unobserved prepared histories')
    cohort=[conversations[i] for i in selected]
    question_ids=[str(q['id']) for c in cohort for q in c['qa']]
    if len(set(question_ids))!=len(question_ids):raise ValueError('Duplicate question IDs')
    payload=b'\n'.join(lines[i] for i in selected)+b'\n'
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'prepared.jsonl').write_bytes(payload)
    manifest=dict(protocol=__doc__,selection='First unobserved complete histories in original prepared order',
        input_prepared=str(args.prepared.resolve()),input_sha256=digest(prepared_bytes),cohort_sha256=digest(payload),
        scanned_result_files=len(inventory),scored_rows=sum(v['scored_rows'] for v in inventory),
        observed_conversation_ids=sorted(observed),observed_prepared_indices=[i for i,cid in enumerate(ids) if cid in observed],
        selected=[dict(prepared_index=i,conversation_id=ids[i],questions=len(conversations[i]['qa']),
            sessions=len(conversations[i]['sessions']),messages=sum(len(s) for s in conversations[i]['sessions']),
            source_history=conversations[i]['source_history']) for i in selected],
        questions=len(question_ids),question_ids=question_ids,all_questions_in_selected_histories=True,
        inventory=inventory,malformed_attributed_rows=malformed,script_sha256=digest(Path(__file__).read_bytes()))
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('scanned_result_files','scored_rows','selected','questions','malformed_attributed_rows')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    parser.add_argument('--conversations',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.conversations<1:parser.error('Require a positive number of histories')
    main(args)
