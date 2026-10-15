"""Download pinned PERMA MCQ histories and metadata, without persona or oracle files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_datasets import fetch

REVISION = '440e64e4fb8baec6f7ad10c1de135505f93e7cb1'
VARIANTS = {'clean': 'c', 'noisy': 'n', 'style': 's'}
SOURCE = 'https://huggingface.co/datasets/ustclsc/PERMA'


def download_users(users, output_dir, variant='clean', domains='both', workers=4):
    if not users or len(set(users)) != len(users) or any(not re.fullmatch(r'user\d+', u) for u in users):
        raise ValueError('Expected distinct user IDs such as user108')
    if variant not in VARIANTS or domains not in ('single','multi','both') or not 1 <= workers <= 16:
        raise ValueError('Invalid variant, domains or workers')
    root = Path(output_dir)
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/ustclsc/PERMA/revision/{REVISION}', timeout=60))
    available = {s['rfilename'] for s in info['siblings']}
    histories = [f'tasks/{uid}/input_data{multi}_{VARIANTS[variant]}.json' for uid in users
                 for multi in (['','_multi'] if domains=='both' else ['_multi' if domains=='multi' else ''])]
    if not set(histories) <= available:
        raise ValueError('Requested user/history is not in the pinned dataset')
    files = {}
    for relative in histories:
        files[relative] = fetch(f'{SOURCE}/resolve/{REVISION}/{relative}', root / relative)
    metadata = set()
    for relative in histories:
        uid = relative.split('/')[1]
        data = json.loads((root/relative).read_text())
        for row in data['overall']:
            task, stage = row['task_id'], row['type']
            if not re.fullmatch(r'(SD-[A-Za-z]+|MD)-task-\d+', task) or type(stage) is not int or stage not in (1,2,3):
                raise ValueError('Unexpected task identity')
            metadata.add(f'evaluation/{uid}/meta/overall/{task}_{stage}.json')
    if not metadata <= available:
        raise ValueError('Missing upstream MCQ metadata')
    def get(relative):
        return relative, fetch(f'{SOURCE}/resolve/{REVISION}/{relative}', root/relative)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files.update(pool.map(get, sorted(metadata) + ['README.md']))
    index = dict(dataset='perma',revision=REVISION,users=users,variant=variant,domains=domains,
                 histories=histories,files=files,source=SOURCE,license='Apache-2.0 (dataset card)',
                 protocol='perma-mcq-local-v1',source_questions=len(metadata))
    tmp=root/'index.json.partial';tmp.write_text(json.dumps(index,indent=2)+'\n');tmp.replace(root/'index.json')
    print(f'PERMA: {len(users)} users, {len(histories)} histories, {len(metadata)} MCQs verified',flush=True)
    return index


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--users',nargs='+',default=['user108'])
    p.add_argument('--variant',choices=VARIANTS,default='clean')
    p.add_argument('--domains',choices=['single','multi','both'],default='both')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output-dir',type=Path,default=Path('data/raw/perma'))
    a=p.parse_args();download_users(a.users,a.output_dir,a.variant,a.domains,a.workers)
