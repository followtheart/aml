"""Download pinned, pre-built PersonaMem-v3 users with per-file integrity receipts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.personamem_v3 import APPS
from download_datasets import fetch

REVISION = '7b00a090b35b7293e6efeeb19494207f32b5a9ee'
SOURCE = 'https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3'


def download_users(users, output_dir):
    if any(not u.isdigit() for u in users) or len(set(users)) != len(users):
        raise ValueError('Users must be distinct numeric IDs')
    output_dir = Path(output_dir)
    files = {}
    for uid in users:
        for name in [*(a + '.json' for a in APPS), 'test.json']:
            relative = f'backend/{uid}/{name}'
            files[relative] = fetch(f'{SOURCE}/resolve/{REVISION}/{relative}', output_dir / relative)
        print(f'personamem-v3: user {uid} verified', flush=True)
    meta = dict(dataset='personamem-v3', revision=REVISION, users=users, files=files,
                source=SOURCE, repository='https://github.com/bowen-upenn/PersonaMem-v3',
                license='CC-BY-NC-4.0 (upstream derived data; code MIT)',
                scope='Five app histories and test queries; no profile oracle')
    temp = output_dir / 'index.json.partial'
    temp.write_text(json.dumps(meta, indent=2) + '\n')
    temp.replace(output_dir / 'index.json')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--users', nargs='+', default=['1'])
    p.add_argument('--output-dir', type=Path, default=Path('data/raw/personamem-v3'))
    args = p.parse_args()
    download_users(args.users, args.output_dir)


if __name__ == '__main__':
    main()
