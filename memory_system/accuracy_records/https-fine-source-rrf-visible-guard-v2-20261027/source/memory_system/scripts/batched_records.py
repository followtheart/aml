"""Binary block readers for evaluation artifacts; never iterate file lines."""
import csv
import hashlib
import json

BLOCK_BYTES = 4 * 1024 * 1024


def physical_records(path, *, partial=False, block_bytes=BLOCK_BYTES):
    if block_bytes < 1:
        raise ValueError('Block size must be positive')
    pending = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(block_bytes):
            batch = block.split(b'\n')
            if len(batch) == 1:
                pending.extend(block)
                continue
            pending.extend(batch[0])
            yield bytes(pending) + b'\n'
            pending.clear()
            for record in batch[1:-1]:
                yield record + b'\n'
            pending.extend(batch[-1])
        if pending and not partial:
            yield bytes(pending)


def json_records(path, *, partial=False, block_bytes=BLOCK_BYTES):
    for record in physical_records(path, partial=partial, block_bytes=block_bytes):
        if record.strip():
            yield json.loads(record)


def csv_records(path, *, block_bytes=BLOCK_BYTES):
    yield from csv.DictReader(record.decode('utf-8') for record in physical_records(path, block_bytes=block_bytes))


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()
