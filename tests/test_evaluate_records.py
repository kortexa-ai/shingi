import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_records import load_split


def test_load_split_requires_frozen_checksum(tmp_path):
    rows = [{'id': 'a', 'source': 's'}, {'id': 'b', 'source': 's'}]
    split = tmp_path / 'dev.jsonl'
    split.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps({'dev_sha256': digest}))
    records, manifest_sha = load_split(tmp_path, 'dev')
    assert records == rows and len(manifest_sha) == 64
    split.write_text(split.read_text() + json.dumps({'id': 'c', 'source': 's'}) + '\n')
    with pytest.raises(ValueError, match='frozen split changed'):
        load_split(tmp_path, 'dev')
    with pytest.raises(ValueError, match='frozen split changed'):
        load_split(tmp_path, 'test')
