from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_stage1 import compare, correct, paired_interval


def test_paired_interval_brackets_mean_and_is_deterministic():
    diffs = [1] * 30 + [0] * 60 + [-1] * 10
    a, b = paired_interval(diffs, 2000), paired_interval(diffs, 2000)
    assert a == b and a['low'] <= a['mean'] == .2 <= a['high'] and a['low'] > 0


def test_correct_follows_metric_rules():
    noul = {'primitive': 'noul', 'label': '1', 'question': {'type': 'noul'}}
    assert correct(noul, {'answer': {'noul': .5}}) and not correct(noul, {'answer': {'noul': .49}})
    assert not correct(noul, {'error': 'failed', 'answer': None})


def test_compare_groups_by_source():
    records = [{'id': str(i), 'source': 's' + str(i % 2), 'primitive': 'noul', 'label': '1',
                'question': {'type': 'noul'}} for i in range(20)]
    new = {r['id']: {'answer': {'noul': .9}} for r in records}
    old = {r['id']: {'answer': {'noul': .1 if int(r['id']) % 2 else .9}} for r in records}
    out = compare(records, new, old, resamples=500)
    assert out['overall']['mean'] == .5 and out['by_source']['s1']['mean'] == 1 and out['by_source']['s0']['mean'] == 0
