import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from flip_analysis import analyse, consistency, load_run, option_bucket, state_bucket, state_chars, tally


def noul(i, label='1', group=None, subset='hard'):
    return {'id': f'jevbench/{i}', 'suite': 'jevbench', 'subset': subset, 'family': 'f', 'group': group or str(i),
            'primitive': 'noul', 'label': label, 'state': 'x' * 10, 'question': {'type': 'noul'},
            'input_sha256': f'h{i}'}


def pred(record, yes):
    return {'id': record['id'], 'input_sha256': record['input_sha256'], 'error': None,
            'answer': {'type': 'noul', 'noul': yes}}


def test_four_way_split_counts_every_combination():
    items = [{'old': o, 'new': n} for o, n in [(1, 1)] * 5 + [(1, 0)] * 2 + [(0, 1)] * 3 + [(0, 0)] * 4]
    t = tally(items)
    assert (t['n'], t['old_correct'], t['new_correct']) == (14, 7, 8)
    assert (t['both_correct'], t['old_only'], t['new_only'], t['both_wrong']) == (5, 2, 3, 4)
    assert t['difference']['mean'] == pytest.approx(1 / 14)


def test_buckets_have_closed_boundaries():
    assert [option_bucket(n) for n in (2, 3, 4, 5, 8, 9, 24, 25, 52, 53, 120)] == [
        '2', '3-4', '3-4', '5-8', '5-8', '9-24', '9-24', '25-52', '25-52', '>52', '>52']
    assert [state_bucket(c) for c in (0, 499, 500, 1999, 2000, 7999, 8000)] == [
        '<500', '<500', '500-2k', '500-2k', '2k-8k', '2k-8k', '>8k']
    assert state_chars('abc') == 3 and state_chars({'a': 1}) == len('{"a":1}')


def test_group_consistency_ignores_singletons():
    items = [{'id': i, 'old': o, 'new': n} for i, o, n in
             [('a1', 1, 1), ('a2', 1, 0), ('b1', 1, 1), ('b2', 1, 1), ('c1', 0, 1), ('c2', 0, 1), ('d1', 0, 0)]]
    group_of = {i['id']: i['id'][0] for i in items}
    c = consistency(items, group_of)
    assert (c['groups'], c['items'], c['old_all_correct'], c['new_all_correct']) == (3, 6, 2, 2)
    assert (c['both'], c['old_only'], c['new_only']) == (1, 1, 1)


def test_analyse_reports_flip_confidence_and_structure():
    rows = [noul(0), noul(1), noul(2, '0'), noul(3, subset='easy')]
    old = {r['id']: pred(r, y) for r, y in zip(rows, (.9, .2, .3, .8))}
    new = {r['id']: pred(r, y) for r, y in zip(rows, (.4, .7, .3, .9))}
    result, items = analyse('jevbench', rows, old, new)
    assert result['all']['old_only'] == 1 and result['all']['new_only'] == 1
    flips = result['confidence']['old_only']
    assert flips['new_prob_chosen']['mean'] == pytest.approx(.6) and flips['old_prob_gold']['mean'] == pytest.approx(.9)
    assert result['confidence']['both_correct']['gold_probability_change']['mean'] == pytest.approx(.05)
    assert set(result['subsets']) == {'easy', 'hard'}
    assert result['structure']['all']['by_options']['2']['n'] == 4


def test_inventory_mismatch_fails(tmp_path):
    rows = [noul(0), noul(1)]
    (tmp_path / 'jevbench.jsonl').write_text(json.dumps(pred(rows[0], .9)) + '\n')
    with pytest.raises(SystemExit, match='prediction IDs differ'):
        load_run(tmp_path, rows, 'jevbench')
    changed = dict(pred(rows[1], .9), input_sha256='other')
    (tmp_path / 'jevbench.jsonl').write_text(json.dumps(pred(rows[0], .9)) + '\n' + json.dumps(changed) + '\n')
    with pytest.raises(SystemExit, match='input hash'):
        load_run(tmp_path, rows, 'jevbench')


def test_bucket_tables_keep_declared_order():
    from flip_analysis import OPTION_BUCKETS, grouped
    items = [{'old': 1, 'new': 1, 'n': n} for n in (30, 2, 4)]
    assert list(grouped(items, lambda i: option_bucket(i['n']), [b[2] for b in OPTION_BUCKETS])) == ['2', '3-4', '25-52']
