import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_data_v3 import ngrams, prior_records, select, write_profile
from shingi.sources_v3 import (FIT, HELPSTEER_ATTRIBUTES, OOD, SYNTHETIC, commonsense_qa_record, helpsteer_record,
                               winogrande_record)
from train_decision_adapter import validate_fitting_sources
from verify_licenses import REGISTRY, card_license


def test_locked_mix_is_thirty_thousand_with_thirty_five_percent_synthetic():
    natural = sum(count for _, count in FIT.values())
    synthetic = sum(SYNTHETIC.values())
    assert natural == 19500 and synthetic == 10500
    assert synthetic / (natural + synthetic) == pytest.approx(.35)


def test_fitting_and_evaluation_sources_are_disjoint_and_audited():
    assert not set(FIT) & set(OOD)
    fit = {n for n, spec in REGISTRY.items() if spec[0] == 'fit'}
    assert fit == {'helpsteer2' if n.startswith('helpsteer2_') else n for n in FIT}
    for share_alike in ('boolq', 'fever_evidence', 'arc_challenge', 'chaosnli', 'mnli', 'sst5'):
        assert share_alike in OOD
    assert REGISTRY['hellaswag'][0] == 'excluded' and REGISTRY['wanli'][0] == 'excluded'


def test_card_license_parses_scalar_and_list_front_matter():
    assert card_license('---\nlicense: cc-by-4.0\nlanguage:\n- en\n---\n') == ['cc-by-4.0']
    assert card_license('---\nlicense:\n- mit\n- apache-2.0\npretty_name: x\n---\n') == ['mit', 'apache-2.0']
    assert card_license('# no front matter') == []


def test_converters_emit_isolated_shingi_records():
    scales = {a: ('How?', ['0', '1', '2', '3', '4']) for a in HELPSTEER_ATTRIBUTES}
    row = {'prompt': 'p', 'response': 'r', 'correctness': 3}
    record = helpsteer_record('helpsteer2', 'helpsteer2_correctness', 'train', 7, row, 'correctness', scales)
    assert record['label'] == '3' and len(record['question']['criteria']) == 5
    assert record['state'] == {'prompt': 'p', 'response': 'r'}
    same_state = helpsteer_record('helpsteer2', 'helpsteer2_coherence', 'train', 7, {**row, 'coherence': 4}, 'coherence', scales)
    assert same_state['state_sha256'] == record['state_sha256'] and same_state['id'] != record['id']
    with pytest.raises(ValueError):
        helpsteer_record('helpsteer2', 'helpsteer2_correctness', 'train', 1, {**row, 'correctness': 5}, 'correctness', scales)
    csqa = commonsense_qa_record('train', 0, {'id': 'q1', 'question': 'Where?', 'answerKey': 'B',
                                              'choices': {'label': ['A', 'B'], 'text': ['bank', 'mall']}})
    assert csqa['label'] == 'mall' and list(csqa['question']['criteria']) == ['bank', 'mall']
    assert commonsense_qa_record('train', 0, {'id': 'q', 'question': 'x', 'answerKey': 'A',
                                              'choices': {'label': ['A', 'B'], 'text': ['same', 'same']}}) is None
    wg = winogrande_record('train', 3, {'sentence': 'A beat B because _ trained.', 'option1': 'A', 'option2': 'B', 'answer': '1'})
    assert wg['label'] == 'option1'
    assert winogrande_record('test', 3, {'sentence': 'x _', 'option1': 'A', 'option2': 'B', 'answer': ''}) is None


def test_select_is_deterministic_and_updates_exclusions():
    rows = [{'id': f'r{i}', 'state_sha256': f's{i % 5}'} for i in range(10)]
    ids, states = {'r0'}, set()
    chosen = select(rows, 4, ids, states, 'tag')
    assert chosen == select(rows, 4, {'r0'}, set(), 'tag')
    assert len({r['state_sha256'] for r in chosen}) == 4 and 'r0' not in [r['id'] for r in chosen]
    assert {r['id'] for r in chosen} <= ids
    with pytest.raises(ValueError, match='need 6'):
        select(rows, 6, set(), set(), 'tag')


def test_prior_records_skip_raw_and_separate_training(tmp_path):
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw' / 'test.jsonl').write_text(json.dumps({'id': 'raw-row', 'state_sha256': 'x'}) + '\n')
    (tmp_path / 'train.jsonl').write_text(json.dumps({'id': 't', 'state': '"text"'}) + '\n')
    (tmp_path / 'test.jsonl').write_text(json.dumps({'record_id': 'e', 'state_sha256': 'h'}) + '\n')
    seen, files = prior_records([tmp_path])
    assert seen['train'][0] == {'t'} and seen['eval'] == ({'e'}, {'h'})
    assert len(seen['train'][1]) == 1 and len(files) == 2


def test_ngrams_ignore_case_and_punctuation():
    text = ' '.join(f'Word{i}' for i in range(14))
    assert len(ngrams(text)) == 2 and ngrams(text.upper() + '!') == ngrams(text)


def test_v3_profile_allowlist_and_manifest(tmp_path):
    record = {'id': 'synth/policy/train/0', 'source': 'synth_policy', 'primitive': 'noul', 'state': 'Rule: approve.',
              'question': {'type': 'noul', 'instructions': 'Approve?', 'criteria': {'true': 'yes', 'false': 'no'}},
              'label': '1', 'soft_label': None, 'meta': {}, 'state_sha256': 'a'}
    natural = {**record, 'id': 'banking77/train/1', 'source': 'banking77', 'state_sha256': 'b'}
    manifest = write_profile(tmp_path / 'pilot', 'v3-pilot', [record, natural], [natural], [natural], 'parent')
    assert manifest['synthetic_record_share'] == .5 and manifest['train_sources'] == ['banking77', 'synth_policy']
    validate_fitting_sources(manifest, [record], [natural])
    with pytest.raises(ValueError, match='non-allowlisted'):
        validate_fitting_sources(manifest, [{'source': 'boolq'}], [])


def test_plain_text_states_hash_without_json_parsing(tmp_path):
    from prepare_data_v3 import parse_state
    assert parse_state('Snake game on a board.') == 'Snake game on a board.'
    assert parse_state('"quoted"') == 'quoted' and parse_state('{"a": 1}') == {'a': 1}
    (tmp_path / 'records.jsonl').write_text(json.dumps({'id': 'x', 'state': 'Plain text state.'}) + '\n')
    seen, _ = prior_records([tmp_path])
    assert seen['eval'][0] == {'x'} and len(seen['eval'][1]) == 1
