import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_data_v3 import balanced_yes_no, build, check_yes_no, ngrams, prior_records, select, write_profile
from shingi.sources_v3 import (EVAL_COUNTS_V31, FIT, FIT_V31, HELPSTEER_ATTRIBUTES, OOD, PROFILES, SYNTHETIC, SYNTHETIC_V31, SkipRow,
                               commonsense_qa_record, gsm8k_judge_pool, gsm8k_judge_record, gsm8k_variant,
                               helpsteer_record, license_source, winogrande_record)
from train_decision_adapter import validate_fitting_sources
from verify_licenses import REGISTRY, card_license


def test_locked_mix_targets_thirty_five_percent_synthetic_tokens():
    natural = sum(count for _, count in FIT.values())
    synthetic = sum(SYNTHETIC.values())
    assert natural == 19500 and synthetic == 6100
    # Synthetic prompts average about 1.4x natural tokens, so 24% of records is about 35% of tokens.
    assert synthetic / (natural + synthetic) == pytest.approx(.238, abs=.001)


def test_fitting_and_evaluation_sources_are_disjoint_and_audited():
    assert not set(FIT) & set(OOD)
    fit = {n for n, spec in REGISTRY.items() if spec[0] == 'fit'}
    assert fit == {license_source(n) for profile, _ in PROFILES.values() for n in profile}
    assert not set(FIT_V31) & set(OOD) and REGISTRY['gsm8k'][1:3] == ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866')
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


def test_helpsteer_prompt_groups_ignore_responses():
    from prepare_data_v3 import prompt_group
    a = {'state': {'prompt': 'Write a poem.', 'response': 'One.'}}
    b = {'state': {'prompt': 'Write a poem.', 'response': 'Two.'}}
    assert prompt_group(a) == prompt_group(b) and prompt_group({'state': 'text'}) is None


def test_v31_mix_changes_only_helpsteer_and_adds_gsm8k():
    assert sum(count for _, count in FIT_V31.values()) == 18900
    changed = {s: FIT_V31[s][1] for s in FIT_V31 if FIT.get(s) != FIT_V31[s]}
    assert changed == {'helpsteer': 500, 'helpsteer2_helpfulness': 700, 'helpsteer2_verbosity': 300,
                       'helpsteer2_correctness': 300, 'helpsteer2_coherence': 300, 'helpsteer2_complexity': 300,
                       'gsm8k_judge': 1500}
    assert PROFILES['v3'] == (FIT, SYNTHETIC)
    assert set(SYNTHETIC) < set(SYNTHETIC_V31) and set(SYNTHETIC_V31) - set(SYNTHETIC) == {
        'synth_judge', 'synth_severity', 'synth_arithmetic'}


def test_v31_evaluation_count_exceptions_name_real_sources():
    for (source, split), count in EVAL_COUNTS_V31.items():
        assert source in (OOD if split == 'ood' else FIT_V31) and split in ('test', 'ood', 'calibration') and count > 0
    # Helpfulness and verbosity share one pool of 200 free test states.
    assert EVAL_COUNTS_V31['helpsteer2_helpfulness', 'test'] + EVAL_COUNTS_V31['helpsteer2_verbosity', 'test'] == 200


def test_v31_synthetic_counts_keep_the_proposed_proportions(monkeypatch):
    from argparse import Namespace
    assert sum(SYNTHETIC_V31.values()) == 6300 and all(count % 50 == 0 for count in SYNTHETIC_V31.values())
    assert SYNTHETIC_V31['synth_judge'] > SYNTHETIC_V31['synth_severity'] == SYNTHETIC_V31['synth_arithmetic']
    monkeypatch.setitem(SYNTHETIC_V31, 'synth_judge', None)
    with pytest.raises(SystemExit, match='counts are not set'):
        build(Namespace(profile='v3.1', licenses=None, output=None))


PROBLEM = 'Tom has 3 boxes with 4 apples each. He buys 5 more apples. How many apples does he have?'
RATIONALE = 'Tom has 3*4=<<3*4=12>>12 apples in boxes.\nWith 5 more he has 12+5=<<12+5=17>>17 apples.\n#### 17'
GOLD = 'Tom has 3*4=12 apples in boxes.\nWith 5 more he has 12+5=17 apples.\nAnswer: 17'


def gsm8k(variant, problem=PROBLEM, answer=RATIONALE, index=0):
    return gsm8k_judge_record('train', index, {'question': problem, 'answer': answer}, variant, 'seed')


def test_gsm8k_gold_strips_markup_and_renders_the_answer():
    record = gsm8k('gold')
    assert record['state'] == {'problem': PROBLEM, 'response': GOLD}
    assert record['label'] == '1' and record['primitive'] == record['question']['type'] == 'noul'
    assert set(record['question']['criteria']) == {'true', 'false'}
    assert record['meta']['upstream_split'] == 'train' and record['meta']['perturbed_step_index'] is None
    assert gsm8k_judge_record('validation', 1, {'question': PROBLEM, 'answer': RATIONALE}, 'gold', 's')['meta'][
        'upstream_split'] == 'test'


def lines(record):
    return record['state']['response'].splitlines()


def test_gsm8k_variants_change_exactly_what_they_claim():
    gold = GOLD.splitlines()
    final = gsm8k('final_wrong')
    wrong = int(lines(final)[2].removeprefix('Answer: '))
    assert wrong != 17 and wrong >= 0 and final['label'] == '0'
    assert lines(final) == gold[:1] + [f'With 5 more he has 12+5={wrong} apples.', f'Answer: {wrong}']
    assert final['meta']['perturbed_step_index'] == 1
    step = gsm8k('step_wrong')
    first = lines(step)[0]
    value = int(first.removeprefix('Tom has 3*4=').removesuffix(' apples in boxes.'))
    assert value != 12 and lines(step)[1:] == gold[1:] and step['label'] == '0'
    assert step['meta']['perturbed_step_index'] == 0
    propagated = gsm8k('step_wrong_propagated')
    value = int(lines(propagated)[0].removeprefix('Tom has 3*4=').removesuffix(' apples in boxes.'))
    assert value != 12 and lines(propagated)[1:] == [f'With 5 more he has {value}+5={value + 5} apples.',
                                                      f'Answer: {value + 5}']
    assert propagated['label'] == '0' and propagated['meta']['perturbed_step_index'] == 0
    for record in (final, step, propagated):
        text = record['state']['response']
        assert '<<' not in text and '>>' not in text and '####' not in text


def test_gsm8k_is_deterministic_and_half_gold():
    assert gsm8k('step_wrong_propagated', index=5) == gsm8k('step_wrong_propagated', index=5)
    variants = [gsm8k_variant('train', i, 'seed') for i in range(2000)]
    assert variants == [gsm8k_variant('train', i, 'seed') for i in range(2000)]
    assert .45 < variants.count('gold') / 2000 < .55 and set(variants) == {
        'gold', 'final_wrong', 'step_wrong', 'step_wrong_propagated'}


def test_gsm8k_skips_perturbations_that_do_not_apply_cleanly():
    with pytest.raises(SkipRow, match='appears in the problem'):
        gsm8k('step_wrong', problem=PROBLEM + ' He had 12 before.')
    with pytest.raises(SkipRow, match='no calculator steps'):
        gsm8k('step_wrong', answer='He has 17 apples.\n#### 17')
    with pytest.raises(SkipRow, match='no intermediate step'):
        gsm8k('step_wrong_propagated', answer='He has 12+5=<<12+5=17>>17.\n#### 17')
    with pytest.raises(SkipRow, match='final answer unchanged'):
        gsm8k('step_wrong_propagated', answer='A is 3*4=<<3*4=12>>12. B is 2+5=<<2+5=7>>7.\n#### 7')
    with pytest.raises(SkipRow, match='negative final'):
        gsm8k('gold', answer='It is 2-5=<<2-5=-3>>-3.\n#### -3')
    rows = [{'question': PROBLEM, 'answer': RATIONALE}, {'question': PROBLEM, 'answer': 'x\n#### -1'}]
    records, skipped = gsm8k_judge_pool('train', rows, 'seed')
    assert len(records) == 1 and sum(skipped.values()) == 1 and 'negative final answer' in next(iter(skipped))


def yes_no_rows(source, yes, no, kind='noul'):
    return [{'id': f'{source}/{i}', 'source': source, 'state_sha256': f'{source}-{i}', 'question': {'type': kind},
             'label': '1' if i < yes else '0'} for i in range(yes + no)]


def test_calibration_yes_no_is_balanced_per_source():
    everything = lambda row: True
    pools = {'a': (yes_no_rows('a', 10, 30), everything), 'b': (yes_no_rows('b', 3, 20), everything),
             'c': (yes_no_rows('c', 0, 5), everything), 'd': (yes_no_rows('d', 0, 9, 'choice'), everything)}
    ids, states = {'a/0'}, set()
    chosen, composition = balanced_yes_no(pools, 5, ids, states, 'calibration')
    assert {s: (c['records'], c['gold_yes']) for s, c in composition.items()} == {'a': (10, 5), 'b': (6, 3), 'c': (0, 0)}
    assert composition['a']['available_yes'] == 9 and 'a/0' not in {r['id'] for r in chosen}
    assert {r['id'] for r in chosen} <= ids and chosen == balanced_yes_no(pools, 5, {'a/0'}, set(), 'calibration')[0]
    with pytest.raises(ValueError, match='gate'):
        check_yes_no(chosen, composition)
    with pytest.raises(ValueError, match='1 sources'):
        check_yes_no(chosen[:10], {'a': composition['a']}, minimum=10)
    summary = check_yes_no(chosen, composition, minimum=16, sources=2)
    assert summary['gold_yes_share'] == .5 and summary['sources'] == 2
    skewed = chosen + [r for r in yes_no_rows('e', 0, 10)]
    with pytest.raises(ValueError, match='gold yes'):
        check_yes_no(skewed, composition, minimum=16, sources=2)


def fixture_record(source, key, kind='noul', label='1', split='train'):
    """A tiny record whose state is unique to source and key, so shared keys share a state."""
    from shingi.sources_v3 import finish
    question = ({'type': 'noul', 'instructions': 'Is it so?', 'criteria': {'true': 'yes', 'false': 'no'}}
                if kind == 'noul' else {'type': 'choice', 'instructions': 'Which?', 'criteria': {'a': 'A', 'b': 'B'}})
    return finish({'id': f'{source}/{split}/{key}', 'source': source, 'primitive': kind, 'split': split,
                   'state': f'{source} state {key}.', 'question': question,
                   'label': label if kind == 'noul' else 'a', 'soft_label': None, 'meta': {}})


def fixture_build(tmp_path, monkeypatch, profile):
    """Run build() on tiny pools: one natural source, one OOD source, one synthetic family plus long context.

    The prior root holds a v3-like dataset whose synthetic evaluation files are byte-identical to the
    current ones, and whose natural test holds two of the natural test records.
    """
    from argparse import Namespace
    from functools import partial
    import prepare_data_v3 as build_module
    from prepare_benchmark import write_jsonl
    nat = {'train': [fixture_record('nat', f't{i}', 'choice') for i in range(6)],
           'validation': [fixture_record('nat', f'v{i}', label=str(int(i < 4)), split='validation') for i in range(8)]
                         + [fixture_record('nat', f'c{i}', 'choice', split='validation') for i in range(6)],
           'test': [fixture_record('nat', f'e{i}', 'choice', split='test') for i in range(6)]}
    pools = {('nat', split): rows for split, rows in nat.items()}
    pools['ood', 'test'] = [fixture_record('ood', f'o{i}', 'choice', split='test') for i in range(3)]
    judge = {'train': [fixture_record('synth_judge', f't{i}') for i in range(4)]
                      # Share states with the calibration records (the balanced pool leaves some yes records
                      # unselected) and with the v3 development record d0.
                      + [fixture_record('synth_judge', key) for key in ('c0', 'c2', 'c3', 'c4', 'd0')],
             'dev': [fixture_record('synth_judge', 'd0', split='dev')],
             'calibration': [fixture_record('synth_judge', f'c{i}', label=str(int(i != 1)), split='calibration')
                             for i in range(5)] + [fixture_record('synth_judge', 'cc', 'choice', split='calibration')],
             'transfer': [fixture_record('synth_judge', 'x0', split='transfer')]}
    longctx = {split: [fixture_record('synth_longctx', f'{split}0', split=split)]
               for split in ('train', 'dev', 'transfer')}
    synth = {('synth_judge', split): rows for split, rows in judge.items()}
    synth |= {('synth_longctx', split): rows for split, rows in longctx.items()}
    prior = tmp_path / 'v3'
    (prior / 'longctx').mkdir(parents=True)
    write_jsonl(prior / 'test.jsonl', nat['test'][:2])
    write_jsonl(prior / 'train.jsonl', judge['train'][:1])
    write_jsonl(prior / 'dev.jsonl', judge['dev'])
    # The yes calibration records are new, as in a new family, so only this build's rule keeps them out of training.
    write_jsonl(prior / 'calibration.jsonl', [r for r in judge['calibration'] if r['label'] != '1'])
    write_jsonl(prior / 'transfer.jsonl', judge['transfer'])
    write_jsonl(prior / 'longctx' / 'dev.jsonl', longctx['dev'])
    write_jsonl(prior / 'longctx' / 'eval.jsonl', longctx['transfer'])
    licenses = tmp_path / 'licenses.json'
    licenses.write_text(json.dumps({'sources': {'nat': {'role': 'fit', 'cleared': True}}}))
    monkeypatch.setitem(PROFILES, profile, ({'nat': ('jev', 2)}, {'synth_judge': 4}))
    for name, value in (('OOD', ('ood',)), ('TEST_PER_SOURCE', 2), ('OOD_PER_SOURCE', 2), ('DEV_PER_SOURCE', 2),
                        ('CALIBRATION_PER_SOURCE', 2), ('YES_NO_PER_LABEL', 2),
                        ('check_yes_no', partial(check_yes_no, minimum=4, sources=2)),
                        ('jev_pools', lambda raw, sources, files: dict(pools)),
                        ('converted_pools', lambda *args: {}),
                        ('synthetic_splits', lambda directory, synthetic: ({'files': {}}, synth))):
        monkeypatch.setattr(build_module, name, value)
    output = tmp_path / 'out'
    build(Namespace(profile=profile, licenses=licenses, output=output, cache=tmp_path / 'cache',
                    synthetic=tmp_path, synthetic_revision='rev', prior=[prior], external=[]))
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
    return {name: {r['id'] for r in read(output / f'{name}.jsonl')}
            for name in ('train', 'test', 'dev', 'calibration', 'transfer')} | {
        'longctx': {r['id'] for name in ('dev', 'eval') for r in read(output / 'longctx' / f'{name}.jsonl')}}


def test_v31_reuses_synthetic_evaluation_and_draws_fresh_natural_evaluation(tmp_path, monkeypatch):
    out = fixture_build(tmp_path, monkeypatch, 'v3.1')
    # Natural evaluation is fresh: the two v3 test records are not drawn again.
    assert len(out['test']) == 2 and not out['test'] & {'nat/test/e0', 'nat/test/e1'}
    # Synthetic evaluation files are reused although v3 already evaluated on them.
    assert out['dev'] >= {'synth_judge/dev/d0'} and out['transfer'] == {'synth_judge/transfer/x0'}
    assert out['longctx'] == {'synth_longctx/dev/dev0', 'synth_longctx/transfer/transfer0'}
    synthetic_calibration = {i for i in out['calibration'] if i.startswith('synth_judge/')}
    # One yes and the only no (c1) from the balanced pool, plus the choice record.
    assert len(synthetic_calibration) == 3 and {'synth_judge/calibration/cc', 'synth_judge/calibration/c1'} <= synthetic_calibration
    # Training excludes every evaluation state: v3 evaluation, and synthetic records selected or not.
    assert {i for i in out['train'] if i.startswith('synth_judge/')} == {f'synth_judge/train/t{i}' for i in range(4)}


def test_v3_profile_still_excludes_earlier_synthetic_evaluation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='synth_judge/calibration: need 6, found 4'):
        fixture_build(tmp_path, monkeypatch, 'v3')
