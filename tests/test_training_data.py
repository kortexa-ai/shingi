import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_training_data import select, training_prompt
from prepare_training_data import APACHE_SOURCES
from train_decision_adapter import validate_fitting_sources
import pytest


def test_apache_fitting_rejects_share_alike_sources():
    manifest = {'profile':'apache-v2', 'train_sources':list(APACHE_SOURCES)}
    validate_fitting_sources(manifest, [{'source':'mmlu'}], [{'source':'civil_comments'}])
    for source in ('arc_challenge', 'boolq', 'unknown'):
        with pytest.raises(ValueError, match='non-allowlisted'):
            validate_fitting_sources(manifest, [{'source':source}], [])
        with pytest.raises(ValueError, match='non-allowlisted'):
            validate_fitting_sources(manifest, [], [{'source':source}])
    manifest['train_sources'].append('boolq')
    with pytest.raises(ValueError, match='exact audited'):
        validate_fitting_sources(manifest, [], [])


def test_empty_available_source_does_not_select_a_record():
    assert select([{'id':'one','state_sha256':'one'}], 0, set(), set(), 'empty') == []


def test_exclude_duplicate_state_across_different_record_ids():
    ids={'old'};states={'seen-state'}
    rows=[{'id':'new-id','state_sha256':'seen-state'}, {'id':'valid','state_sha256':'fresh'}]
    assert select(rows,1,ids,states,'split')[0]['id']=='valid'
    assert 'fresh' in states


def test_many_choices_preserve_gold_and_correct_letter_target():
    row={'id':'example','source':'example','primitive':'choice','state_sha256':'hash','state':'find 58',
         'label':'58','question':{'type':'choice','instructions':'Pick the number','criteria':{str(i):str(i) for i in range(60)}}}
    for variant in range(8):
        p=training_prompt(row,variant)
        assert len(p['labels'])<=52
        assert p['option_keys'][p['hard_target']]=='58'
        assert p['target'][p['hard_target']]==1.0
        assert sum(p['target'])==1.0


def test_soft_noul_target_follows_semantic_option_after_permutation():
    row={'id':'example','source':'example','primitive':'noul','state_sha256':'hash','state':'text',
         'label':'1','soft_label':.75,'question':{'type':'noul','instructions':'Is it true?'}}
    for variant in range(8):
        p=training_prompt(row,variant)
        assert dict(zip(p['option_keys'],p['target']))=={'yes':.75,'no':.25}
        assert p['option_keys'][p['hard_target']]=='yes'
