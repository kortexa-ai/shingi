import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_training_data import select, training_prompt


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
