import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_release_data import select_fresh
from evaluate_release import accept_canonical, ordered_rows


def test_release_excludes_prior_ids_and_duplicate_content():
    pool = [{"id": "old", "state_sha256": "unseen"},
            {"id": "new-id", "state_sha256": "seen"},
            {"id": "valid", "state_sha256": "fresh"}]
    assert select_fresh(pool, 1, {"old"}, {"seen"}) == [pool[2]]
    with pytest.raises(ValueError, match="found 1"):
        select_fresh(pool, 2, {"old"}, {"seen"})


def test_readout_gate_uses_both_development_metrics():
    original = {'nll': {'mean': .5}, 'accuracy_failures_incorrect': .8}
    assert accept_canonical(original, {'nll': {'mean': .52}, 'accuracy_failures_incorrect': .8})
    assert not accept_canonical(original, {'nll': {'mean': .56}, 'accuracy_failures_incorrect': .9})
    assert not accept_canonical(original, {'nll': {'mean': .4}, 'accuracy_failures_incorrect': .77})


def test_ordering_does_not_mutate_the_frozen_input():
    records = [{'primitive': 'choice', 'question': {'criteria': {'z': None, 'a': 'first'}}}]
    result = ordered_rows(records, True)
    assert list(result[0]['question']['criteria']) == ['a', 'z']
    assert list(records[0]['question']['criteria']) == ['z', 'a']
