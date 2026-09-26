import hashlib

import pytest

from shingi.calibration import fit, replay
from shingi.decision import Calibration, DecisionEngine


class RecordingBackend:
    def infer(self, prompt, labels):
        return {"logits": [5., -5.], "input_tokens": 20,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def sample(kind="choice", label="b"):
    question = {"type": kind, "instructions": "pick"}
    if kind == "choice":
        question["criteria"] = {"a": None, "b": None}
    row = {"id": kind, "state": "x", "question": question, "primitive": kind,
           "label": label, "split": "validation", "input_sha256": "input"}
    answer, traces = DecisionEngine(RecordingBackend()).answer(row["state"], question)
    return row, {"answer": answer, "traces": traces, "error": None, "input_sha256": "input"}


def test_calibration_replay_matches_uncalibrated_engine():
    row, prediction = sample()
    assert replay(row, prediction, Calibration()) == prediction["answer"]


def test_calibration_rejects_test_data_and_changed_prompts():
    row, prediction = sample()
    row["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        fit([row], {row["id"]: prediction})
    row["state"] = "different"
    with pytest.raises(ValueError, match="prompt differs"):
        replay(row, prediction, Calibration())


def test_calibration_reduces_overconfident_validation_loss():
    choice, p_choice = sample()
    noul, p_noul = sample("noul", "0")
    result = fit([choice, noul], {"choice": p_choice, "noul": p_noul})
    assert result["after"]["choice_score_nll"] < result["before"]["choice_score_nll"]
    assert result["after"]["noul_nll"] < result["before"]["noul_nll"]
    assert result["parameters"]["temperature"] > 1


def test_frozen_split_fit_allows_non_validation_records_only_when_declared():
    import pytest
    from shingi.calibration import fit
    records = [{'id': 'synth/policy/calibration/0', 'split': 'calibration', 'primitive': 'noul'}]
    with pytest.raises(ValueError, match='validation records only'):
        fit(records, {})


def test_temperature_only_fit_keeps_the_bias_that_the_full_fit_moves(tmp_path):
    import argparse
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from fit_calibration import fit_frozen_split, fit_temperatures
    from shingi.release import sha256
    rows, predictions = [], {}
    for i, (kind, label) in enumerate([('choice', 'b'), ('noul', '0'), ('noul', '0')]):
        row, prediction = sample(kind, label)
        row.update(id=f'synth/{i}', source='civil_comments' if i == 1 else 'synth_state', split='calibration')
        rows.append(row)
        predictions[row['id']] = dict(prediction, id=row['id'])
    biased = fit(rows, predictions, frozen_calibration_split=True, canonical=True)
    assert biased['parameters']['noul_bias'] < 0
    only = fit_temperatures(rows, predictions)
    assert only['parameters']['noul_bias'] == 0 and only['parameters']['temperature'] > 1
    assert only['after']['noul_nll'] < only['before']['noul_nll'] and only['grid']['noul_biases'] == [0.0]
    (tmp_path / 'calibration.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (tmp_path / 'manifest.json').write_text(json.dumps({'calibration_sha256': sha256(tmp_path / 'calibration.jsonl')}))
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'predictions.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in predictions.values()))
    (run / 'receipt.json').write_text(json.dumps({
        'split': 'calibration', 'data_manifest_sha256': sha256(tmp_path / 'manifest.json'), 'calibration_sha256': None,
        'source_revision': 'r', 'model_sha256': 'm', 'adapter_sha256': 'a', 'executable_sha256': 'e'}))
    output = tmp_path / 'fit.json'
    fit_frozen_split(argparse.Namespace(data=tmp_path, split='calibration', run=run, output=output, temperature_only=True))
    result = json.loads(output.read_text())
    assert result['parameters'] == only['parameters'] and 'bias fixed at 0' in result['objective']
    assert result['noul_pool'] == {'records': 2, 'gold_yes': 0, 'gold_yes_share': 0.0, 'by_source': {
        'civil_comments': {'records': 1, 'gold_yes': 0}, 'synth_state': {'records': 1, 'gold_yes': 0}}}
    assert result['provenance']['choice_order'] == 'canonical'
