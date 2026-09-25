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
