import json
import math

import pytest

from shingi.external_benchmarks import analytic_metrics, decision, jev, spatial, suite_report


def spatial_row():
    return {"id": "stochastic-0", "family": "stochastic", "state": "grid", "question": "Safe?",
            "options": ["yes", "no"], "answer_index": 0, "answer": "yes",
            "answer_distribution": [.75, .25], "deterministic": False,
            "seen_in_training": False, "fingerprint": "upstream"}


def test_spatial_preserves_option_identity_and_keeps_labels_out_of_prompt():
    raw = spatial_row()
    r = spatial(raw)
    assert r["question"] == {"type": "choice", "instructions": "Safe?", "criteria": {"0": "yes", "1": "no"}}
    raw.update(answer_index=1, answer="no", answer_distribution=[.25, .75])
    other = spatial(raw)
    assert r["input_sha256"] == other["input_sha256"]
    assert other["label"] == "1"
    raw["answer_distribution"] = [1, 1]
    with pytest.raises(ValueError):
        spatial(raw)


def test_analytic_metrics_do_not_confuse_sample_label_with_true_distribution():
    r = spatial(spatial_row())
    p = {r["id"]: {"answer": {"choice": "0", "probabilities": {"0": .75, "1": .25}}}}
    result = analytic_metrics([r], p)
    assert result["expected_accuracy"] == .75
    assert result["tvd"] == 0
    assert result["kl_divergence"] == pytest.approx(0)
    assert result["expected_brier"] == .375
    assert result["cross_entropy"] == pytest.approx(-.75*math.log(.75)-.25*math.log(.25))
    assert analytic_metrics([r], {}) == {"planned": 1, "valid": 0}


def test_decision_boolean_label_and_nested_json_are_decoded_once():
    raw = {"id": "case", "family": "policy", "n_questions": 1,
           "state": json.dumps({"message": "text"}),
           "questions": json.dumps({"q": {"type": "noul", "instructions": "Allowed?"}}),
           "answers": json.dumps({"q": False})}
    r, = decision(raw, "medium")
    assert r["state"] == {"message": "text"}
    assert r["label"] == "0"
    assert r["group"] == "case"
    raw["answers"] = '{"q": "false"}'
    with pytest.raises(ValueError):
        decision(raw, "medium")


def test_null_jev_groups_are_independent_and_failures_remain_in_macro():
    def row(identifier, group):
        return jev({"id": identifier, "group": group, "family": "policy", "state": "state",
                    "expected": "yes", "labels": ["no", "yes"],
                    "question": {"type": "noul", "instructions": "yes?"}}, "easy")
    a, b, c = row("a", "pair"), row("b", "pair"), row("c", None)
    assert c["group"] == "c"
    result = suite_report([a, b, c], {a["id"]: {"answer": {"noul": .9}}})
    assert result["correct"] == 1
    assert result["accuracy_failures_incorrect"] == 1/3
    assert result["group_macro_failures_incorrect"] == .25
    assert result["group_macro_observed_only"] == 1
    assert result["valid"] == 1


def test_score_accuracy_is_argmax_while_distance_uses_expected_value():
    r = jev({"id": "score", "family": "ordinal", "state": "state", "expected": 2,
             "labels": ["0", "1", "2"], "question": {"type": "score", "instructions": "rate",
             "criteria": ["low", "mid", "high"]}}, "original")
    result = suite_report([r], {r["id"]: {"answer": {"probabilities": {"0": .35, "1": .25, "2": .4}}}})
    assert result["correct"] == 1
    assert result["score_mae"]["mean"] == pytest.approx(.95)
    assert result["score_rounded_ev_accuracy_diagnostic"]["correct"] == 0
