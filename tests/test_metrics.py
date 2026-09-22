import pytest

from shingi.metrics import human_distribution, probabilities, summarize


def record(identifier="x", kind="choice", label="a", soft=None):
    criteria = {"a": None, "b": None} if kind == "choice" else ["low", "high"]
    return {"id": identifier, "primitive": kind, "question": {"type": kind, "criteria": criteria},
            "label": label, "soft_label": soft}


def test_accuracy_counts_failures_and_probability_denominators_are_explicit():
    records = [record("x"), record("missing")]
    out = summarize(records, {"x": {"answer": {"choice": "a", "probabilities": {"a": .8, "b": .2}}}})
    assert out["accuracy_failures_incorrect"] == .5
    assert out["valid"] == 1
    assert out["nll"]["n"] == 1
    assert out["brier_multiclass"]["mean"] == pytest.approx(.08)
    assert out["ece_top_label"] == pytest.approx(.2)


def test_ordinal_metrics_respect_distance_and_expected_value():
    out = summarize([record(kind="score", label="1")], {"x": {"answer": {"probabilities": {"0": .25, "1": .75}}}})
    assert out["score_mae"]["mean"] == .25
    assert out["rps"]["mean"] == .0625


def test_noul_tie_uses_explicit_half_threshold():
    out = summarize([record(kind="noul", label="1")], {"x": {"answer": {"noul": .5}}})
    assert out["correct"] == 1


@pytest.mark.parametrize("kind,soft,expected", [
    ("choice", {"a": 3, "b": 1}, {"a": .75, "b": .25}),
    ("score", [.2, .8], {"0": .2, "1": .8}),
    ("noul", .3, {"0": .7, "1": .3}),
])
def test_human_distributions_support_all_three_dataset_encodings(kind, soft, expected):
    assert human_distribution(record(kind=kind, soft=soft)) == pytest.approx(expected)


def test_rounding_normalization_is_explicit_and_never_hides_missing_options():
    row = record()
    answer = {"probabilities": {"a": .6, "b": .39}}
    with pytest.raises(ValueError):
        probabilities(row, answer)
    assert sum(probabilities(row, answer, allow_rounding=True).values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        probabilities(row, {"probabilities": {"a": 1}}, allow_rounding=True)
