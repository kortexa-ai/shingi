"""Replay saved raw logits to fit global calibration without new inference."""
from dataclasses import asdict
import hashlib
import math

from .decision import Calibration, DecisionEngine
from .metrics import probabilities

TEMPERATURES = (.5, .65, .8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
BIASES = (-2.0, -1.5, -1.0, -.5, 0.0, .5, 1.0, 1.5, 2.0)


class ReplayReadout:
    def __init__(self, traces):
        self.traces = iter(traces)

    def infer(self, prompt, labels):
        trace = next(self.traces)
        if trace["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError("prompt differs from recorded calibration run")
        if len(trace["logits"]) != len(labels):
            raise ValueError("candidate count differs from recorded calibration run")
        return trace


def replay(record, prediction, calibration):
    if prediction.get("error") or prediction["input_sha256"] != record["input_sha256"]:
        raise ValueError("calibration prediction failed or input hash differs")
    backend = ReplayReadout(prediction["traces"])
    answer, _ = DecisionEngine(backend, calibration).answer(record["state"], record["question"])
    if next(backend.traces, None) is not None:
        raise ValueError("unused calibration traces")
    return answer


def mean_nll(records, predictions, calibration):
    if not records:
        raise ValueError("calibration group is empty")
    losses = []
    for record in records:
        answer = replay(record, predictions[record["id"]], calibration)
        p = probabilities(record, answer)
        losses.append(-math.log(max(p[str(record["label"])], 1e-12)))
    return sum(losses) / len(losses)


def fit(records, predictions):
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("duplicate calibration IDs")
    if any(row.get("split") != "validation" for row in records):
        raise ValueError("fit accepts validation records only, never test")
    non_binary = [row for row in records if row["primitive"] != "noul"]
    binary = [row for row in records if row["primitive"] == "noul"]
    trials = [(mean_nll(non_binary, predictions, Calibration(temperature=t)), t) for t in TEMPERATURES]
    _, temperature = min(trials, key=lambda pair: (pair[0], abs(pair[1] - 1)))
    binary_trials = []
    for effective_t in TEMPERATURES:
        for bias in BIASES:
            calibration = Calibration(temperature, effective_t / temperature, bias)
            binary_trials.append((mean_nll(binary, predictions, calibration), calibration))
    loss, selected = min(binary_trials, key=lambda pair: (pair[0], abs(pair[1].noul_bias), abs(pair[1].noul_temperature - 1)))
    return {"parameters": asdict(selected), "objective": "hard-label NLL on validation records only",
            "fit_records": len(records), "choice_score_records": len(non_binary), "noul_records": len(binary),
            "grid": {"temperatures": TEMPERATURES, "noul_biases": BIASES},
            "before": {"choice_score_nll": mean_nll(non_binary, predictions, Calibration()),
                       "noul_nll": mean_nll(binary, predictions, Calibration())},
            "after": {"choice_score_nll": min(trials)[0], "noul_nll": loss},
            "limitations": "Global calibration only; validation fit is not held-out test performance. No weight changes."}
