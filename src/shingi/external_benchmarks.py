"""Frozen external benchmark mappings and scoring; no model selection."""
from collections import defaultdict
import hashlib
import json
import math
import unicodedata

from .metrics import expected_labels, probabilities, summarize


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalized_state(value):
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if isinstance(value, list):
        return [normalized_state(x) for x in value]
    if isinstance(value, dict):
        return {k: normalized_state(v) for k, v in value.items()}
    return value


def record(identifier, suite, subset, family, group, state, question, label, **metadata):
    row = dict(id=identifier, suite=suite, subset=subset, source=f"{suite}/{subset}",
               family=family, group=group, state=state, question=question,
               primitive=question["type"], label=str(label), **metadata)
    if row["label"] not in expected_labels(row):
        raise ValueError(f"gold label not in options: {identifier}")
    row["input_sha256"] = digest([state, question])
    row["state_sha256"] = digest(state)
    row["normalized_state_sha256"] = digest(normalized_state(state))
    return row


def spatial(row):
    options = row["options"]
    distribution = row["answer_distribution"]
    if (len(options) != len(distribution) or len(set(options)) != len(options)
            or row["answer"] != options[row["answer_index"]]
            or any(not math.isfinite(x) or x < 0 or x > 1 for x in distribution)
            or abs(sum(distribution) - 1) > 1e-8):
        raise ValueError("invalid spatial options or analytic distribution")
    return record("this-that/" + row["id"], "this-that", "test", row["family"],
                  digest(row["state"]), row["state"],
                  {"type": "choice", "instructions": row["question"],
                   "criteria": {str(i): option for i, option in enumerate(options)}},
                  row["answer_index"], analytic_distribution=distribution,
                  deterministic=row["deterministic"],
                  authors_seen_in_training=row["seen_in_training"],
                  upstream_fingerprint=row["fingerprint"])


def decision(row, subset):
    state, questions, answers = [json.loads(row[k]) for k in ("state", "questions", "answers")]
    if set(questions) != set(answers) or len(questions) != row["n_questions"]:
        raise ValueError("DecisionBench question/answer inventory mismatch")
    out = []
    for key, q in questions.items():
        label = answers[key]
        if q["type"] == "noul":
            if not isinstance(label, bool):
                raise ValueError("DecisionBench Noul label is not boolean")
            label = int(label)
        out.append(record(f"decisionbench/{subset}/{row['id']}/{key}", "decisionbench",
                          subset, row["family"], row["id"], state, q, label))
    return out


def jev(row, subset):
    q = row["question"]
    label = row["expected"]
    if row.get("provenance", {}).get("exclude_reason") or label is None:
        raise ValueError("unexpected unscored public JevBench task")
    if q["type"] == "noul":
        label = {"no": "0", "yes": "1"}[label]
    # Null means unpaired, not one group containing all easy decisions.
    result = record("jevbench/" + row["id"], "jevbench", subset, row["family"],
                    row.get("group") or row["id"], row["state"], q, label)
    labels = ["no", "yes"] if q["type"] == "noul" else expected_labels(result)
    if set(labels) != set(row["labels"]):
        raise ValueError("JevBench candidate inventory mismatch")
    return result


def analytic_metrics(rows, predictions):
    values = []
    for row in rows:
        pred = predictions.get(row["id"])
        if not pred or pred.get("error"):
            continue
        p = probabilities(row, pred["answer"])
        q = {str(i): x for i, x in enumerate(row["analytic_distribution"])}
        chosen = pred["answer"]["choice"]
        ce = -sum(q[k] * math.log(max(p[k], 1e-12)) for k in p)
        entropy = -sum(x * math.log(x) for x in q.values() if x)
        values.append({"expected_accuracy": q[chosen], "bayes_accuracy_ceiling": max(q.values()),
                       "cross_entropy": ce, "kl_divergence": ce - entropy,
                       "tvd": .5 * sum(abs(p[k] - q[k]) for k in p),
                       "squared_distribution_error": sum((p[k] - q[k]) ** 2 for k in p),
                       "expected_brier": sum(p[k] ** 2 - 2*p[k]*q[k] + q[k] for k in p)})
    means = {k: sum(x[k] for x in values) / len(values) for k in values[0]} if values else {}
    return {"planned": len(rows), "valid": len(values), **means}


def grouped(rows, predictions, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {k: summarize(v, predictions) for k, v in sorted(groups.items())}


def suite_report(rows, predictions):
    result = summarize(rows, predictions)
    groups = grouped(rows, predictions, "group")
    result["groups"] = len(groups)
    result["group_macro_failures_incorrect"] = sum(x["accuracy_failures_incorrect"] for x in groups.values()) / len(groups)
    observed = [x["correct"] / x["valid"] for x in groups.values() if x["valid"]]
    result["group_macro_observed_only"] = sum(observed) / len(observed) if observed else None
    result["by_family"] = grouped(rows, predictions, "family")
    result["by_primitive"] = grouped(rows, predictions, "primitive")
    # Extra ordinal readout diagnostic, not a selected scoring rule.
    rounded = []
    ties = 0
    for row in rows:
        pred = predictions.get(row["id"])
        if not pred or pred.get("error"):
            continue
        p = probabilities(row, pred["answer"])
        ties += sum(x == max(p.values()) for x in p.values()) > 1
        if row["primitive"] == "score":
            ev = sum(int(k)*v for k, v in p.items())
            rounded.append(math.floor(ev + .5) == int(row["label"]))
    result["exact_top_probability_ties"] = ties
    result["score_rounded_ev_accuracy_diagnostic"] = {"valid": len(rounded), "correct": sum(rounded),
        "accuracy": sum(rounded)/len(rounded) if rounded else None, "rounding": "floor(E[index]+0.5)"}
    if rows[0]["suite"] == "this-that":
        result["analytic"] = {name: analytic_metrics(selected, predictions) for name, selected in (
            ("all", rows), ("deterministic", [r for r in rows if r["deterministic"]]),
            ("stochastic", [r for r in rows if not r["deterministic"]]))}
        result["by_authors_training_family_exposure"] = grouped(rows, predictions, "authors_seen_in_training")
        result["by_deterministic"] = grouped(rows, predictions, "deterministic")
    return result
