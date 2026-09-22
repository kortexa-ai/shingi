"""Decision metrics with explicit denominators and no hidden dropped failures."""
from collections import defaultdict
import math


def expected_labels(record):
    q = record["question"]
    if q["type"] == "noul":
        return ["0", "1"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def probabilities(record, answer, *, allow_rounding=False):
    labels = expected_labels(record)
    if record["primitive"] == "noul":
        yes = answer["noul"]
        p = {"0": 1 - yes, "1": yes}
    else:
        p = answer["probabilities"]
    if set(p) != set(labels) or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1 for v in p.values()):
        raise ValueError("invalid probability keys or values")
    total = sum(p.values())
    if total <= 0 or (not allow_rounding and abs(total - 1) > 1e-6):
        raise ValueError("probabilities do not sum to one")
    return {key: p[key] / total for key in labels}


def human_distribution(record):
    soft = record.get("soft_label")
    if soft is None:
        return None
    if isinstance(soft, (int, float)):
        values = {"0": 1 - soft, "1": soft}
    elif isinstance(soft, list):
        values = {str(i): value for i, value in enumerate(soft)}
    else:
        values = soft
    labels = expected_labels(record)
    if set(values) - set(labels):
        raise ValueError("unexpected human-label key")
    p = {key: values.get(key, 0.0) for key in labels}
    if any(not math.isfinite(x) or x < 0 for x in p.values()) or sum(p.values()) <= 0:
        raise ValueError("invalid human distribution")
    total = sum(p.values())
    return {key: value / total for key, value in p.items()}


def wilson(correct, count):
    if not count:
        return None
    z = 1.959963984540054
    rate = correct / count
    scale = 1 + z * z / count
    mid = (rate + z * z / (2 * count)) / scale
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / scale
    return [mid - half, mid + half]


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def summarize(records, predictions, *, allow_rounding=False):
    observations, failures = [], []
    for record in records:
        prediction = predictions.get(record["id"])
        try:
            if prediction is None or prediction.get("error"):
                raise ValueError("missing or failed prediction")
            answer = prediction["answer"]
            p = probabilities(record, answer, allow_rounding=allow_rounding)
            label = str(record["label"])
            if label not in p:
                raise ValueError("gold label is not an option")
            argmax = max(p, key=p.__getitem__)
            if record["primitive"] == "noul":
                argmax = "1" if p["1"] >= .5 else "0"
            if record["primitive"] == "choice":
                chosen = answer["choice"]
                # A rounded tie may preserve the pre-rounding winner.
                if chosen not in p or p[chosen] != max(p.values()):
                    raise ValueError("choice does not have maximal probability")
                argmax = chosen
            row = {"correct": argmax == label, "top_probability": max(p.values()),
                   "nll": -math.log(max(p[label], 1e-12)),
                   "brier_multiclass": sum((value - (key == label)) ** 2 for key, value in p.items()),
                   "latency_ms": prediction.get("latency_ms")}
            if record["primitive"] == "score":
                expected = sum(int(key) * value for key, value in p.items())
                row["score_mae"] = abs(expected - float(label))
                cumulative = 0.0
                distances = []
                for i in range(len(p) - 1):
                    cumulative += p[str(i)]
                    distances.append((cumulative - (int(label) <= i)) ** 2)
                row["rps"] = sum(distances) / len(distances)
            if record["primitive"] == "noul":
                row["brier_binary"] = (p["1"] - int(label)) ** 2
            human = human_distribution(record)
            if human is not None:
                row["human_tvd"] = .5 * sum(abs(p[key] - human[key]) for key in p)
            observations.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append({"id": record["id"], "reason": str(exc)})
    n = len(records)
    correct = sum(row["correct"] for row in observations)
    result = {"n": n, "valid": len(observations), "failures": failures,
              "correct": correct, "accuracy_failures_incorrect": correct / n if n else None,
              "accuracy_wilson_95": wilson(correct, n)}
    for metric in ("nll", "brier_multiclass", "brier_binary", "score_mae", "rps", "human_tvd"):
        values = [row[metric] for row in observations if metric in row]
        result[metric] = {"mean": sum(values) / len(values) if values else None, "n": len(values)}
    bins = []
    for i in range(10):
        group = [row for row in observations if min(9, int(row["top_probability"] * 10)) == i]
        bins.append({"lower": i / 10, "upper": (i + 1) / 10, "n": len(group),
                     "accuracy": sum(row["correct"] for row in group) / len(group) if group else None,
                     "probability": sum(row["top_probability"] for row in group) / len(group) if group else None})
    result["ece_top_label"] = sum(b["n"] * abs(b["accuracy"] - b["probability"]) for b in bins if b["n"]) / len(observations) if observations else None
    result["reliability_bins"] = bins
    latency = [row["latency_ms"] for row in observations if row["latency_ms"] is not None]
    result["latency_ms"] = {"n": len(latency), "median": percentile(latency, .5), "p95": percentile(latency, .95)}
    return result


def report(records, predictions, **kwargs):
    groups = defaultdict(list)
    for row in records:
        groups[row["source"]].append(row)
    return {"overall": summarize(records, predictions, **kwargs),
            "by_source": {key: summarize(rows, predictions, **kwargs) for key, rows in groups.items()},
            "metric_notes": {"failures": "Included as incorrect in accuracy; excluded from probability metrics with explicit n.",
                             "ece": "10 equal-width bins of maximum class probability, not API confidence.",
                             "nll": "Natural log; probabilities floored at 1e-12 for this metric only.",
                             "brier": "Multiclass sum of squared errors; binary Brier also reported for Noul.",
                             "noul_threshold": "Yes when P(yes) >= 0.5, including an exact tie.",
                             "ordinal": "Zero-based expected score MAE and normalized ranked probability score.",
                             "human": "TVD to normalized human vote shares, where provided."}}
