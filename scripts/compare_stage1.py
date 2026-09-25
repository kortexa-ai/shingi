"""Paired stage 1 comparison against v0.2 on matched frozen v3 splits, with the program #9 gates.

Reads two evaluate_records.py outputs per split, scores every record with the
same rules as shingi.metrics.summarize, and reports paired bootstrap intervals
overall and per source. No inference is run.
"""
import argparse
import json
from pathlib import Path
import random

from evaluate_records import load_split
from shingi.metrics import probabilities

GATES = {"test_accuracy_gain": .03, "ood_lower_bound": -.015}


def correct(record, prediction):
    if prediction.get("error"):
        return False
    answer = prediction["answer"]
    p = probabilities(record, answer)
    label = str(record["label"])
    if record["primitive"] == "noul":
        return ("1" if p["1"] >= .5 else "0") == label
    if record["primitive"] == "choice":
        return answer["choice"] == label
    return max(p, key=p.__getitem__) == label


def paired_interval(differences, resamples=10000, seed=20260926):
    """Mean paired difference with a percentile bootstrap 95% interval."""
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {"n": n, "mean": sum(differences) / n,
            "low": means[int(.025 * resamples)], "high": means[int(.975 * resamples) - 1]}


def compare(records, new, old, resamples=10000):
    by_source, overall = {}, []
    for record in records:
        d = int(correct(record, new[record["id"]])) - int(correct(record, old[record["id"]]))
        overall.append(d)
        by_source.setdefault(record["source"], []).append(d)
    return {"overall": paired_interval(overall, resamples),
            "by_source": {s: paired_interval(v, max(1000, resamples // 5)) for s, v in sorted(by_source.items())}}


def predictions(path):
    with (path / "predictions.jsonl").open() as stream:
        return {row["id"]: row for row in map(json.loads, stream) if row}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True, help="directory holding <model>-<split> evaluations")
    p.add_argument("--new", default="full-2048")
    p.add_argument("--old", default="v0.2")
    p.add_argument("--splits", nargs="+", default=["test", "ood", "transfer"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summary = {"new": a.new, "old": a.old, "splits": {}}
    for split in a.splits:
        records, manifest_sha = load_split(a.data, split)
        runs = {name: a.eval / f"{name}-{split}" for name in (a.new, a.old)}
        for name, run in runs.items():
            receipt = json.loads((run / "receipt.json").read_text())
            if receipt["data_manifest_sha256"] != manifest_sha or receipt["split"] != split:
                raise SystemExit(f"{run}: evaluated a different frozen split")
        new, old = predictions(runs[a.new]), predictions(runs[a.old])
        if set(new) != {r["id"] for r in records} or set(old) != set(new):
            raise SystemExit(f"{split}: prediction inventories differ")
        metrics = {name: json.loads((run / "metrics.json").read_text()) for name, run in runs.items()}
        summary["splits"][split] = {
            "paired_accuracy_difference": compare(records, new, old),
            "overall": {name: m["overall"] for name, m in metrics.items()},
            "adapter_sha256": {name: json.loads((run / "receipt.json").read_text())["adapter_sha256"]
                               for name, run in runs.items()}}
    s = summary["splits"]
    gates = {}
    if "test" in s:
        t = s["test"]["paired_accuracy_difference"]["overall"]
        n, o = s["test"]["overall"][a.new], s["test"]["overall"][a.old]
        gates["test_accuracy_gain"] = {"passed": t["mean"] >= GATES["test_accuracy_gain"] and t["low"] > 0, **t}
        gates["test_calibrated_nll_not_worse"] = {"passed": n["nll"]["mean"] <= o["nll"]["mean"],
                                                  "new": n["nll"]["mean"], "old": o["nll"]["mean"]}
        gates["test_ece_not_worse"] = {"passed": n["ece_top_label"] <= o["ece_top_label"],
                                       "new": n["ece_top_label"], "old": o["ece_top_label"]}
        gates["test_source_regressions"] = [src for src, v in s["test"]["paired_accuracy_difference"]["by_source"].items()
                                            if v["high"] < 0]
    if "ood" in s:
        o = s["ood"]["paired_accuracy_difference"]["overall"]
        gates["ood_lower_bound"] = {"passed": o["low"] >= GATES["ood_lower_bound"], **o}
        gates["ood_source_regressions"] = [src for src, v in s["ood"]["paired_accuracy_difference"]["by_source"].items()
                                           if v["high"] < 0]
    summary["gates"] = gates
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
