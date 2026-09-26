"""Paired item-level comparison of two external runs on the frozen external inventory.

Scores both runs with compare_stage1.correct and reports four-way splits, paired
bootstrap intervals, the probabilities behind the flips, option-count and
state-length effects, group consistency and order sensitivity. No inference is
run. The committed output holds IDs, counts and numbers only. Full-text flip
dumps go to --dump-dir, which must be an ignored artifact directory.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from compare_stage1 import correct, paired_interval
from prepare_benchmark import read_jsonl
from shingi.external_benchmarks import canonical
from shingi.metrics import expected_labels, percentile, probabilities
from shingi.release import sha256

SUITES = ("decisionbench", "jevbench", "this-that")
DUMP_SUITES = ("decisionbench", "jevbench")
OPTION_BUCKETS = ((2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 24, "9-24"), (25, 52, "25-52"), (53, None, ">52"))
STATE_BUCKETS = ((0, 500, "<500"), (500, 2000, "500-2k"), (2000, 8000, "2k-8k"), (8000, None, ">8k"))
NOTES = {
    "accuracy": "Accuracy is invariant to temperature-only calibration: temperatures keep the argmax, and with a zero "
                "yes/no bias the P(yes) >= 0.5 threshold keeps its side. Both runs here have a zero bias.",
    "probabilities": "Probabilities are as recorded: the old run with its released calibration, the new run with the "
                     "calibration in its receipt (identity for the stage 1 candidate).",
    "state_chars": "Characters of a string state, or of its canonical JSON when the state is structured.",
    "options": "Yes/no questions count as two options; Score counts its levels.",
    "intervals": "Paired new minus old accuracy with a percentile bootstrap 95% interval; 10,000 resamples for "
                 "groups of 200 or more, 2,000 otherwise. Items within a state or group are not independent."}


def option_bucket(n):
    return next(name for low, high, name in OPTION_BUCKETS if n >= low and (high is None or n <= high))


def state_bucket(chars):
    return next(name for low, high, name in STATE_BUCKETS if chars >= low and (high is None or chars < high))


def state_chars(state):
    return len(state) if isinstance(state, str) else len(canonical(state))


def chosen(record, answer):
    p = probabilities(record, answer)
    if record["primitive"] == "noul":
        return "1" if p["1"] >= .5 else "0"
    if record["primitive"] == "choice":
        return answer["choice"]
    return max(p, key=p.__getitem__)


def load_run(run, rows, suite):
    """Predictions for one suite; the ID inventory and input hashes must match the frozen records exactly."""
    predictions = read_jsonl(run / f"{suite}.jsonl")
    by_id = {p["id"]: p for p in predictions}
    expected = {r["id"]: r for r in rows}
    if len(by_id) != len(predictions) or set(by_id) != set(expected):
        raise SystemExit(f"{run}/{suite}.jsonl: prediction IDs differ from the frozen inventory")
    if any(p["input_sha256"] != expected[i]["input_sha256"] for i, p in by_id.items()):
        raise SystemExit(f"{run}/{suite}.jsonl: input hash differs from the frozen inventory")
    return by_id


def item(record, old, new):
    """Correctness and recorded probabilities of one frozen record under both runs."""
    row = {"id": record["id"], "old": correct(record, old), "new": correct(record, new),
           "n_options": len(expected_labels(record)), "state_chars": state_chars(record["state"])}
    for name, prediction in (("old", old), ("new", new)):
        if prediction.get("error"):
            row[name + "_prob_gold"] = row[name + "_prob_chosen"] = None
            continue
        p = probabilities(record, prediction["answer"])
        row[name + "_prob_gold"] = p[str(record["label"])]
        row[name + "_prob_chosen"] = p[chosen(record, prediction["answer"])]
    return row


def tally(items):
    n = len(items)
    old = sum(i["old"] for i in items)
    new = sum(i["new"] for i in items)
    both = sum(i["old"] and i["new"] for i in items)
    differences = [int(i["new"]) - int(i["old"]) for i in items]
    return {"n": n, "old_correct": old, "new_correct": new, "both_correct": both,
            "old_only": old - both, "new_only": new - both, "both_wrong": n - old - new + both,
            "old_accuracy": old / n, "new_accuracy": new / n,
            "difference": paired_interval(differences, 10000 if n >= 200 else 2000)}


def grouped(items, key, order=None):
    groups = defaultdict(list)
    for i in items:
        groups[key(i)].append(i)
    keys = [k for k in order if k in groups] if order else sorted(groups, key=str)
    return {str(k): tally(groups[k]) for k in keys}


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": sum(values) / len(values), "q1": percentile(values, .25),
            "median": percentile(values, .5), "q3": percentile(values, .75)}


def confidence(items):
    old_only = [i for i in items if i["old"] and not i["new"]]
    new_only = [i for i in items if i["new"] and not i["old"]]
    both = [i for i in items if i["old"] and i["new"]]
    return {"old_only": {"n": len(old_only), "new_prob_chosen": spread([i["new_prob_chosen"] for i in old_only]),
                         "new_prob_gold": spread([i["new_prob_gold"] for i in old_only]),
                         "old_prob_gold": spread([i["old_prob_gold"] for i in old_only])},
            "new_only": {"n": len(new_only), "old_prob_chosen": spread([i["old_prob_chosen"] for i in new_only]),
                         "old_prob_gold": spread([i["old_prob_gold"] for i in new_only]),
                         "new_prob_gold": spread([i["new_prob_gold"] for i in new_only])},
            "both_correct": {"n": len(both), "gold_probability_change": spread(
                [i["new_prob_gold"] - i["old_prob_gold"] for i in both])}}


def consistency(items, group_of):
    """Groups of two or more items: the share on which each model answers every item correctly."""
    groups = defaultdict(list)
    for i in items:
        groups[group_of[i["id"]]].append(i)
    groups = [g for g in groups.values() if len(g) >= 2]
    old = [all(i["old"] for i in g) for g in groups]
    new = [all(i["new"] for i in g) for g in groups]
    n = len(groups)
    return {"groups": n, "items": sum(map(len, groups)), "old_all_correct": sum(old), "new_all_correct": sum(new),
            "old_share": sum(old) / n if n else None, "new_share": sum(new) / n if n else None,
            "both": sum(a and b for a, b in zip(old, new)), "old_only": sum(a and not b for a, b in zip(old, new)),
            "new_only": sum(b and not a for a, b in zip(old, new))}


def breakdown(items, records):
    result = {"all": tally(items), "by_primitive": grouped(items, lambda i: records[i["id"]]["primitive"])}
    if any(records[i["id"]].get("family") is not None for i in items):
        result["by_family"] = grouped(items, lambda i: records[i["id"]]["family"])
    return result


def analyse(suite, rows, old, new):
    records = {r["id"]: r for r in rows}
    items = [item(r, old[r["id"]], new[r["id"]]) for r in rows]
    result = breakdown(items, records)
    result["confidence"] = confidence(items)
    subsets = sorted({r["subset"] for r in rows})
    if len(subsets) > 1:
        result["subsets"] = {}
        for subset in subsets:
            selected = [i for i in items if records[i["id"]]["subset"] == subset]
            result["subsets"][subset] = {**breakdown(selected, records), "confidence": confidence(selected)}
    if suite in DUMP_SUITES:
        group_of = {r["id"]: r["group"] for r in rows}
        structure = {}
        for scope in ["all"] + (subsets if len(subsets) > 1 else []):
            selected = [i for i in items if scope == "all" or records[i["id"]]["subset"] == scope]
            structure[scope] = {"by_options": grouped(selected, lambda i: option_bucket(i["n_options"]),
                                                      [b[2] for b in OPTION_BUCKETS]),
                                "by_state_chars": grouped(selected, lambda i: state_bucket(i["state_chars"]),
                                                          [b[2] for b in STATE_BUCKETS]),
                                "group_consistency": consistency(selected, group_of)}
        result["structure"] = structure
    if suite == "this-that":
        families = result["by_family"].values()
        result["families_beyond_5_points"] = {"improved": sum(f["difference"]["mean"] > .05 for f in families),
                                              "regressed": sum(f["difference"]["mean"] < -.05 for f in families),
                                              "total": len(result["by_family"])}
    return result, items


def dump(path, rows, old, new, items):
    """Private full-text rows for every item whose correctness differs; never commit these."""
    by_id = {i["id"]: i for i in items}
    with path.open("w") as stream:
        for r in rows:
            i = by_id[r["id"]]
            if i["old"] == i["new"]:
                continue
            stream.write(json.dumps({
                "id": r["id"], "subset": r["subset"], "family": r.get("family"), "group": r.get("group"),
                "primitive": r["primitive"], "direction": "old_only" if i["old"] else "new_only", "label": r["label"],
                "state": r["state"], "question": r["question"], "old_answer": old[r["id"]].get("answer"),
                "new_answer": new[r["id"]].get("answer"), "old_prob_gold": i["old_prob_gold"],
                "new_prob_gold": i["new_prob_gold"], "n_options": i["n_options"], "state_chars": i["state_chars"]},
                ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True, help="external run directory of the reference model")
    p.add_argument("--new", type=Path, required=True, help="external run directory of the candidate")
    p.add_argument("--old-name", default="v0.2")
    p.add_argument("--new-name", default="stage1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dump-dir", type=Path, required=True, help="ignored directory for private full-text flip rows")
    a = p.parse_args()
    rows = read_jsonl(a.records)
    if len({r["id"] for r in rows}) != len(rows) or {r["suite"] for r in rows} != set(SUITES):
        raise SystemExit("unexpected frozen inventory")
    summaries = {name: json.loads((run / "summary.json").read_text()) for name, run in (("old", a.old), ("new", a.new))}
    result = {"old": a.old_name, "new": a.new_name, "notes": NOTES,
              "inputs": {"records_sha256": sha256(a.records),
                         **{f"{name}_{key}": summaries[name]["receipt"][key]
                            for name in ("old", "new") for key in ("calibration", "calibration_sha256", "identity")}},
              "suites": {}}
    a.dump_dir.mkdir(parents=True, exist_ok=True)
    for suite in SUITES:
        selected = [r for r in rows if r["suite"] == suite]
        old, new = load_run(a.old, selected, suite), load_run(a.new, selected, suite)
        result["suites"][suite], items = analyse(suite, selected, old, new)
        for name, key in (("old", "old_correct"), ("new", "new_correct")):
            if result["suites"][suite]["all"][key] != summaries[name]["benchmarks"][suite]["all"]["correct"]:
                raise SystemExit(f"{suite}: rescored {name} accuracy differs from its run summary")
        if suite in DUMP_SUITES:
            dump(a.dump_dir / f"{suite}-flips.jsonl", selected, old, new, items)
        print(json.dumps({"suite": suite, **{k: result["suites"][suite]["all"][k]
                                              for k in ("n", "old_correct", "new_correct", "old_only", "new_only")}}))
    result["order"] = {suite: {a.old_name: summaries["old"]["order"].get(suite), a.new_name: summaries["new"]["order"].get(suite)}
                       for suite in SUITES}
    (a.dump_dir / "README.txt").write_text(
        "Private flip dumps from scripts/flip_analysis.py. They contain full evaluation states, questions and\n"
        "answers from third-party benchmarks. Do not commit, publish or redistribute them.\n")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
