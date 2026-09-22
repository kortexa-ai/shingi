"""Freeze a fresh source-stratified release test, disjoint from all prior splits."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random

from prepare_benchmark import SOURCES, REPO, REVISION, decode_row, digest, read_jsonl, write_jsonl
from shingi.release import sha256, ADAPTER_SHA256

SEED = "shingi-cuda-v01-release-20260922"


def select_fresh(pool, count, excluded_ids, excluded_states):
    selected = []
    for row in sorted(pool, key=lambda r: digest(SEED + "|test|" + r["id"])):
        if row["id"] in excluded_ids or row["state_sha256"] in excluded_states:
            continue
        selected.append(row)
        excluded_ids.add(row["id"])
        excluded_states.add(row["state_sha256"])
        if len(selected) == count:
            return selected
    raise ValueError(f"need {count} fresh records, found {len(selected)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--training-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    old, exclusions = [], {}
    for directory, splits in ((a.baseline, ("test", "calibration")),
                              (a.training_data, ("train", "dev", "calibration", "test"))):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["revision"] != REVISION:
            raise ValueError("prior data revision differs")
        for split in splits:
            path = directory / (split + ".jsonl")
            if sha256(path) != manifest[split + "_sha256"]:
                raise ValueError("prior split changed: " + str(path))
            old.extend(read_jsonl(path))
            exclusions[directory.name + "/" + path.name] = sha256(path)
    ids, states = {r["id"] for r in old}, {r["state_sha256"] for r in old}
    excluded_counts = {"ids": len(ids), "states": len(states)}
    prior_ids, prior_states = ids.copy(), states.copy()
    baseline_manifest = json.loads((a.baseline / "manifest.json").read_text())
    test, files = [], {}
    for source in SOURCES:
        relative = f"data/{source}/test.jsonl"
        path = a.baseline / "raw" / relative
        if sha256(path) != baseline_manifest["files"][relative]["sha256"]:
            raise ValueError("pinned public data changed: " + relative)
        files[relative] = baseline_manifest["files"][relative]
        pool = [decode_row(r) for r in read_jsonl(path)]
        test.extend(select_fresh(pool, 100, ids, states))
    shuffled = []
    for row in sorted(test, key=lambda r: digest(SEED + "|shuffle|" + r["id"])):
        if row["primitive"] != "choice":
            continue
        row = copy.deepcopy(row)
        options = list(row["question"]["criteria"].items())
        original = list(options)
        random.Random(SEED + "|order|" + row["id"]).shuffle(options)
        if options == original:
            options = options[1:] + options[:1]
        row["question"]["criteria"] = dict(options)
        shuffled.append(row)
        if len(shuffled) == 200:
            break
    if len(shuffled) != 200:
        raise ValueError("insufficient order pairs")
    overlap = {"id": len(prior_ids & {r["id"] for r in test}),
               "state_sha256": len(prior_states & {r["state_sha256"] for r in test})}
    if any(overlap.values()):
        raise ValueError("release data overlaps prior splits")
    a.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(a.output / "test.jsonl", test)
    write_jsonl(a.output / "shuffle.jsonl", shuffled)
    manifest = {"dataset": REPO, "revision": REVISION, "seed": SEED,
                "counts": {"test": len(test), "shuffle": len(shuffled)},
                "by_source": dict(Counter(r["source"] for r in test)),
                "excluded_splits": exclusions, "excluded_counts": excluded_counts,
                "overlap": overlap, "files": files,
                "candidate_adapter_sha256": ADAPTER_SHA256,
                "test_sha256": sha256(a.output / "test.jsonl"),
                "shuffle_sha256": sha256(a.output / "shuffle.jsonl"),
                "records": [{k: r[k] for k in ("id", "source", "state_sha256", "input_sha256")} for r in test],
                "shuffle_ids": [r["id"] for r in shuffled],
                "pretraining_overlap": "Unknown; exclusions cover this project's adaptation and prior evaluation."}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "excluded_counts", "overlap")}, indent=2))


if __name__ == "__main__":
    main()
