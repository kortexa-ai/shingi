"""Freeze a deterministic, source-stratified JevBench slice before inference.

Downloads public, revision-pinned data only. No model or GPU is used.
Raw text and labels stay in ignored artifacts; the manifest contains only IDs,
hashes, counts, and provenance. Dataset licenses remain upstream's.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "Praveenrajus/jev-bench"
REVISION = "002ad22de8db2df5e0eb898b3da8072dbd4af4de"
SOURCES = ("banking77", "clinc150", "ledgar", "go_emotions", "mmlu", "arc_challenge", "mnli",
           "sst5", "helpsteer2_helpfulness", "measuring_hate_speech", "boolq", "fever_evidence",
           "civil_comments", "sms_spam", "chaosnli")
SEED = "shingi-baseline-v1-20260922"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def decode_row(row):
    parsed = dict(row)
    for key in ("state", "question", "soft_label", "meta"):
        if isinstance(parsed.get(key), str):
            parsed[key] = json.loads(parsed[key])
    parsed["input_sha256"] = digest(canonical([parsed["state"], parsed["question"]]))
    parsed["state_sha256"] = digest(canonical(parsed["state"]))
    return parsed


def ordered(rows):
    return sorted(rows, key=lambda row: digest(SEED + "|" + row["id"]))


def select_unique(rows, count, excluded_ids=None, excluded_states=None):
    ids, states = set(excluded_ids or ()), set(excluded_states or ())
    chosen, rejected = [], []
    for row in ordered(rows):
        if row["id"] in ids or row["state_sha256"] in states:
            rejected.append(row["id"])
            continue
        chosen.append(row)
        ids.add(row["id"])
        states.add(row["state_sha256"])
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"only {len(chosen)} distinct records, need {count}")
    return chosen, rejected


def write_jsonl(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    # str.splitlines() also splits Unicode separators inside JSON strings.
    # JSONL boundaries are actual newline bytes, not every Unicode line break.
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark-v1"))
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise SystemExit("manifest already exists; preserve it and use a new output directory")
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = ["manifest.json", "results/jev-1.13.0/test_predictions.jsonl"]
    paths += [f"data/{source}/test.jsonl" for source in SOURCES]
    paths += [f"data/{source}/validation.jsonl" for source in SOURCES if source != "chaosnli"]

    def download(path):
        dest = raw / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            with dest.open("xb") as stream:
                stream.write(data)
        data = dest.read_bytes()
        return path, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "url": url}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(download, paths))

    def rows(source, split):
        return [decode_row(row) for row in read_jsonl(raw / f"data/{source}/{split}.jsonl")]

    test, calibration, rejected = [], [], {}
    for source in SOURCES:
        chosen, duplicates = select_unique(rows(source, "test"), 100,
                                           excluded_ids={x["id"] for x in test},
                                           excluded_states={x["state_sha256"] for x in test})
        test.extend(chosen)
        rejected[source + "/test"] = duplicates
    for source in SOURCES:
        if source == "chaosnli":
            continue
        chosen, duplicates = select_unique(rows(source, "validation"), 25,
                                           excluded_ids={x["id"] for x in test + calibration},
                                           excluded_states={x["state_sha256"] for x in test + calibration})
        calibration.extend(chosen)
        rejected[source + "/calibration"] = duplicates
    assert not ({x["id"] for x in test} & {x["id"] for x in calibration})
    assert not ({x["state_sha256"] for x in test} & {x["state_sha256"] for x in calibration})
    choices = [row for row in test if row["primitive"] == "choice" and len(row["question"]["criteria"]) > 1]
    shuffle_ids = [x["id"] for x in ordered(choices)[:200]]
    baseline_rows = read_jsonl(raw / "results/jev-1.13.0/test_predictions.jsonl")
    baseline = {row["id"]: row for row in baseline_rows}
    if len(baseline) != len(baseline_rows):
        raise ValueError("duplicate hosted prediction IDs")
    missing = [x["id"] for x in test if x["id"] not in baseline]
    upstream = json.loads((raw / "manifest.json").read_text())
    manifest = {"dataset": REPO, "revision": REVISION, "seed": SEED,
                "test_count": len(test), "calibration_count": len(calibration),
                "domains": dict(Counter(x["meta"]["domain"] for x in test)),
                "sources": {source: upstream["sources"][source] for source in SOURCES},
                "files": files, "excluded_duplicates": rejected,
                "test_ids": [x["id"] for x in test], "calibration_ids": [x["id"] for x in calibration],
                "shuffle_ids": shuffle_ids, "hosted_predictions_missing": missing,
                "comparison_limit": "Saved Jev predictions carry IDs but no input hashes. Same-revision joins are provenance evidence, not independent proof of identical inference payloads.",
                "overlap": {"id": 0, "state_sha256": 0},
                "pretraining_contamination": "Unknown; held out from this investigation only."}
    write_jsonl(args.output / "test.jsonl", test)
    write_jsonl(args.output / "calibration.jsonl", calibration)
    for split in ("test", "calibration"):
        manifest[split + "_sha256"] = hashlib.sha256((args.output / (split + ".jsonl")).read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("revision", "test_count", "calibration_count", "domains", "overlap", "hosted_predictions_missing")}, indent=2))


if __name__ == "__main__":
    main()
