"""Download pinned public tasks and freeze the evaluation inventory before inference."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from prepare_benchmark import read_jsonl, write_jsonl
from shingi.external_benchmarks import decision, digest, jev, normalized_state, spatial
from shingi.release import sha256

SPATIAL_REV = "423c49f4ad2b8d200b38e845fdf046ba6b37d46b"
DECISION_REV = "19334fec40b54b693a63e1ffd91636651d39e847"
JEV_REV = "f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84"
OMNI_REV = "c050d51354147985d13286cf4acf90f562f2c631"
SEED = "shingi-external-v1-20260923"
FILES = {}
for name, repo, rev, files in (
    ("this-that", "limberc/this-that-spatial-bench", SPATIAL_REV,
     ("README.md", "test.jsonl", "stats.json", "fingerprints.txt")),
    ("decisionbench", "akhilaaa3/decision-bench", DECISION_REV,
     ("README.md", "DATASET_INFO.md", "data/medium.jsonl", "data/hard.jsonl", "results.json")),
):
    for file in files:
        FILES[f"{name}/{file}"] = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}"
for file in ("README.md", "LICENSE", "THIRD-PARTY.md", "datasets/manifest.json",
             "datasets/public/original.jsonl", "datasets/public/easy.jsonl", "datasets/public/hard.jsonl",
             "jevbench/scoring.py", "jevbench/tasks.py"):
    FILES[f"jevbench/{file}"] = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEV_REV}/{file}"
FILES["jev-omni/README.md"] = f"https://huggingface.co/akhilaaa3/Jev-Omni/resolve/{OMNI_REV}/README.md"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior", type=Path, action="append", required=True,
                   help="Previously used normalized Shingi records; repeat for every split")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    raw = a.output / "raw"

    def download(item):
        file, url = item
        dest = raw / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        dest.write_bytes(data)
        return file, {"url": url, "sha256": sha256(dest), "bytes": len(data)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(download, FILES.items()))
    rows = [spatial(r) for r in read_jsonl(raw / "this-that/test.jsonl")]
    for split in ("medium", "hard"):
        for r in read_jsonl(raw / f"decisionbench/data/{split}.jsonl"):
            rows.extend(decision(r, split))
    for split in ("original", "easy", "hard"):
        rows.extend(jev(r, split) for r in read_jsonl(raw / f"jevbench/datasets/public/{split}.jsonl"))
    expected = {"this-that/test": 7305, "decisionbench/medium": 293, "decisionbench/hard": 293,
                "jevbench/original": 72, "jevbench/easy": 48, "jevbench/hard": 111}
    counts = dict(Counter(r["source"] for r in rows))
    if counts != expected or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"unexpected inventory: {counts}")
    # Question content and source labels stay out of the prompt except the explicitly mapped fields.
    prior = []
    for path in a.prior:
        old = read_jsonl(path)
        exact = {digest(r["state"]) for r in old}
        normalized = {digest(normalized_state(r["state"])) for r in old}
        prior.append({"path": str(path), "sha256": sha256(path), "records": len(old),
                      "exact_state_matches": [r["id"] for r in rows if r["state_sha256"] in exact],
                      "normalized_state_matches": [r["id"] for r in rows if r["normalized_state_sha256"] in normalized]})
    order_ids = []
    for suite in sorted({r["suite"] for r in rows}):
        candidates = [r for r in rows if r["suite"] == suite and r["primitive"] == "choice"]
        order_ids.extend(r["id"] for r in sorted(candidates, key=lambda r: digest([SEED, r["id"]]))[:100])
    write_jsonl(a.output / "records.jsonl", rows)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "counts": counts,
                "records": len(rows), "files": files, "records_sha256": sha256(a.output / "records.jsonl"),
                "seed": SEED, "order_ids": order_ids, "overlap": prior,
                "overlap_limits": "Checks exact JSON state and NFKC/casefold/whitespace-normalized string leaves. Does not detect semantic paraphrases, different structural wrappers, or base pretraining exposure.",
                "licenses": {"this-that": "MIT", "decisionbench": "Apache-2.0", "jevbench": "MIT"},
                "mapping": {"this-that": "Original state and question, choice criteria keyed by zero-based option indices with unchanged option text. Analytic distributions and source metadata are scoring-only.",
                            "decisionbench": "Decode state/questions/answers JSON once. One unchanged native typed question per inference; boolean gold labels map to 0/1.",
                            "jevbench": "Unchanged state and typed question. no/yes gold labels map to 0/1. Null group falls back to task ID; 195 independent public groups."},
                "scoring": "All planned questions; failures incorrect. Choice/Score argmax, Noul >=0.5. Score expected-value MAE and rounded-EV diagnostic reported separately. Equal-weight scenario/group macro plus micro; no pooling across suites.",
                "comparator": {"model": "akhilaaa3/Jev-Omni", "revision": OMNI_REV,
                               "decisionbench_medium": {"macro": .8757, "micro": .8601, "n": 293, "groups": 80},
                               "jevbench_public": {"macro": .8615, "micro": .8745, "n": 231, "groups": 195},
                               "limits": "Third-party model-card claims, not our measurements. No exact evaluation script, frozen task manifest or predictions supplied. DecisionBench score-level rule and JevBench group membership are not fully specified. Training overlap cannot be independently checked."}}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": counts, "overlap": prior, "order_pairs": len(order_ids)}, indent=2))


if __name__ == "__main__":
    main()
