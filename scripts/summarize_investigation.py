"""Create a compact, reproducible report from completed native run artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from shingi.calibration import replay
from shingi.decision import Calibration
from shingi.metrics import percentile, report, wilson


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        items = [json.loads(line) for line in stream if line.strip()]
    result = {row["id"]: row for row in items}
    if len(result) != len(items):
        raise ValueError(f"duplicate IDs: {path}")
    return result


def distribution(values):
    return {"n": len(values), "mean": sum(values) / len(values),
            "median": percentile(values, .5), "p95": percentile(values, .95)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/benchmark-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    test_path = args.benchmark / "test.jsonl"
    if sha(test_path) != manifest["test_sha256"]:
        raise ValueError("test data changed")
    records = rows(test_path)
    phases = {}
    predictions = {}
    for phase, count in (("canary", 14), ("calibration", 350), ("test", 1500), ("shuffle", 200), ("context", 12)):
        root = args.run / phase
        run = json.loads((root / "run.json").read_text())
        pred = rows(root / "predictions.jsonl")
        if run["aborted"] or run["completed_records"] != count or len(pred) != count:
            raise ValueError(f"incomplete phase: {phase}")
        if run["dataset_manifest_sha256"] != sha(manifest_path):
            raise ValueError(f"different manifest: {phase}")
        if any(row["error"] for row in pred.values()):
            raise ValueError(f"failed prediction: {phase}")
        phases[phase] = {"run": run, "predictions_sha256": sha(root / "predictions.jsonl"),
                         "metrics": json.loads((root / "metrics.json").read_text())}
        predictions[phase] = pred
    for key in ("code_revision", "model_sha256", "executable_sha256", "context_tokens"):
        if len({value["run"][key] for value in phases.values()}) != 1:
            raise ValueError(f"phase provenance differs: {key}")
    if set(records) != set(predictions["test"]):
        raise ValueError("test ID inventory differs")
    raw = {}
    for key, record in records.items():
        measured = predictions["test"][key]
        if measured["input_sha256"] != record["input_sha256"]:
            raise ValueError("test input hash differs")
        raw[key] = {"answer": replay(record, measured, Calibration())}
    raw_metrics = report(list(records.values()), raw)
    if set(predictions["shuffle"]) != set(manifest["shuffle_ids"]):
        raise ValueError("shuffle ID inventory differs")
    pairs = []
    for key, shuffled in predictions["shuffle"].items():
        original = predictions["test"][key]
        if original["input_sha256"] != shuffled["input_sha256"] or original["wire_input_sha256"] == shuffled["wire_input_sha256"]:
            raise ValueError("shuffle input identity/order check failed")
        a, b = original["answer"], shuffled["answer"]
        if set(a["probabilities"]) != set(b["probabilities"]):
            raise ValueError("shuffle label set differs")
        pairs.append({"id": key, "options": len(a["probabilities"]),
                      "flipped": a["choice"] != b["choice"],
                      "original_correct": a["choice"] == str(records[key]["label"]),
                      "shuffled_correct": b["choice"] == str(records[key]["label"]),
                      "tvd": .5 * sum(abs(a["probabilities"][label] - b["probabilities"][label]) for label in a["probabilities"])})
    order = {}
    for name, group in (("all", pairs), ("at_most_52", [p for p in pairs if p["options"] <= 52]),
                        ("over_52", [p for p in pairs if p["options"] > 52])):
        n = len(group)
        if not n:
            continue
        flips = sum(p["flipped"] for p in group)
        order[name] = {"n": n, "flips": flips, "flip_rate": flips / n,
                       "flip_rate_wilson_95": wilson(flips, n),
                       "original_correct": sum(p["original_correct"] for p in group),
                       "shuffled_correct": sum(p["shuffled_correct"] for p in group),
                       "label_aligned_tvd": distribution([p["tvd"] for p in group])}
    context = [{key: row[key] for key in ("id", "target_tokens", "input_tokens", "needle_position", "label", "latency_ms", "gpu_free_after_mib")} |
               {"choice": row["answer"]["choice"], "correct": row["answer"]["choice"] == row["label"]}
               for row in predictions["context"].values()]
    memory = {name: {"minimum_sampled_free_mib": min(row["gpu_free_after_mib"] for row in pred.values()),
                     "maximum_sampled_allocation_delta_mib": phases[name]["run"]["gpu_free_before_mib"] - min(row["gpu_free_after_mib"] for row in pred.values())}
              for name, pred in predictions.items()}
    native_latency = distribution([sum(trace["prefill_ms"] for trace in row["traces"]) for row in predictions["test"].values()])
    result = {"report_code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "data_manifest_sha256": sha(manifest_path), "calibration": json.loads((args.run / "calibration.json").read_text()),
              "phases": phases, "uncalibrated_test_replay": raw_metrics, "option_order": order,
              "context_probes": context, "sampled_memory": memory, "test_native_prefill_ms": native_latency,
              "notes": ["No weight training. Raw test probabilities replay the same saved logits without fitted calibration.",
                        "End-to-end evaluation latency includes Python and per-readout GPU memory safety queries; it is not HTTP service throughput.",
                        "Memory observations are checkpoints, not a continuous peak measurement.",
                        "Synthetic context probes test fact retrieval, not general long-document comprehension.",
                        "No hosted Jev requests. Third-party public prediction reanalysis is stored separately."]}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"test_correct": phases["test"]["metrics"]["overall"]["correct"],
                      "raw_correct": raw_metrics["overall"]["correct"], "option_order": order,
                      "context_correct": sum(row["correct"] for row in context), "sampled_memory": memory}, indent=2))


if __name__ == "__main__":
    main()
