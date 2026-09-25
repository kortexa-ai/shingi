"""Native evaluation of one frozen record split, base or adapter, with every trace kept.

Checks the split against its data manifest before loading, then scores with the
canonical choice order used by v0.2 and data v3. Use only on an authorized GPU;
this script never manages services.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from prepare_benchmark import read_jsonl
from shingi.backend import NativeReadout
from shingi.decision import Calibration, DecisionEngine
from shingi.gpu import gpu_snapshot
from shingi.metrics import report
from shingi.release import BASE_SHA256, require_release_environment, sha256


def infer(engine, records, path):
    predictions = {}
    with path.open("x") as stream:
        for index, row in enumerate(records):
            started = time.perf_counter()
            item = {"id": row["id"], "source": row["source"], "primitive": row["primitive"],
                    "input_sha256": row["input_sha256"], "error": None}
            try:
                answer, traces = engine.answer(row["state"], row["question"])
                item.update(answer=answer, traces=traces)
            except ValueError as exc:
                item["error"] = str(exc)
            item["latency_ms"] = 1000 * (time.perf_counter() - started)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            predictions[row["id"]] = item
            if (index + 1) % 100 == 0:
                print(path.name, index + 1, flush=True)
    return predictions


def load_split(data, split):
    """Records of a frozen split after checking the manifest checksum."""
    manifest = json.loads((data / "manifest.json").read_text())
    path = data / f"{split}.jsonl"
    if f"{split}_sha256" not in manifest or not path.exists() or sha256(path) != manifest[f"{split}_sha256"]:
        raise ValueError(f"frozen split changed: {split}")
    return read_jsonl(path), sha256(data / "manifest.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--calibration", type=Path, help="frozen calibration JSON; raw logits when omitted")
    p.add_argument("--executable", type=Path, default=Path("artifacts/bin/readout"))
    p.add_argument("--data", type=Path, required=True, help="directory holding manifest.json and <split>.jsonl")
    p.add_argument("--split", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require_release_environment()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit source before evaluation")
    if sha256(a.model) != BASE_SHA256:
        raise ValueError("base checksum mismatch")
    records, manifest_sha = load_split(a.data, a.split)
    calibration = Calibration(**json.loads(a.calibration.read_text())["parameters"]) if a.calibration else Calibration()
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = {"split": a.split, "records": len(records), "data_manifest_sha256": manifest_sha,
               "model_sha256": BASE_SHA256, "adapter_sha256": sha256(a.adapter) if a.adapter else None,
               "calibration_sha256": sha256(a.calibration) if a.calibration else None,
               "executable_sha256": sha256(a.executable), "gpu": gpu_snapshot(),
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "started_at": datetime.now(timezone.utc).isoformat(), "context_tokens": 16384, "kv": "q8_0",
               "choice_order": "canonical"}
    (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    backend = NativeReadout(a.executable, a.model, 16384, adapter=a.adapter)
    try:
        engine = DecisionEngine(backend, calibration, canonical_choices=True)
        # Batch-one native canary before the complete split.
        canary = {"type": "choice", "instructions": "Read the selected key.", "criteria": {"red": None, "blue": None}}
        if engine.answer({"selected_key": "blue"}, canary)[0]["choice"] != "blue":
            raise RuntimeError("native batch-one canary failed")
        predictions = infer(engine, records, a.output / "predictions.jsonl")
    finally:
        backend.close()
    metrics = report(records, predictions)
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (a.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    overall = metrics["overall"]
    print(json.dumps({"split": a.split, "n": len(records), "accuracy": overall["accuracy_failures_incorrect"],
                      "nll": overall["nll"]["mean"], "valid": overall["valid"]}))


if __name__ == "__main__":
    main()
