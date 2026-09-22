"""Run a bounded native-Bonsai evaluation and preserve every readout trace.

Use only within an authorized, claimed GPU block. This script does not stop
services. Source must be committed so run provenance identifies the code.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from shingi.backend import NativeReadout, gpu_free_mib
from shingi.decision import Calibration, DecisionEngine, MODEL_ID
from shingi.metrics import report
from shingi.probes import probe_at_tokens


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def selected_records(benchmark, phase):
    split = "calibration" if phase in ("canary", "calibration") else "test"
    manifest = json.loads((benchmark / "manifest.json").read_text())
    path = benchmark / (split + ".jsonl")
    if file_sha256(path) != manifest[split + "_sha256"]:
        raise ValueError("dataset hash no longer matches the frozen manifest")
    records = read_jsonl(path)
    if phase == "canary":
        seen = set()
        canaries = []
        for row in records:
            if row["source"] not in seen:
                canaries.append(row)
                seen.add(row["source"])
        records = canaries
    elif phase == "shuffle":
        chosen = set(manifest["shuffle_ids"])
        records = [row for row in records if row["id"] in chosen]
        for row in records:
            original = list(row["question"]["criteria"].items())
            shuffled = list(original)
            seed = int(hashlib.sha256((manifest["seed"] + "|shuffle|" + row["id"]).encode()).hexdigest(), 16)
            random.Random(seed).shuffle(shuffled)
            if shuffled == original:
                shuffled = shuffled[1:] + shuffled[:1]
            row["question"]["criteria"] = dict(shuffled)
            row["original_input_sha256"] = row["input_sha256"]
            row["option_order"] = list(row["question"]["criteria"])
    return records, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("artifacts/bin/readout"))
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/benchmark-v1"))
    parser.add_argument("--phase", choices=["canary", "calibration", "test", "shuffle", "context"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--max-seconds", type=int, default=8 * 3600)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("commit source changes before a measured model run")
    if args.phase == "context":
        manifest = json.loads((args.benchmark / "manifest.json").read_text())
        records = []
    else:
        records, manifest = selected_records(args.benchmark, args.phase)
    calibration_file = json.loads(args.calibration.read_text()) if args.calibration else None
    calibration = Calibration(**calibration_file["parameters"]) if calibration_file else Calibration()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "model": MODEL_ID, "model_path": str(args.model), "model_sha256": file_sha256(args.model),
                  "adapter_path": str(args.adapter) if args.adapter else None,
                  "adapter_sha256": file_sha256(args.adapter) if args.adapter else None,
                  "executable_sha256": file_sha256(args.executable), "phase": args.phase,
                  "context_tokens": args.context_tokens, "calibration": asdict(calibration),
                  "calibration_sha256": file_sha256(args.calibration) if args.calibration else None,
                  "dataset_revision": manifest["revision"], "dataset_manifest_sha256": file_sha256(args.benchmark / "manifest.json"),
                  "started_at": datetime.now(timezone.utc).isoformat(), "planned_records": len(records),
                  "gpu_free_before_mib": gpu_free_mib(), "max_seconds": args.max_seconds}
    if calibration_file:
        fitted = calibration_file["provenance"]
        if fitted.get("adapter_sha256") != provenance["adapter_sha256"]:
            raise SystemExit("calibration provenance differs: adapter_sha256")
        for field in ("model_sha256", "executable_sha256", "dataset_manifest_sha256"):
            if fitted[field] != provenance[field]:
                raise SystemExit(f"calibration provenance differs: {field}")
    (args.output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    backend = None
    predictions = {}
    started = time.monotonic()
    aborted = None
    try:
        backend = NativeReadout(args.executable, args.model, args.context_tokens, adapter=args.adapter)
        engine = DecisionEngine(backend, calibration)
        if args.phase == "context":
            for target in (512, 4096, 8192, min(15000, args.context_tokens - 128)):
                for pos_index, position in enumerate((0.0, .5, 1.0)):
                    row, token_count = probe_at_tokens(backend, target, position, seed=target + pos_index)
                    row.update(id=f"context/{target}/{position}", source="synthetic-context", primitive="choice",
                               input_tokens=token_count, target_tokens=target, needle_position=position)
                    row["input_sha256"] = hashlib.sha256(json.dumps([row["state"], row["question"]], sort_keys=True).encode()).hexdigest()
                    records.append(row)
            provenance["planned_records"] = len(records)
            provenance["context_probe"] = "Synthetic repeated filler and planted fact; not real-document comprehension."
        with (args.output / "predictions.jsonl").open("x") as stream:
            for index, record in enumerate(records):
                if time.monotonic() - started >= args.max_seconds:
                    aborted = "wall-time budget exhausted"
                    break
                item = {"id": record["id"], "source": record["source"], "primitive": record["primitive"],
                        "input_sha256": record["input_sha256"], "option_order": record.get("option_order"), "error": None}
                if args.phase == "context":
                    item.update({key: record[key] for key in ("target_tokens", "input_tokens", "needle_position", "label")})
                # Preserve actual insertion order separately from the benchmark's
                # canonical semantic hash: candidate order affects the prompt.
                wire = json.dumps([record["state"], record["question"]], ensure_ascii=False, separators=(",", ":"))
                item["wire_input_sha256"] = hashlib.sha256(wire.encode()).hexdigest()
                before = time.monotonic()
                try:
                    answer, traces = engine.answer(record["state"], record["question"])
                    item.update(answer=answer, traces=traces, gpu_free_after_mib=gpu_free_mib())
                except (ValueError, RuntimeError, TimeoutError, BrokenPipeError) as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                    if not isinstance(exc, ValueError):
                        aborted = item["error"]
                item["latency_ms"] = (time.monotonic() - before) * 1000
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                predictions[record["id"]] = item
                if index == 0 or (index + 1) % 25 == 0:
                    print(json.dumps({"completed": index + 1, "planned": len(records), "id": item["id"],
                                      "error": item["error"], "elapsed_seconds": time.monotonic() - started}), flush=True)
                if aborted:
                    break
    except BaseException as exc:
        aborted = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if backend is not None:
            backend.close()
        provenance.update(completed_at=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=time.monotonic() - started, completed_records=len(predictions),
                          aborted=aborted, gpu_free_after_mib=gpu_free_mib())
        (args.output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    metrics = report(records, predictions)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"planned": len(records), "completed": len(predictions), "aborted": aborted,
                      "valid": metrics["overall"]["valid"], "correct": metrics["overall"]["correct"]}), flush=True)
    if aborted or metrics["overall"]["valid"] != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
