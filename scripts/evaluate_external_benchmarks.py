"""Evaluate only the frozen v0.2 release on a pre-hashed external inventory."""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from benchmark_speed import MemorySampler
from evaluate_release import infer
from prepare_benchmark import read_jsonl
from shingi.backend import NativeReadout
from shingi.decision import DecisionEngine
from shingi.decision import Calibration
from shingi.external_benchmarks import digest, suite_report
from shingi.gpu import gpu_snapshot
from shingi.release import artifact_identity, load_calibration, RELEASE_MODEL_ID, require_release_environment, sha256


def audit(data, run):
    """Replay saved logits, verify exact prompts/answers, and retain raw prompts."""
    import hashlib
    import math
    manifest = json.loads((data / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    receipt = summary["receipt"]
    if (sha256(data / "manifest.json") != receipt["data_manifest_sha256"]
            or sha256(data / "records.jsonl") != manifest["records_sha256"]):
        raise ValueError("audit input manifest mismatch")
    rows = read_jsonl(data / "records.jsonl")
    pairs = [r for r in rows if r["id"] in set(manifest["order_ids"])]
    reversed_rows = copy.deepcopy(pairs)
    for r in reversed_rows:
        r["question"]["criteria"] = dict(reversed(list(r["question"]["criteria"].items())))
    variants = [(suite, [r for r in rows if r["suite"] == suite], True)
                for suite in ("decisionbench", "jevbench", "this-that")]
    variants += [("canonical-reverse", reversed_rows, True), ("input-order", pairs, False),
                 ("input-order-reverse", reversed_rows, False)]
    count, failures, traces_count, max_tokens = 0, [], 0, 0
    token_ids = {}
    output = run / "verified-prompts.jsonl"
    with output.open("x") as stream:
        for variant, records, canonical in variants:
            path = run / (variant+".jsonl")
            if sha256(path) != summary["artifact_sha256"][path.name]:
                raise ValueError("prediction artifact hash mismatch")
            predictions = read_jsonl(path)
            if [r["id"] for r in records] != [p["id"] for p in predictions]:
                raise ValueError("prediction inventory/order mismatch")
            for row, pred in zip(records, predictions):
                if row["input_sha256"] != pred["input_sha256"]:
                    raise ValueError("prediction input hash mismatch")
                if pred.get("error"):
                    failures.append({"variant": variant, "id": row["id"], "error": pred["error"]})
                    continue

                class Replay:
                    index = 0

                    def infer(self, prompt, labels):
                        nonlocal traces_count, max_tokens
                        trace = pred["traces"][self.index]
                        h = hashlib.sha256(prompt.encode()).hexdigest()
                        if h != trace["prompt_sha256"]:
                            raise ValueError("reconstructed prompt mismatch")
                        if (len(labels) != len(trace["logits"]) or len(labels) != len(trace["candidate_ids"])
                                or len(set(trace["candidate_ids"])) != len(labels)
                                or not all(math.isfinite(x) for x in trace["logits"])):
                            raise ValueError("incomplete candidate logits")
                        for label, candidate in zip(labels, trace["candidate_ids"]):
                            if label in token_ids and token_ids[label] != candidate:
                                raise ValueError("candidate token identity changed")
                            token_ids[label] = candidate
                        stream.write(json.dumps({"variant": variant, "id": row["id"], "step": self.index,
                            "prompt": prompt, "letters": labels, **trace}, ensure_ascii=False)+"\n")
                        self.index += 1
                        traces_count += 1
                        max_tokens = max(max_tokens, trace["input_tokens"])
                        return trace

                replay = Replay()
                engine = DecisionEngine(replay, Calibration(**receipt["calibration"]), canonical_choices=canonical)
                answer, _ = engine.answer(row["state"], row["question"])
                if answer != pred["answer"] or replay.index != len(pred["traces"]):
                    raise ValueError("saved answer does not match frozen logit replay")
                count += 1
    result = {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "main_questions": len(rows), "order_diagnostic_predictions": 3*len(pairs),
              "replayed_answers": count, "verified_traces": traces_count, "failures": failures,
              "candidate_token_ids": token_ids, "maximum_input_tokens": max_tokens,
              "raw_prompts_sha256": sha256(output), "summary_sha256": sha256(run / "summary.json"),
              "manifest_sha256": sha256(data / "manifest.json")}
    (run / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def order_summary(rows, first, second):
    valid = []
    for r in rows:
        a, b = first.get(r["id"]), second.get(r["id"])
        if a and b and not a.get("error") and not b.get("error"):
            x, y = a["answer"]["probabilities"], b["answer"]["probabilities"]
            valid.append({"flip": a["answer"]["choice"] != b["answer"]["choice"],
                          "tvd": sum(abs(x[k] - y[k]) for k in x)/2})
    return {"planned": len(rows), "valid": len(valid), "flips": sum(r["flip"] for r in valid),
            "mean_tvd": sum(r["tvd"] for r in valid)/len(valid) if valid else None,
            "max_tvd": max((r["tvd"] for r in valid), default=None)}


def main():
    p = argparse.ArgumentParser()
    for name in ("data", "output", "model", "adapter", "executable", "calibration"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    a = p.parse_args()
    require_release_environment()
    if sha256(a.data / "manifest.json") != a.manifest_sha256:
        raise ValueError("external manifest differs from frozen identity")
    manifest = json.loads((a.data / "manifest.json").read_text())
    if sha256(a.data / "records.jsonl") != manifest["records_sha256"]:
        raise ValueError("evaluation records changed after freeze")
    rows = read_jsonl(a.data / "records.jsonl")
    if len(rows) != manifest["records"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unexpected records inventory")
    if any(r["input_sha256"] != digest([r["state"], r["question"]]) for r in rows):
        raise ValueError("input hash mismatch")
    identity = artifact_identity(a.model, a.adapter)
    if identity["model"] != RELEASE_MODEL_ID:
        raise ValueError("this experiment requires current v0.2 release weights")
    calibration, cal_hash = load_calibration(a.calibration, identity)
    if cal_hash != "d9523c7a27aaf85fe34bd68bffe4c7b81c31d4a7e9036172394e3d461de52819":
        raise ValueError("this experiment requires frozen v0.2 calibration")
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {"started_at": datetime.now(timezone.utc).isoformat(), "identity": identity,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "gpu": gpu_snapshot(), "data_manifest_sha256": a.manifest_sha256,
               "executable_sha256": sha256(a.executable), "calibration_sha256": cal_hash,
               "calibration": asdict(calibration), "context_tokens": 16384,
               "kv": "q8_0", "batch": 1, "choice_order": "canonical", "retries": 0}
    (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    sampler = MemorySampler()
    backend = None
    results = {"receipt": receipt, "benchmarks": {}, "order": {}}
    try:
        sampler.phase = "load"
        backend = NativeReadout(a.executable, a.model, 16384, adapter=a.adapter)
        sampler.native_pid = backend.process.pid
        receipt["native"] = backend.info
        engine = DecisionEngine(backend, calibration, model_id=RELEASE_MODEL_ID, canonical_choices=True)
        sampler.phase = "warmup"
        answer, traces = engine.answer({"selected_key": "blue"},
            {"type": "choice", "instructions": "Read the selected key.", "criteria": {"red": None, "blue": None}})
        (a.output / "canary.json").write_text(json.dumps({"answer": answer, "traces": traces}, indent=2)+"\n")
        if answer["choice"] != "blue":
            raise RuntimeError("batch-one canary failed")
        sampler.phase = "mixed"
        all_predictions = {}
        # The comparator sets are small; complete these first without changing the frozen inventory.
        for suite in ("decisionbench", "jevbench", "this-that"):
            selected = [r for r in rows if r["suite"] == suite]
            predictions = infer(engine, selected, a.output / (suite+".jsonl"))
            all_predictions.update(predictions)
            keys = sorted({r["subset"] for r in selected}) if suite != "this-that" else []
            results["benchmarks"][suite] = {"all": suite_report(selected, predictions),
                "subsets": {key: suite_report([r for r in selected if r["subset"] == key], predictions) for key in keys}}
            (a.output / "summary.json").write_text(json.dumps(results, indent=2)+"\n")
            print(json.dumps({"suite": suite, "correct": results["benchmarks"][suite]["all"]["correct"],
                              "n": len(selected)}), flush=True)
        order_ids = set(manifest["order_ids"])
        pairs = [r for r in rows if r["id"] in order_ids]
        reversed_rows = copy.deepcopy(pairs)
        for r in reversed_rows:
            r["question"]["criteria"] = dict(reversed(list(r["question"]["criteria"].items())))
        canonical_reverse = infer(engine, reversed_rows, a.output / "canonical-reverse.jsonl")
        raw_engine = DecisionEngine(backend, calibration, model_id=RELEASE_MODEL_ID)
        raw_first = infer(raw_engine, pairs, a.output / "input-order.jsonl")
        raw_reverse = infer(raw_engine, reversed_rows, a.output / "input-order-reverse.jsonl")
        for suite in ("decisionbench", "jevbench", "this-that"):
            selected = [r for r in pairs if r["suite"] == suite]
            results["order"][suite] = {
                "released_canonical": order_summary(selected, all_predictions, canonical_reverse),
                "raw_input_order_diagnostic": order_summary(selected, raw_first, raw_reverse)}
    finally:
        if backend is not None:
            backend.close()
        sampler.close()
        (a.output / "memory-samples.json").write_text(json.dumps(sampler.samples)+"\n")
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        receipt["total_seconds"] = time.perf_counter() - started
        results["memory"] = sampler.summary()
        results["artifact_sha256"] = {p.name: sha256(p) for p in a.output.iterdir()
                                       if p.is_file() and p.name not in ("summary.json", "receipt.json")}
        (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
        (a.output / "summary.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
