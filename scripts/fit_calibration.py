import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from shingi.calibration import fit, mean_nll, TEMPERATURES
from shingi.decision import Calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/benchmark-v1"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="data v3 directory; --run is then an evaluate_records.py output")
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--temperature-only", action="store_true", help="data v3 only: fix the yes/no bias at 0")
    args = parser.parse_args()
    if args.temperature_only and not args.data:
        parser.error("--temperature-only requires --data")
    if args.data:
        return fit_frozen_split(args)
    run = json.loads((args.run / "run.json").read_text())
    manifest = json.loads((args.benchmark / "manifest.json").read_text())
    data = (args.benchmark / "calibration.jsonl").read_bytes()
    if run["phase"] != "calibration" or run["aborted"] is not None:
        raise SystemExit("only a completed calibration run can be fitted")
    if hashlib.sha256(data).hexdigest() != manifest["calibration_sha256"]:
        raise SystemExit("calibration data hash changed")
    if run["dataset_manifest_sha256"] != hashlib.sha256((args.benchmark / "manifest.json").read_bytes()).hexdigest():
        raise SystemExit("run uses a different dataset manifest")
    records = [json.loads(line) for line in data.decode().split("\n") if line.strip()]
    with (args.run / "predictions.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    predictions = {row["id"]: row for row in rows}
    if len(predictions) != len(rows) or set(predictions) != {row["id"] for row in records}:
        raise SystemExit("calibration prediction ID inventory differs")
    result = fit(records, predictions)
    result["provenance"] = {key: run[key] for key in ("code_revision", "model_sha256", "executable_sha256", "dataset_revision", "dataset_manifest_sha256")}
    result["provenance"]["adapter_sha256"] = run.get("adapter_sha256")
    result["provenance"]["predictions_sha256"] = hashlib.sha256((args.run / "predictions.jsonl").read_bytes()).hexdigest()
    result["provenance"]["calibration_data_sha256"] = manifest["calibration_sha256"]
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


def fit_frozen_split(args):
    """Data v3: fit on the whole frozen calibration split, replayed in canonical choice order."""
    from evaluate_records import load_split
    records, manifest_sha = load_split(args.data, args.split)
    receipt = json.loads((args.run / "receipt.json").read_text())
    if receipt["split"] != args.split or receipt["data_manifest_sha256"] != manifest_sha or receipt["calibration_sha256"]:
        raise SystemExit("run is not an uncalibrated evaluation of this frozen split")
    with (args.run / "predictions.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    predictions = {row["id"]: row for row in rows}
    if len(predictions) != len(rows) or set(predictions) != {row["id"] for row in records}:
        raise SystemExit("calibration prediction ID inventory differs")
    if args.temperature_only:
        result = fit_temperatures(records, predictions)
        result["objective"] = f"hard-label NLL on the frozen data v3 {args.split} split only; temperatures only, yes/no bias fixed at 0"
        result["noul_pool"] = noul_pool(records)
    else:
        result = fit(records, predictions, frozen_calibration_split=True, canonical=True)
        result["objective"] = f"hard-label NLL on the frozen data v3 {args.split} split only"
    result["provenance"] = {"source_revision": receipt["source_revision"], "model_sha256": receipt["model_sha256"],
                            "adapter_sha256": receipt["adapter_sha256"], "executable_sha256": receipt["executable_sha256"],
                            "data_manifest_sha256": manifest_sha, "choice_order": "canonical",
                            "predictions_sha256": hashlib.sha256((args.run / "predictions.jsonl").read_bytes()).hexdigest()}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("parameters", "before", "after")}, indent=2))


def fit_temperatures(records, predictions):
    """fit() with canonical replay, its grid and tie-break toward 1.0, but a yes/no bias fixed at 0."""
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("duplicate calibration IDs")
    nll = lambda group, calibration: mean_nll(group, predictions, calibration, canonical=True)
    non_binary = [row for row in records if row["primitive"] != "noul"]
    binary = [row for row in records if row["primitive"] == "noul"]
    trials = [(nll(non_binary, Calibration(temperature=t)), t) for t in TEMPERATURES]
    loss, temperature = min(trials, key=lambda pair: (pair[0], abs(pair[1] - 1)))
    binary_trials = [(nll(binary, Calibration(temperature, t / temperature)), t / temperature) for t in TEMPERATURES]
    binary_loss, noul_temperature = min(binary_trials, key=lambda pair: (pair[0], abs(pair[1] - 1)))
    return {"parameters": asdict(Calibration(temperature, noul_temperature, 0.0)),
            "fit_records": len(records), "choice_score_records": len(non_binary), "noul_records": len(binary),
            "grid": {"temperatures": TEMPERATURES, "noul_biases": [0.0]},
            "before": {"choice_score_nll": nll(non_binary, Calibration()), "noul_nll": nll(binary, Calibration())},
            "after": {"choice_score_nll": loss, "noul_nll": binary_loss},
            "limitations": "Global temperatures only; no weight changes. The yes/no bias is not fitted because the "
                           "split's yes/no pool is small and its gold-yes share reflects a few sources."}


def noul_pool(records):
    """Yes/no records of the split by source, with gold-yes counts."""
    binary = [row for row in records if row["primitive"] == "noul"]
    counts, yes = Counter(row["source"] for row in binary), Counter(row["source"] for row in binary if str(row["label"]) == "1")
    return {"records": len(binary), "gold_yes": sum(yes.values()),
            "gold_yes_share": sum(yes.values()) / len(binary) if binary else None,
            "by_source": {source: {"records": n, "gold_yes": yes[source]} for source, n in sorted(counts.items())}}


if __name__ == "__main__":
    main()
