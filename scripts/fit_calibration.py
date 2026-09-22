import argparse
import hashlib
import json
from pathlib import Path

from shingi.calibration import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/benchmark-v1"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
