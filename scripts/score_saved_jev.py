"""Score published Jev predictions on the frozen slice; no hosted API calls."""
import argparse
import json
from pathlib import Path

from shingi.metrics import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("artifacts/benchmark-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.benchmark / "test.jsonl").open()]
    selected = {row["id"] for row in records}
    predictions, models, normalized = {}, set(), 0
    for line in (args.benchmark / "raw/results/jev-1.13.0/test_predictions.jsonl").open():
        original = json.loads(line)
        if original["id"] not in selected:
            continue
        models.add(original["model"])
        kind = original["primitive"]
        answer = {"type": kind}
        if kind == "noul":
            answer["noul"] = original["p_yes"]
        else:
            answer["probabilities"] = original["probabilities"]
            answer["confidence"] = original["confidence"]
            if kind == "choice":
                answer["choice"] = original["answer"]
            else:
                answer["score"] = original["answer"]
            if original["probabilities"] and abs(sum(original["probabilities"].values()) - 1) > 1e-6:
                normalized += 1
        predictions[original["id"]] = {"answer": answer, "error": original["error"],
                                        "latency_ms": original["latency_ms"]}
    result = report(records, predictions, allow_rounding=True)
    result["provenance"] = {"models": sorted(models), "predictions_renormalized": normalized,
                            "evidence_class": "Our reanalysis of third-party public predictions; no hosted Jev calls.",
                            "publisher": "Praveenrajus / uspraveen (JevBench)",
                            "source_url": "https://huggingface.co/datasets/Praveenrajus/jev-bench/blob/002ad22de8db2df5e0eb898b3da8072dbd4af4de/results/jev-1.13.0/test_predictions.jsonl",
                            "source": "Published predictions from pinned JevBench revision; not our live hosted run.",
                            "pairing": "ID join to same-revision records. Prediction file lacks independent input hashes.",
                            "latency": "Published hosted latency includes its original hardware and network; not comparable to local timing."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"overall": {key: result["overall"][key] for key in ("n", "valid", "correct", "accuracy_failures_incorrect")},
                      "by_source": {key: value["accuracy_failures_incorrect"] for key, value in result["by_source"].items()},
                      "provenance": result["provenance"]}, indent=2))


if __name__ == "__main__":
    main()
