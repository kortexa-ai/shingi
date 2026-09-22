"""Start the local Shingi API and exercise the real TypeSafe SDK on native logits."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import httpx
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul
from shingi.metrics import percentile
from shingi.release import artifact_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("artifacts/bin/readout"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = "http://127.0.0.1:8765"
    # Do not start a duplicate service on the reserved local port.
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 8765))
    command = [sys.executable, "-m", "shingi.server", "--model", str(args.model), "--calibration", str(args.calibration)]
    command += ["--executable", str(args.executable)]
    if args.adapter:
        command += ["--adapter", str(args.adapter)]
    process = subprocess.Popen(command)
    observations = {}
    try:
        deadline = time.monotonic() + 180
        with httpx.Client(base_url=base, timeout=120) as http:
            while True:
                if process.poll() is not None:
                    raise RuntimeError("API exited during startup")
                try:
                    if http.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("API readiness timeout")
                time.sleep(.5)
            observations["version"] = http.get("/v1/version").json()
            assert observations["version"]["trained"] == (args.adapter is not None)
            expected = artifact_identity(args.model, args.adapter)
            assert observations["version"]["model_sha256"] == expected["model_sha256"]
            assert observations["version"]["adapter_sha256"] == expected["adapter_sha256"]
            with TypeSafeClient(api_key="local-research", base_url=base, model="shingi", timeout=120) as client:
                result = client.system_one(state={"message": "The parcel arrived damaged. Please send a replacement. I do not want a refund.", "severity": "low"}, questions={
                    "route": Choice(instructions="Which team handles the requested replacement?", criteria={"returns": "Damaged goods and replacements", "billing": "Charges and refunds", "shipping": "Delivery tracking"}),
                    "refund": Noul(instructions="Does the customer want a refund?"),
                    "severity": Score(instructions="Read the explicit severity field.", criteria=["low", "medium", "high"]),
                })
                observations["mixed"] = result.model_dump()
                assert result.choices["route"].choice == "returns"
                assert result.nouls["refund"].noul < .5
                assert result.scores["severity"].score < .5
                for count in (52, 53, 255):
                    key = f"item_{count - 1}"
                    result = client.system_one(state={"selected_key": key}, questions={"selection": Choice(
                        instructions="Choose the exact key named in state.selected_key.", criteria={f"item_{i}": None for i in range(count)})})
                    answer = result.choices["selection"]
                    observations[f"choices_{count}"] = result.model_dump()
                    assert len(answer.probabilities) == count
                    assert abs(sum(answer.probabilities.values()) - 1) < 1e-6
                    assert all(math.isfinite(p) and 0 <= p <= 1 for p in answer.probabilities.values())
                    assert answer.probabilities[answer.choice] == max(answer.probabilities.values())
                    assert answer.choice == key
                short = {"color": Choice(instructions="Read the stored color.",
                    criteria={"red": None, "green": None, "blue": None})}
                for _ in range(3):
                    client.system_one(state={"color": "blue"}, questions=short)
                elapsed = []
                for _ in range(30):
                    started = time.perf_counter()
                    result = client.system_one(state={"color": "blue"}, questions=short)
                    elapsed.append(1000 * (time.perf_counter() - started))
                    assert result.choices["color"].choice == "blue"
                observations["short_http_latency_ms"] = {"n": len(elapsed), "warmup": 3,
                    "median": percentile(elapsed, .5), "p95": percentile(elapsed, .95), "samples": elapsed,
                    "scope": "Sequential localhost TypeSafe SDK round trips; one short three-choice question; model already loaded."}
                reversed_short = {"color": Choice(instructions="Read the stored color.",
                    criteria={"blue": None, "green": None, "red": None})}
                reordered = client.system_one(state={"color": "blue"}, questions=reversed_short)
                assert reordered.choices["color"].probabilities == result.choices["color"].probabilities
            invalid = http.post("/v1/systemone", json={"model": "shingi", "state": "x", "questions": {"bad": {"type": "choice", "instructions": "pick", "criteria": {str(i): None for i in range(256)}}}})
            observations["invalid_status"] = invalid.status_code
            assert invalid.status_code == 422
            oversized = http.post("/v1/systemone", json={"model": "shingi", "state": "oversized context " * 20000,
                "questions": {"q": {"type": "noul", "instructions": "Is this short?"}}})
            observations["oversized_status"] = oversized.status_code
            assert oversized.status_code == 422
            assert http.get("/health").status_code == 200
            observations["passed"] = True
    except BaseException as exc:
        observations["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        (args.output / "result.json").write_text(json.dumps(observations, indent=2) + "\n")


if __name__ == "__main__":
    main()
