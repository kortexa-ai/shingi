"""Process-owned, explicitly authorized 4090 borrowing block for Shingi.

Only named 4090 services can stop. The 6000 and Miso are untouched.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from shingi.backend import GPU_4090

SERVICES = [("models/lfm2.5-8b-a1b", 2059), ("tts.server", 4003), ("asr.server/qwen", 4002)]


def running_services(snapshot):
    running, group = set(), None
    for line in snapshot.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0] in ("PROJECT", "-------"):
            continue
        if not line[0].isspace():
            group = fields[0]
            if fields[3] == "yes":
                running.add(group)
        elif fields[3] == "yes" and group:
            prefix = "models" if group == "models.server" else group
            running.add(prefix + "/" + fields[0])
    return running


def free_mib():
    return int(subprocess.check_output(["nvidia-smi", "--id=" + GPU_4090,
                                       "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def terminate_child(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def execute_block(command, output, *, timeout=8 * 3600):
    if socket.gethostname().split(".")[0] != "smarty":
        raise RuntimeError("4090 borrowing must run on Smarty")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = subprocess.check_output(["ktxsvc", "list"], text=True)
    baseline = running_services(snapshot)
    baseline_gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True)
    (output / "services-before.txt").write_text(snapshot)
    (output / "gpu-before.csv").write_text(baseline_gpu)
    active = [(service, port) for service, port in SERVICES if service in baseline]
    if any(not health(port) for _, port in active):
        raise RuntimeError("a target service is unhealthy before borrowing")
    record = {"gpu_uuid": GPU_4090, "started_at": datetime.now(timezone.utc).isoformat(),
              "baseline_targets": [name for name, _ in active], "stopped": [], "restored": [],
              "free_before_mib": free_mib(), "timeout_seconds": timeout}
    stopped, child, status = [], None, 1

    def save():
        (output / "block.json").write_text(json.dumps(record, indent=2) + "\n")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        save()
        for service, port in active:
            if free_mib() >= 14 * 1024:
                break
            stopped.append((service, port))
            record["stopped"].append(service)
            save()
            print(f"Stopping {service} for the authorized 4090 block", flush=True)
            before = free_mib()
            subprocess.run(["ktxsvc", "stop", service], check=True)
            deadline = time.monotonic() + 45
            while free_mib() < before + 256 and time.monotonic() < deadline:
                time.sleep(1)
        if free_mib() < 14 * 1024:
            raise RuntimeError("insufficient 4090 memory after authorized stops")
        record["free_after_stops_mib"] = free_mib()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_4090, SHINGI_ALLOW_4090="1")
        child = subprocess.Popen(command, env=environment, start_new_session=True)
        record["child_pid"] = child.pid
        save()
        status = child.wait(timeout=timeout)
        record["child_exit_code"] = status
    except BaseException as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        terminate_child(child)
        restore_errors = []
        deadline = time.monotonic() + 60
        release_target = record.get("free_after_stops_mib", record["free_before_mib"]) - 512
        while free_mib() < release_target and time.monotonic() < deadline:
            time.sleep(1)
        memory_released = free_mib() >= release_target
        for service, port in reversed(stopped):
            try:
                if not memory_released:
                    raise RuntimeError("child GPU allocation was not released; refusing unsafe service load")
                subprocess.run(["ktxsvc", "start", service], check=True)
                deadline = time.monotonic() + 600
                while not health(port) and time.monotonic() < deadline:
                    time.sleep(2)
                if not health(port):
                    raise RuntimeError("health timeout")
                record["restored"].append(service)
            except Exception as exc:
                restore_errors.append(f"{service}: {exc}")
        record["health_after"] = {service: health(port) for service, port in active}
        record["restore_errors"] = restore_errors
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["free_after_mib"] = free_mib()
        (output / "services-after.txt").write_text(subprocess.check_output(["ktxsvc", "list"], text=True))
        (output / "gpu-after.csv").write_text(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True))
        save()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if restore_errors or not all(record["health_after"].values()):
            raise RuntimeError("4090 service restoration failed; inspect block.json")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required")
    raise SystemExit(execute_block(command, args.output))


if __name__ == "__main__":
    main()
