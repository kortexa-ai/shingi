"""One process owns the native model and serializes access to its context."""
import json
import os
import selectors
import subprocess
import threading

GPU_UUID = "GPU-a71210ca-e14a-755a-88bb-77f53a2102f6"


def gpu_free_mib():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("pin CUDA_VISIBLE_DEVICES to the RTX PRO 6000 UUID")
    row = subprocess.check_output(["nvidia-smi", "--id=" + GPU_UUID,
                                   "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    uuid, free = row.strip().split(",")
    if uuid != GPU_UUID:
        raise RuntimeError("unexpected GPU identity")
    return int(free.strip())


class NativeReadout:
    def __init__(self, executable, model, context_tokens=16384):
        self.lock = threading.Lock()
        if gpu_free_mib() < 30 * 1024:
            raise RuntimeError("native canary requires at least 30 GiB free before loading")
        self.process = subprocess.Popen([str(executable), str(model), str(context_tokens)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.info = self._read(300)
            if not self.info.get("ready"):
                raise RuntimeError("native model failed to initialize")
            if gpu_free_mib() < 10 * 1024:
                raise RuntimeError("less than 10 GiB headroom after model load")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                self.close()
                raise TimeoutError("native readout timed out; process stopped")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native readout exited")
        result = json.loads(line)
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def infer(self, prompt, labels):
        with self.lock:
            if gpu_free_mib() < 10 * 1024:
                self.close()
                raise RuntimeError("GPU headroom fell below 10 GiB; native model stopped")
            if self.process.poll() is not None:
                raise RuntimeError("native readout is not running")
            self.process.stdin.write(json.dumps({"prompt": prompt, "labels": labels}) + "\n")
            self.process.stdin.flush()
            return self._read(300)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
