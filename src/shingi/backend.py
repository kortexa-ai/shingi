"""One process owns the native model and serializes access to its context."""
import json
import hashlib
import selectors
import subprocess
import threading

from .gpu import gpu_free_mib, gpu_profile


class NativeReadout:
    def __init__(self, executable, model, context_tokens=16384, *, adapter=None):
        self.lock = threading.Lock()
        if not 512 <= context_tokens <= 16384:
            raise ValueError("CUDA v1 supports contexts from 512 through 16,384 tokens")
        gpu, preload, self.headroom = gpu_profile()
        if gpu_free_mib() < preload:
            raise RuntimeError(f"native canary requires at least {preload} MiB free before loading")
        command = [str(executable), str(model), str(context_tokens)]
        if adapter is not None:
            command.append(str(adapter))
        self.process = subprocess.Popen(command,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.info = self._read(300)
            if not self.info.get("ready"):
                raise RuntimeError("native model failed to initialize")
            if gpu_free_mib() < self.headroom:
                raise RuntimeError("insufficient GPU headroom after model load")
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
        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("native readout returned malformed JSON") from exc
        if "error" in result:
            error = ValueError if result.get("error_kind") == "input" else RuntimeError
            raise error(result["error"])
        return result

    def infer(self, prompt, labels, *, tokenize_only=False):
        with self.lock:
            if gpu_free_mib() < self.headroom:
                self.close()
                raise RuntimeError("GPU headroom fell below profile floor; native model stopped")
            if self.process.poll() is not None:
                raise RuntimeError("native readout is not running")
            self.process.stdin.write(json.dumps({"prompt": prompt, "labels": labels, "tokenize_only": tokenize_only}) + "\n")
            self.process.stdin.flush()
            result = self._read(300)
            result["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
