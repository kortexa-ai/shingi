import importlib.util
from pathlib import Path

import pytest

from shingi.backend import GPU_6000, GPU_4090, gpu_profile

spec = importlib.util.spec_from_file_location("borrow_4090", Path(__file__).parents[1] / "scripts/borrow_4090.py")
borrow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(borrow)


def test_4090_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_4090)
    monkeypatch.delenv("SHINGI_ALLOW_4090", raising=False)
    with pytest.raises(RuntimeError):
        gpu_profile()
    monkeypatch.setenv("SHINGI_ALLOW_4090", "1")
    assert gpu_profile() == (GPU_4090, 14 * 1024, 4 * 1024)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_6000)
    assert gpu_profile() == (GPU_6000, 30 * 1024, 10 * 1024)


def test_service_parser_keeps_parent_groups_and_running_state():
    snapshot = """PROJECT INSTALLED ENABLED RUNNING UNIT
models.server - - - -
  lfm2.5-8b-a1b yes yes yes ok
  bonsai-2-27b yes yes no ok
asr.server no - - -
  qwen yes yes yes ok
tts.server yes yes yes ok
legolm - - - -
  legolm.miso yes yes yes ok
"""
    running = borrow.running_services(snapshot)
    assert running == {"models/lfm2.5-8b-a1b", "asr.server/qwen", "tts.server", "legolm/legolm.miso"}
    assert [s for s, _ in borrow.SERVICES if s in running] == ["models/lfm2.5-8b-a1b", "tts.server", "asr.server/qwen"]


def test_failed_child_restores_only_services_stopped_for_memory(monkeypatch, tmp_path):
    snapshot = "models.server - - - -\n  lfm2.5-8b-a1b yes yes yes ok\ntts.server yes yes yes ok\nasr.server no - - -\n  qwen yes yes yes ok\n"
    calls, stopped = [], set()
    monkeypatch.setattr(borrow.socket, "gethostname", lambda: "smarty")
    monkeypatch.setattr(borrow, "health", lambda port: True)
    monkeypatch.setattr(borrow, "free_mib", lambda: 500 + (10000 if "models/lfm2.5-8b-a1b" in stopped else 0) + (5000 if "tts.server" in stopped else 0))
    monkeypatch.setattr(borrow.subprocess, "check_output", lambda cmd, **kwargs: snapshot if cmd[0] == "ktxsvc" else "gpu baseline")

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "stop":
            stopped.add(cmd[2])
        elif cmd[1] == "start":
            stopped.remove(cmd[2])
    monkeypatch.setattr(borrow.subprocess, "run", run)

    class Child:
        pid = 123456
        def wait(self, **kwargs):
            return 7
        def poll(self):
            return 7
    monkeypatch.setattr(borrow.subprocess, "Popen", lambda cmd, **kwargs: Child())
    assert borrow.execute_block(["failing-job"], tmp_path / "block") == 7
    assert calls == [["ktxsvc", "stop", "models/lfm2.5-8b-a1b"], ["ktxsvc", "stop", "tts.server"],
                     ["ktxsvc", "start", "tts.server"], ["ktxsvc", "start", "models/lfm2.5-8b-a1b"]]
    assert not stopped
