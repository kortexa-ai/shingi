import pytest
from shingi import gpu
from shingi.backend import NativeReadout

UUID = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


@pytest.mark.parametrize('value', ['', '0', '0,1', 'GPU-short', UUID + ',' + UUID])
def test_requires_one_unambiguous_gpu(value, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', value)
    with pytest.raises(RuntimeError, match='exactly one full GPU UUID'):
        gpu.selected_gpu()


@pytest.mark.parametrize('total,expected', [(23028, (14*1024, 4*1024)), (97887, (30*1024, 10*1024))])
def test_profile_uses_capacity_without_machine_whitelist(total, expected, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: f'{UUID}, Test GPU, {total}, 18000\n')
    assert gpu.gpu_profile() == (UUID, *expected)
    assert gpu.gpu_free_mib() == 18000


def test_rejects_wrong_device_response(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    monkeypatch.setattr(gpu.subprocess, 'check_output', lambda *a, **kw: 'GPU-other, Test GPU, 23028, 18000\n')
    with pytest.raises(RuntimeError, match='unexpected GPU identity'):
        gpu.gpu_snapshot()


def test_context_bound_checked_before_gpu_access(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with pytest.raises(ValueError, match='16,384'):
        NativeReadout('unused', 'unused', 16385)


def test_driver_failure_is_backend_unavailability(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    def fail(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(gpu.subprocess, 'check_output', fail)
    with pytest.raises(RuntimeError, match='could not query'):
        gpu.gpu_snapshot()
