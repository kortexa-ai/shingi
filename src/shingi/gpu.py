"""Portable CUDA memory gates. This module never manages system services."""
import os
import re
import subprocess


def selected_gpu():
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one full GPU UUID from nvidia-smi -L")
    return uuid


def gpu_snapshot():
    uuid = selected_gpu()
    row = subprocess.check_output(
        ["nvidia-smi", "--id=" + uuid, "--query-gpu=uuid,name,memory.total,memory.free",
         "--format=csv,noheader,nounits"], text=True, timeout=10)
    fields = [s.strip() for s in row.strip().split(",")]
    if len(fields) != 4 or fields[0] != uuid:
        raise RuntimeError("nvidia-smi returned an unexpected GPU identity")
    return {"uuid": fields[0], "name": fields[1], "total_mib": int(fields[2]), "free_mib": int(fields[3])}


def gpu_profile():
    gpu = gpu_snapshot()
    if gpu["total_mib"] < 20 * 1024:
        raise RuntimeError("CUDA v1 requires a GPU with at least 20 GiB usable VRAM")
    preload, headroom = (14, 4) if gpu["total_mib"] <= 32 * 1024 else (30, 10)
    return gpu["uuid"], preload * 1024, headroom * 1024


def gpu_free_mib():
    return gpu_snapshot()["free_mib"]


def require_training_gpu():
    """The expanded frozen training base needs a separate 96 GB class GPU."""
    selected_gpu()
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("training requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.88, 0)
    if torch.cuda.mem_get_info()[0] < 80 * 2**30:
        raise RuntimeError("training requires 80 GiB free before loading the expanded base")


def assert_post_first_backward_free():
    import torch
    torch.cuda.synchronize()
    if torch.cuda.mem_get_info()[0] < 10 * 2**30:
        raise RuntimeError("training free memory fell below the 10 GiB floor")
