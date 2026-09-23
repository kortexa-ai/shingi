"""Artifact identity and calibration checks shared by serving and evaluation."""
import hashlib
import json
import os
from pathlib import Path

from .decision import Calibration, MODEL_ID

BASE_SHA256 = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
LEGACY_ADAPTER_SHA256 = "d7ea6bf61f6ef5fe26bd82ca1ece4686c20d29a1937834a18239629bb6db31d4"
ADAPTER_SHA256 = "6f4c2b7434b5ef6e0bb83b6f9c196de00c850a79b9380175e06a1df2ae1b7dd2"
RELEASE_MODEL_ID = "shingi-bonsai-2-27b-v0.2"
READOUT_VERSION = "bonsai-sorted-choice-v2"


def require_release_environment():
    if any(name in os.environ for name in ("SHINGI_KV_F16", "SHINGI_VOCAB_ONLY")):
        raise RuntimeError("unset SHINGI_KV_F16 and SHINGI_VOCAB_ONLY for the Q8 CUDA release runtime")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_identity(model, adapter=None):
    base_hash = sha256(model)
    adapter_hash = sha256(adapter) if adapter is not None else None
    if base_hash == BASE_SHA256 and adapter_hash == ADAPTER_SHA256:
        name = RELEASE_MODEL_ID
    elif base_hash == BASE_SHA256 and adapter_hash == LEGACY_ADAPTER_SHA256:
        name = 'shingi-bonsai-2-27b-v0.1'
    elif adapter is None and base_hash == BASE_SHA256:
        name = MODEL_ID
    else:
        name = "shingi-custom"
    return {"model": name, "model_sha256": base_hash, "adapter_sha256": adapter_hash,
            "trained": adapter is not None, "readout_version": READOUT_VERSION}


def load_calibration(path, identity):
    if path is None:
        return Calibration(), None
    data = Path(path).read_bytes()
    fitted = json.loads(data)
    provenance = fitted["provenance"]
    for key in ("model_sha256", "adapter_sha256"):
        if provenance.get(key) != identity[key]:
            raise ValueError(f"calibration was fitted to different weights: {key}")
    if provenance.get("readout_version", "bonsai-letter-v1") != READOUT_VERSION:
        raise ValueError("calibration uses a different readout version")
    return Calibration(**fitted["parameters"]), hashlib.sha256(data).hexdigest()
