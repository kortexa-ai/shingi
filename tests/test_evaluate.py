import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluate_native", Path(__file__).parents[1] / "scripts/evaluate_native.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_shuffling_preserves_labels_and_forces_changed_order(tmp_path):
    row = {"id": "test-1", "source": "x", "primitive": "choice", "state": "sample", "input_sha256": "original",
           "question": {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}}, "label": "a"}
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(row) + "\n")
    manifest = {"test_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": "fixed", "shuffle_ids": ["test-1"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows, _ = evaluate.selected_records(tmp_path, "shuffle")
    assert list(rows[0]["question"]["criteria"]) == ["b", "a"]
    assert rows[0]["label"] == "a"
    assert rows[0]["original_input_sha256"] == "original"


def test_frozen_dataset_mutation_is_rejected(tmp_path):
    (tmp_path / "test.jsonl").write_text('{}\n')
    (tmp_path / "manifest.json").write_text(json.dumps({"test_sha256": "wrong"}))
    import pytest
    with pytest.raises(ValueError, match="hash"):
        evaluate.selected_records(tmp_path, "test")
