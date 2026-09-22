import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prepare_benchmark", Path(__file__).parents[1] / "scripts/prepare_benchmark.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def row(identifier, state):
    return prepare.decode_row({"id": identifier, "state": state, "question": {"type": "noul", "instructions": "yes?"}})


def test_overlap_guard_checks_content_not_only_ids():
    existing = row("test-1", {"text": "duplicate"})
    candidates = [row("calibration-1", {"text": "duplicate"}), row("calibration-2", {"text": "new"})]
    selected, _ = prepare.select_unique(candidates, 1, excluded_states={existing["state_sha256"]})
    assert selected[0]["id"] == "calibration-2"


def test_selection_is_stable_and_fails_when_unique_population_too_small():
    records = [row(str(i), {"text": str(i)}) for i in range(8)]
    assert prepare.select_unique(records, 4) == prepare.select_unique(list(reversed(records)), 4)
    with pytest.raises(ValueError, match="distinct"):
        prepare.select_unique(records, 9)


def test_unicode_line_separator_inside_json_is_not_a_record_boundary(tmp_path):
    record = {"text": "one\u2028two"}
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    assert prepare.read_jsonl(path) == [record]
