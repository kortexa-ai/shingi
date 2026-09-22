from shingi.probes import context_record, probe_at_tokens


class FakeTokenizer:
    def infer(self, prompt, labels, *, tokenize_only=False):
        assert tokenize_only
        return {"input_tokens": len(prompt.split())}


def test_context_probe_targets_budget_and_moves_fact():
    row, count = probe_at_tokens(FakeTokenizer(), 1000, .5, 7)
    assert 970 < count <= 1000
    assert row["label"] in row["question"]["criteria"]
    assert row["state"].count("Audit entry:") == 1
    assert context_record(20, 0, 7)["state"].startswith("Audit entry:")
    assert context_record(20, 1, 7)["state"].rstrip().endswith(row["label"] + ".")
