"""Data v3 source registry and converters for sources absent from jev-bench.

Every converter emits the jev-bench record schema used by earlier Shingi data:
id, source, primitive, split, state, question, label, soft_label and meta.
Licenses are recorded here and verified by scripts/verify_licenses.py; the
registry records provenance, not a legal conclusion.
"""
import hashlib
import json

JEV_REPO = "Praveenrajus/jev-bench"
JEV_REVISION = "002ad22de8db2df5e0eb898b3da8072dbd4af4de"

# Natural fitting sources: name -> (origin, training records). Origin "jev" uses
# the pinned jev-bench transformation; other origins use converters below.
FIT = {
    "go_emotions": ("jev", 3000),
    "helpsteer2_helpfulness": ("jev", 1000),
    "helpsteer2_verbosity": ("jev", 500),
    "helpsteer2_correctness": ("helpsteer2", 500),
    "helpsteer2_coherence": ("helpsteer2", 500),
    "helpsteer2_complexity": ("helpsteer2", 500),
    "helpsteer": ("helpsteer", 1500),
    "ledgar": ("jev", 2500),
    "massive": ("jev", 1500),
    "banking77": ("jev", 1500),
    "clinc150": ("jev", 1500),
    "measuring_hate_speech": ("jev", 1500),
    "civil_comments": ("jev", 1000),
    # jev-bench has only 285 MMLU training rows; the remainder moves to CommonsenseQA.
    "mmlu": ("jev", 270),
    "commonsense_qa": ("commonsense_qa", 1230),
    "winogrande": ("winogrande", 1000),
}

# Proprietary synthetic families (kortexa-ai/shingi-synthetic): name -> training records.
SYNTHETIC = {"synth_policy": 3000, "synth_routing": 2000, "synth_taxonomy": 2000,
             "synth_state": 2500, "synth_rubric": 1000}
LONG_CONTEXT = "synth_longctx"

# Evaluation-only sources. Share-alike and unclear licenses stay here; StrategyQA
# and PAWS are permissive but deliberately kept out of distribution.
OOD = ("mnli", "chaosnli", "sst5", "sms_spam", "boolq", "fever_evidence", "arc_challenge",
       "paws", "stsb", "strategyqa_closed", "strategyqa_grounded")

TEST_PER_SOURCE = 200
OOD_PER_SOURCE = 200
DEV_PER_SOURCE = 50
CALIBRATION_PER_SOURCE = 50
PILOT_FRACTION = 1 / 3

# Upstream files for converted sources, pinned by revision. Checksums are frozen
# into the data manifest on first download.
UPSTREAM = {
    "helpsteer2": {"repo": "nvidia/HelpSteer2", "revision": "990b2711a36180dd19d9c94b8627844866f8982a",
                   "files": {"train": "train.jsonl.gz", "validation": "validation.jsonl.gz"}},
    "helpsteer": {"repo": "nvidia/HelpSteer", "revision": "3ca5d59c1bc1080af195b4254e7407db60b6f450",
                  "files": {"train": "train.jsonl.gz", "validation": "validation.jsonl.gz"}},
    "commonsense_qa": {"repo": "tau/commonsense_qa", "revision": "94630fe30dad47192a8546eb75f094926d47e155",
                       "files": {"train": "data/train-00000-of-00001.parquet",
                                 "validation": "data/validation-00000-of-00001.parquet"}},
    "winogrande": {"repo": "allenai/winogrande", "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                   "files": {"train": "winogrande_xl/train-00000-of-00001.parquet",
                             "validation": "winogrande_xl/validation-00000-of-00001.parquet"}},
}

# HelpSteer annotation scales (0-4), paraphrasing the HelpSteer2 annotation guidelines.
# Helpfulness and verbosity reuse the jev-bench wording so all HelpSteer data agrees.
SCALES = {
    "correctness": ("How correct and complete is `response` as an answer to `prompt`?", [
        "Completely incorrect: all or almost all information is wrong, or the response does not attempt the task.",
        "Mostly incorrect: some correct information, but most of it is wrong or key requirements are missing.",
        "Partially correct: a mix of correct and incorrect information, or notable omissions.",
        "Mostly correct: accurate on the whole, with a small error or omission.",
        "Perfectly correct: all information is accurate and every requirement of the prompt is covered."]),
    "coherence": ("How coherent and clear is `response`?", [
        "Completely incoherent: the response cannot be followed or understood.",
        "Mostly incoherent: hard to follow, with contradictions or disorganized reasoning in most of it.",
        "A little unclear: mostly followable, but some parts are confusing, repetitive or contradictory.",
        "Mostly coherent and clear: easy to follow, with minor lapses in clarity or style.",
        "Perfectly coherent and clear: consistent, well organized and easy to follow throughout."]),
    "complexity": ("How much expertise does it take to write `response`?", [
        "Basic: simple language anyone could produce and understand.",
        "Simple: straightforward language a young student could produce.",
        "Intermediate: language and ideas at a high-school level.",
        "Advanced: language and ideas that need college-level education.",
        "Expert: specialized vocabulary and deep domain expertise."]),
}
HELPSTEER_ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def finish(record):
    """Add the hashes that every Shingi split uses for isolation."""
    record["input_sha256"] = digest(canonical([record["state"], record["question"]]))
    record["state_sha256"] = digest(canonical(record["state"]))
    return record


def meta(origin, **extra):
    spec = UPSTREAM[origin]
    return {"hf_id": spec["repo"], "revision": spec["revision"], **extra}


def helpsteer_record(origin, source, split, index, row, attribute, jev_scales):
    if attribute in SCALES:
        instructions, criteria = SCALES[attribute]
    else:
        instructions, criteria = jev_scales[attribute]
    value = row[attribute]
    if not isinstance(value, int) or not 0 <= value <= 4:
        raise ValueError(f"{source}: invalid {attribute} {value!r}")
    return finish({"id": f"{source}/{split}/{index}/{attribute}", "source": source, "primitive": "score",
                   "split": split, "state": {"prompt": row["prompt"], "response": row["response"]},
                   "question": {"type": "score", "instructions": instructions, "criteria": list(criteria)},
                   "label": str(value), "soft_label": None,
                   "meta": meta(origin, attribute=attribute, upstream_split=split, row=index)})


def commonsense_qa_record(split, index, row):
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    if row["answerKey"] not in labels or len(set(texts)) != len(texts):
        return None  # ambiguous upstream row: duplicate option text or missing answer
    # Answer texts are the keys, so displayed letters never disagree with upstream letters.
    criteria = {text: None for text in texts}
    return finish({"id": f"commonsense_qa/{split}/{row['id']}", "source": "commonsense_qa", "primitive": "choice",
                   "split": split, "state": {"question": row["question"]},
                   "question": {"type": "choice", "instructions": "Which option is the most sensible answer to `question`?",
                                "criteria": criteria},
                   "label": texts[labels.index(row["answerKey"])], "soft_label": None,
                   "meta": meta("commonsense_qa", upstream_split=split, row=index)})


def winogrande_record(split, index, row):
    if row["answer"] not in ("1", "2") or row["option1"] == row["option2"] or row["sentence"].count("_") != 1:
        return None  # unusable upstream row
    return finish({"id": f"winogrande/{split}/{index}", "source": "winogrande", "primitive": "choice",
                   "split": split, "state": {"sentence": row["sentence"]},
                   "question": {"type": "choice", "instructions": "Which option correctly fills the blank (_) in `sentence`?",
                                "criteria": {"option1": row["option1"], "option2": row["option2"]}},
                   "label": "option" + row["answer"], "soft_label": None,
                   "meta": meta("winogrande", upstream_split=split, row=index)})
