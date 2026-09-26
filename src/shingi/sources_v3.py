"""Data v3 source registry and converters for sources absent from jev-bench.

Every converter emits the jev-bench record schema used by earlier Shingi data:
id, source, primitive, split, state, question, label, soft_label and meta.
Licenses are recorded here and verified by scripts/verify_licenses.py; the
registry records provenance, not a legal conclusion.

Data versions are profiles. FIT and SYNTHETIC stay the frozen v3 mix; data v3.1
(issue #11) adds FIT_V31 and SYNTHETIC_V31 beside them, so the v3 build remains
reproducible with `--profile v3` and v3.1 is built with `--profile v3.1`.
"""
from collections import Counter
from fractions import Fraction
import ast
import hashlib
import json
import operator
import random
import re

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
# Sized for about 35% of exact Bonsai training tokens: synthetic prompts average 1.4x
# natural tokens, so this is about 24% of records. Generated pools hold more; selection
# takes a deterministic subset.
SYNTHETIC = {"synth_policy": 1750, "synth_routing": 1150, "synth_taxonomy": 1150,
             "synth_state": 1450, "synth_rubric": 600}
LONG_CONTEXT = "synth_longctx"

# Evaluation-only sources. Share-alike and unclear licenses stay here; StrategyQA
# and PAWS are permissive but deliberately kept out of distribution.
OOD = ("mnli", "chaosnli", "sst5", "sms_spam", "boolq", "fever_evidence", "arc_challenge",
       "paws", "stsb", "strategyqa_closed", "strategyqa_grounded")

# Data v3.1: less HelpSteer, plus strict GSM8K judging. Natural total 18,900.
FIT_V31 = {**FIT, "helpsteer2_helpfulness": ("jev", 700), "helpsteer2_verbosity": ("jev", 300),
           "helpsteer2_correctness": ("helpsteer2", 300), "helpsteer2_coherence": ("helpsteer2", 300),
           "helpsteer2_complexity": ("helpsteer2", 300), "helpsteer": ("helpsteer", 500),
           "gsm8k_judge": ("gsm8k", 1500)}
# Synthetic generator v1.1 (kortexa-ai/shingi-synthetic at 2a46d571). The proposal (judge 1,500,
# severity 1,000, arithmetic 1,000, policy 1,300, routing 900, taxonomy 700, state 1,300, rubric 400)
# measured 41% of exact Bonsai tokens; every family was scaled by 0.78 and rounded to 50s.
# Predicted from the probe tokenization: 35.15% full, 34.95% pilot.
SYNTHETIC_V31 = {"synth_policy": 1000, "synth_routing": 700, "synth_taxonomy": 550, "synth_state": 1000,
                 "synth_rubric": 300, "synth_judge": 1150, "synth_severity": 800, "synth_arithmetic": 800}
PROFILES = {"v3": (FIT, SYNTHETIC), "v3.1": (FIT_V31, SYNTHETIC_V31)}

TEST_PER_SOURCE = 200
OOD_PER_SOURCE = 200
DEV_PER_SOURCE = 50
CALIBRATION_PER_SOURCE = 50
PILOT_FRACTION = 1 / 3
# v3.1 draws fresh evaluation splits, and these pools have fewer free states than the v3
# per-source counts. HelpSteer2 helpfulness and verbosity rate the same responses, so they share
# 200 free test states and 151 free validation states; SMS Spam has 187 free test records.
EVAL_COUNTS_V31 = {("helpsteer2_helpfulness", "test"): 100, ("helpsteer2_verbosity", "test"): 100,
                   ("helpsteer2_helpfulness", "calibration"): 25, ("helpsteer2_verbosity", "calibration"): 25,
                   ("sms_spam", "ood"): 180}
# v3.1 calibration: yes/no records balanced by gold label within each source.
YES_NO_PER_LABEL = 40
YES_NO_MIN_RECORDS = 150
YES_NO_GOLD_YES = (.4, .6)
YES_NO_MIN_SOURCES = 3

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
    # GSM8K has no validation split; its test split is the evaluation pool.
    "gsm8k": {"repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866",
              "files": {"train": "main/train-00000-of-00001.parquet",
                        "validation": "main/test-00000-of-00001.parquet"}},
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


def license_source(name):
    """The license registry entry that covers a fitting source."""
    return "helpsteer2" if name.startswith("helpsteer2_") else "gsm8k" if name == "gsm8k_judge" else name


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


# GSM8K strict judging: the gold rationale, or one with a planted error; labels are computed.
GSM8K_VARIANTS = ("final_wrong", "step_wrong", "step_wrong_propagated")
GSM8K_INSTRUCTIONS = ("Does the response fully and correctly solve the problem? Judge every step and the final "
                      "answer; any arithmetic error, unsupported step or wrong final answer means no.")
GSM8K_CRITERIA = {"true": "The response is fully correct: every step is valid and the final answer is right.",
                  "false": "The response has an arithmetic error, an unsupported step or a wrong final answer."}
MARKUP = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
NUMBER = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|(?<![\d.,])\d+(?:\.\d+)?(?!\d)")
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


class SkipRow(ValueError):
    """A perturbation cannot be applied cleanly; the message is the recorded reason."""


def number(text):
    return Fraction(text.replace(",", ""))


def show(value, commas=False):
    """Decimal text for an exact value, at most two decimal places."""
    if value.denominator == 1:
        return f"{value.numerator:,}" if commas else str(value.numerator)
    if (value * 100).denominator != 1:
        raise SkipRow("non-terminating result")
    return f"{float(value):.2f}".rstrip("0")


def evaluate(expr):
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and (right.denominator != 1 or abs(right) > 12):
                raise SkipRow("unevaluable expression")
            return OPERATORS[type(node.op)](left, right)
        raise SkipRow("unevaluable expression")
    try:
        return walk(ast.parse(expr.replace(",", ""), mode="eval"))
    except (SyntaxError, ZeroDivisionError):
        raise SkipRow("unevaluable expression") from None


def gsm8k_parts(answer):
    """Split an upstream answer into the rationale body and the final answer text."""
    body, marker, final = answer.rstrip().rpartition("\n#### ")
    if final.strip().startswith("-"):
        raise SkipRow("negative final answer")  # five upstream rows; signed values are not parsed
    if not marker or "####" in body or not NUMBER.fullmatch(final.strip()):
        raise SkipRow("malformed final answer")
    return body, final.strip()


def gsm8k_steps(body):
    """Calculator steps: the markup span, its value and the visible result that follows it."""
    steps = []
    for match in MARKUP.finditer(body):
        visible = NUMBER.match(body, match.end())
        try:
            value = number(match.group(2).strip())
        except ValueError:
            value = None
        ok = visible is not None and value is not None and number(visible.group()) == value
        steps.append({"expr": match.group(1), "value": value, "start": match.start(),
                      "visible": (visible.start(), visible.end()) if ok else None})
    return steps


def render(body, final):
    text = MARKUP.sub("", body)
    if "<<" in text or ">>" in text or "####" in text:
        raise SkipRow("stray calculator markup")
    return text.rstrip() + f"\nAnswer: {final}"


def replace_numbers(text, mapping, start=0):
    """Replace every number token at or after start whose value is a mapping key."""
    def swap(match):
        value = number(match.group())
        if match.start() < start or value not in mapping:
            return match.group()
        return show(mapping[value], "," in match.group())
    return NUMBER.sub(swap, text)


def nearby(value, rng, avoid=()):
    """A plausible wrong value close to value, never negative when value is not."""
    deltas = sorted({1, 2, 3, 5, 10} | {d for d in (round(abs(value) * f) for f in (.1, .2, .5)) if d > 0})
    for _ in range(100):
        wrong = value + rng.choice(deltas) * rng.choice((1, -1))
        if wrong != value and wrong not in avoid and (wrong >= 0 or value < 0):
            return wrong
    raise SkipRow("no nearby wrong value")


def intermediate(steps, final, problem):
    """Candidate intermediate steps; raises with the first filter that leaves none."""
    if not steps:
        raise SkipRow("no calculator steps")
    candidates = [(i, s) for i, s in enumerate(steps[:-1]) if s["visible"]]
    if not candidates:
        raise SkipRow("no intermediate step")
    candidates = [(i, s) for i, s in candidates if s["value"] != final]
    if not candidates:
        raise SkipRow("every intermediate equals the final answer")
    given = {number(n) for n in NUMBER.findall(problem)}
    candidates = [(i, s) for i, s in candidates if s["value"] not in given]
    if not candidates:
        raise SkipRow("step value appears in the problem")
    return candidates


def propagate(body, steps, i, wrong, final, problem):
    """Change step i to wrong and recompute every later step that uses a changed value."""
    changed, affected = {steps[i]["value"]: wrong}, []
    for j in range(i + 1, len(steps)):
        step = steps[j]
        if not any(number(n) in changed for n in NUMBER.findall(step["expr"])):
            if step["value"] in changed:
                raise SkipRow("changed value is ambiguous")
            continue
        if not step["visible"]:
            raise SkipRow("unparsed later step")
        new = evaluate(replace_numbers(step["expr"], changed))
        show(new)
        # GSM8K answers are whole numbers; a fractional count would give the label away.
        if step["value"].denominator == 1 and new.denominator != 1:
            raise SkipRow("fractional propagated result")
        if new < 0 <= step["value"]:
            raise SkipRow("negative propagated result")
        if changed.get(step["value"], new) != new:
            raise SkipRow("changed value is ambiguous")
        changed[step["value"]] = new
        affected.append(j)
    # A changed value must not also stand for a given number or an earlier result.
    if {number(n) for n in NUMBER.findall(problem)} & changed.keys():
        raise SkipRow("changed value appears in the problem")
    if {number(n) for n in NUMBER.findall(body[:steps[i]["start"]])} & changed.keys():
        raise SkipRow("changed value is ambiguous")
    if changed.get(final, final) == final:
        raise SkipRow("final answer unchanged")
    start = steps[i]["visible"][0]
    text = body[:start] + replace_numbers(body[start:], changed)
    # Every recomputed step must be internally consistent; only step i carries the error.
    redone = gsm8k_steps(text)
    for j in affected:
        if not redone[j]["visible"] or evaluate(redone[j]["expr"]) != redone[j]["value"]:
            raise SkipRow("propagation check failed")
    return text, changed[final]


def gsm8k_variant(split, index, seed):
    """Half gold; the rest split evenly over the three planted-error variants."""
    h = int(digest(f"{seed}|gsm8k|{split}|{index}"), 16)
    return "gold" if h % 2 == 0 else GSM8K_VARIANTS[h // 2 % 3]


def gsm8k_judge_record(split, index, row, variant, seed):
    """One noul record per GSM8K row; raises SkipRow when the variant cannot apply cleanly."""
    body, final = gsm8k_parts(row["answer"])
    rng = random.Random(f"{seed}|gsm8k|{split}|{index}|{variant}")
    steps = gsm8k_steps(body)
    target = number(final)
    step_index = None
    if variant == "gold":
        response = render(body, final)
    elif variant == "final_wrong":
        wrong = nearby(target, rng)
        last = steps[-1] if steps else None
        if last and last["visible"] and last["value"] == target:
            step_index = len(steps) - 1
            start = last["visible"][0]
        else:
            start = body.rfind("\n") + 1
        response = render(body[:start] + replace_numbers(body[start:], {target: wrong}), show(wrong, "," in final))
    elif variant == "step_wrong":
        step_index, step = rng.choice(intermediate(steps, target, row["question"]))
        wrong = nearby(step["value"], rng, avoid={target})
        a, b = step["visible"]
        response = render(body[:a] + show(wrong, "," in body[a:b]) + body[b:], final)
    elif variant == "step_wrong_propagated":
        candidates = intermediate(steps, target, row["question"])
        rng.shuffle(candidates)
        reasons = []
        for step_index, step in candidates:
            try:
                text, new_final = propagate(body, steps, step_index, nearby(step["value"], rng, avoid={target}),
                                            target, row["question"])
                break
            except SkipRow as error:
                reasons.append(str(error))
        else:
            raise SkipRow(reasons[0])
        response = render(text, show(new_final, "," in final))
    else:
        raise ValueError(f"unknown GSM8K variant {variant!r}")
    return finish({"id": f"gsm8k_judge/{split}/{index}", "source": "gsm8k_judge", "primitive": "noul",
                   "split": split, "state": {"problem": row["question"], "response": response},
                   "question": {"type": "noul", "instructions": GSM8K_INSTRUCTIONS, "criteria": dict(GSM8K_CRITERIA)},
                   "label": "1" if variant == "gold" else "0", "soft_label": None,
                   "meta": meta("gsm8k", upstream_split={"train": "train", "validation": "test"}[split], row=index,
                                variant=variant, perturbed_step_index=step_index)})


def gsm8k_judge_pool(split, rows, seed):
    """Convert a GSM8K pool; returns the records and skip counts by variant and reason."""
    records, skipped = [], Counter()
    for index, row in enumerate(rows):
        variant = gsm8k_variant(split, index, seed)
        try:
            records.append(gsm8k_judge_record(split, index, row, variant, seed))
        except SkipRow as error:
            skipped[f"{variant}: {error}"] += 1
    return records, skipped
