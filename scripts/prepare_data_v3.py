"""Freeze data v3: natural and synthetic training, development, calibration and locked evaluation.

Run on the host that holds earlier Shingi artifacts. Every earlier project record
is excluded from new evaluation splits by ID and state hash. Training also
excludes every public validation/test state, selected or not, and all earlier
evaluation states. No model or GPU is used; raw data stays in ignored artifacts.

`--profile v3` (the default) reproduces the frozen v3 build and its manifests.
`--profile v3.1` uses the v3.1 mix from sources_v3.PROFILES, adds GSM8K judging
and balances the calibration split's yes/no records by source and gold label.
Its manifests keep the `v3-pilot` and `v3-full` training profile names, so the
trainer applies the v3 recipe, and add `data_version: v3.1`.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from prepare_benchmark import canonical, decode_row, digest, read_jsonl, write_jsonl
from prepare_training_data import training_prompt
from shingi.decision import describe
from shingi.sources_v3 import (CALIBRATION_PER_SOURCE, DEV_PER_SOURCE, HELPSTEER_ATTRIBUTES, JEV_REPO, JEV_REVISION,
                               LONG_CONTEXT, OOD, OOD_PER_SOURCE, PILOT_FRACTION, PROFILES, TEST_PER_SOURCE, UPSTREAM,
                               YES_NO_GOLD_YES, YES_NO_MIN_RECORDS, YES_NO_MIN_SOURCES, YES_NO_PER_LABEL,
                               commonsense_qa_record, finish, gsm8k_judge_pool, helpsteer_record, license_source,
                               winogrande_record)

SEED = "shingi-data-v3-20260924"
TRAIN_PROMPT_CHARS = 7600  # about 1,900 tokens; tokenization later enforces 2,048 exactly
EVAL_STATE_CHARS = 40000   # keeps evaluation prompts well inside the 16K context
NEAR_DUPLICATE = .2        # shared 13-gram fraction; boilerplate stays, reworded duplicates go
TRAIN_FILES = {"train.jsonl", "train-prompts.jsonl", "tokenized.jsonl"}


def synth_families(synthetic):
    return [name.removeprefix("synth_") for name in synthetic] + [LONG_CONTEXT.removeprefix("synth_")]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as response:
            data = response.read()
        with dest.open("xb") as stream:
            stream.write(data)
    return {"url": url, "sha256": sha256(dest), "bytes": dest.stat().st_size}


def prior_records(roots):
    """IDs and state hashes of every earlier split file; raw downloads are skipped."""
    seen = {"train": (set(), set()), "eval": (set(), set())}
    files = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*.jsonl")):
            if "raw" in path.parts:
                continue
            kind = "train" if path.name in TRAIN_FILES else "eval"
            ids, states = seen[kind]
            for row in read_jsonl(path):
                if not isinstance(row, dict):
                    continue
                for key in ("id", "record_id"):
                    if isinstance(row.get(key), str):
                        ids.add(row[key])
                if isinstance(row.get("state_sha256"), str):
                    states.add(row["state_sha256"])
                elif "state" in row:
                    states.add(digest(canonical(parse_state(row["state"]))))
            files[str(path)] = {"kind": kind, "sha256": sha256(path)}
    return seen, files


def parse_state(state):
    """jev-bench encodes states as JSON strings; other sources store plain text."""
    if isinstance(state, str):
        try:
            return json.loads(state)
        except json.JSONDecodeError:
            return state
    return state


def prompt_group(row):
    """HelpSteer states pair one prompt with several responses; a prompt is one group."""
    state = row["state"]
    return digest(canonical(state["prompt"])) if isinstance(state, dict) and "prompt" in state else None


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text, n=13):
    tokens = words(text)
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def record_text(row):
    return describe(row["state"])


def prompt_chars(row):
    return len(training_prompt(row, 0, SEED)["prompt"])


def take(rows, count, ids, states, tag, keep=lambda row: True):
    """Up to count rows in deterministic hash order; chosen IDs and states join the exclusion sets."""
    chosen = []
    for row in sorted(rows, key=lambda r: digest(SEED + "|" + tag + "|" + r["id"])):
        if len(chosen) == count:
            break
        if row["id"] in ids or row["state_sha256"] in states or not keep(row):
            continue
        chosen.append(row)
        ids.add(row["id"])
        states.add(row["state_sha256"])
    return chosen


def select(rows, count, ids, states, tag, keep=lambda row: True):
    chosen = take(rows, count, ids, states, tag, keep)
    if len(chosen) != count:
        raise ValueError(f"{tag}: need {count}, found {len(chosen)}")
    return chosen


def yes_no(row):
    return row["question"]["type"] == "noul"


def balanced_yes_no(pools, per_label, ids, states, tag):
    """Stratify yes/no records by source with equal gold yes and no in each source.

    pools maps source -> (rows, keep). Each source gives min(per_label, available yes,
    available no) records of each label, so a source short of one label still
    contributes a balanced share.
    """
    chosen, composition = [], {}
    for source, (rows, keep) in pools.items():
        free = [r for r in rows if yes_no(r) and keep(r) and r["id"] not in ids and r["state_sha256"] not in states]
        groups = {label: [r for r in free if r["label"] == label] for label in ("1", "0")}
        available = {label: len({r["state_sha256"] for r in group}) for label, group in groups.items()}
        if not any(available.values()):
            continue
        count = min(per_label, *available.values())
        picked = {label: take(group, count, ids, states, f"{source}/{tag}/{label}") for label, group in groups.items()}
        count = min(map(len, picked.values()))
        rows = picked["1"][:count] + picked["0"][:count]
        chosen += rows
        composition[source] = {"records": len(rows), "gold_yes": count,
                               "available_yes": available["1"], "available_no": available["0"]}
    return chosen, composition


def check_yes_no(records, composition, minimum=YES_NO_MIN_RECORDS, share=YES_NO_GOLD_YES, sources=YES_NO_MIN_SOURCES):
    """The calibration yes/no gate; returns the summary recorded in the manifest."""
    yes = sum(r["label"] == "1" for r in records)
    used = sorted(s for s, c in composition.items() if c["records"])
    summary = {"records": len(records), "gold_yes": yes, "gold_yes_share": yes / len(records) if records else 0,
               "sources": len(used), "per_label_cap": YES_NO_PER_LABEL, "by_source": composition}
    if len(records) < minimum or not share[0] <= summary["gold_yes_share"] <= share[1] or len(used) < sources:
        raise ValueError(f"calibration yes/no pool fails its gate: {len(records)} records, "
                         f"{summary['gold_yes_share']:.1%} gold yes, {len(used)} sources")
    return summary


def jev_pools(raw, sources, files):
    pools = {}
    for source in sources:
        for split in ("train", "validation", "test"):
            if source == "chaosnli" and split != "test":
                continue
            relative = f"data/{source}/{split}.jsonl"
            url = f"https://huggingface.co/datasets/{JEV_REPO}/resolve/{JEV_REVISION}/{relative}"
            files["jev/" + relative] = download(url, raw / "jev" / relative)
            pools[source, split] = [decode_row(r) for r in read_jsonl(raw / "jev" / relative)]
    return pools


def upstream_rows(raw, origin, split, files):
    spec = UPSTREAM[origin]
    name = spec["files"][split]
    dest = raw / origin / name
    files[f"{origin}/{name}"] = download(f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{name}", dest)
    if name.endswith(".jsonl.gz"):
        with gzip.open(dest, "rt") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    import pyarrow.parquet as pq
    return pq.read_table(dest).to_pylist()


def converted_pools(raw, pools, files, fit, stats):
    """Converted sources: (source, split) -> records, split in train/validation."""
    scales = {}
    for attribute in ("helpfulness", "verbosity"):
        question = pools["helpsteer2_" + attribute, "validation"][0]["question"]
        scales[attribute] = (question["instructions"], question["criteria"])
    out = {}
    # jev-bench already uses almost all of HelpSteer2 validation. Converted attributes
    # evaluate on a fixed quarter of the HelpSteer2 training states that jev-bench never used.
    jev_states = {r["state_sha256"] for (source, _), rows in pools.items() if source.startswith("helpsteer2_") for r in rows}
    held = lambda r: r["state_sha256"] not in jev_states and (r["split"] == "validation" or int(r["state_sha256"], 16) % 4 == 0)
    hs2 = {split: upstream_rows(raw, "helpsteer2", split, files) for split in ("train", "validation")}
    for attribute in ("correctness", "coherence", "complexity"):
        source = "helpsteer2_" + attribute
        rows = [helpsteer_record("helpsteer2", source, split, i, row, attribute, scales)
                for split in ("train", "validation") for i, row in enumerate(hs2[split])]
        out[source, "validation"] = [r for r in rows if held(r)]
        out[source, "train"] = [r for r in rows if r["split"] == "train" and not held(r)]
    for split in ("train", "validation"):
        rows = upstream_rows(raw, "helpsteer", split, files)
        out["helpsteer", split] = [
            helpsteer_record("helpsteer", "helpsteer", split, i, row,
                             HELPSTEER_ATTRIBUTES[int(digest(SEED + f"|helpsteer|{split}|{i}"), 16) % 5], scales)
            for i, row in enumerate(rows)]
        for source, convert in (("commonsense_qa", commonsense_qa_record), ("winogrande", winogrande_record)):
            records = [convert(split, i, row) for i, row in enumerate(upstream_rows(raw, source, split, files))]
            out[source, split] = [r for r in records if r is not None]
        if "gsm8k_judge" in fit:
            records, skipped = gsm8k_judge_pool(split, upstream_rows(raw, "gsm8k", split, files), SEED)
            out["gsm8k_judge", split] = records
            stats[split] = {"records": len(records), "skipped": sum(skipped.values()),
                            "variants": dict(sorted(Counter(r["meta"]["variant"] for r in records).items())),
                            "skip_reasons": dict(sorted(skipped.items()))}
    return out


def synthetic_splits(directory, synthetic):
    manifest = json.loads((directory / "manifest.json").read_text())
    splits = {}
    for family in synth_families(synthetic):
        for path in sorted((directory / family).glob("*.jsonl")):
            recorded = manifest["files"].get(f"{family}/{path.name}", {}).get("sha256")
            if recorded != sha256(path):
                raise ValueError(f"synthetic file checksum differs: {path}")
            # Synthetic states are stored as given (text or objects), never JSON-encoded strings.
            splits["synth_" + family, path.stem] = [finish(r) for r in read_jsonl(path)]
    return manifest, splits


def training_prompts(rows):
    prompts = [training_prompt(row, 0, SEED) for row in rows]
    return sorted(prompts, key=lambda p: digest(SEED + "|order|" + p["id"]))


def write_profile(directory, profile, train, dev, calibration, parent, extra=None):
    directory.mkdir()
    write_jsonl(directory / "train-prompts.jsonl", training_prompts(train))
    write_jsonl(directory / "dev.jsonl", dev)
    write_jsonl(directory / "calibration.jsonl", calibration)
    by_source = Counter(r["source"] for r in train)
    synthetic = sum(v for k, v in by_source.items() if k.startswith("synth_"))
    manifest = {"profile": profile, "seed": SEED, "parent_manifest_sha256": parent,
                "train_sources": sorted(by_source), "train_by_source": dict(sorted(by_source.items())),
                "train_records": len(train), "synthetic_record_share": synthetic / len(train),
                "choice_order": "one seeded random order per record; inference sorts keys",
                "max_context": 2048, **(extra or {})}
    for name in ("train-prompts", "dev", "calibration"):
        manifest[name + "_sha256"] = sha256(directory / (name + ".jsonl"))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(args):
    fit, synthetic = PROFILES[args.profile]
    if None in synthetic.values():
        raise SystemExit(f"{args.profile}: synthetic family counts are not set yet (TODO #11)")
    v31 = args.profile == "v3.1"
    licenses_path = args.licenses or Path(f"results/data-{args.profile}/licenses.json")
    licenses = json.loads(licenses_path.read_text())
    blocked = [n for n, e in licenses["sources"].items() if e["role"] == "fit" and not e.get("cleared")]
    blocked += [n for n in sorted({license_source(s) for s in fit}) if not licenses["sources"].get(n, {}).get("cleared")]
    if blocked:
        raise SystemExit(f"uncleared fitting sources: {blocked}")
    if args.output.exists():
        raise SystemExit("preserve existing dataset; choose a new output directory")
    raw = args.cache
    files = {}
    prior, prior_files = prior_records(args.prior)
    jev_sources = sorted({s for s, (origin, _) in fit.items() if origin == "jev"} | set(OOD))
    pools = jev_pools(raw, jev_sources, files)
    gsm8k_stats = {}
    pools.update(converted_pools(raw, pools, files, fit, gsm8k_stats))
    synth_manifest, synth = synthetic_splits(args.synthetic, synthetic)

    def eval_pool(source, split):
        # Converted sources hide or lack official test labels; their validation split is the evaluation pool.
        if fit.get(source, ("jev",))[0] == "jev":
            return pools[source, split]
        return pools[source, "validation"]

    keep_eval = lambda row: len(record_text(row)) <= EVAL_STATE_CHARS
    ids = set(prior["train"][0] | prior["eval"][0])
    states = set(prior["train"][1] | prior["eval"][1])
    splits = {"test": [], "ood": [], "dev": [], "calibration": [], "transfer": [], "train": []}
    for source in fit:
        splits["test"] += select(eval_pool(source, "test"), TEST_PER_SOURCE, ids, states, source + "/test", keep_eval)
    for source in OOD:
        splits["ood"] += select(pools[source, "test"], OOD_PER_SOURCE, ids, states, source + "/ood", keep_eval)
    yes_no_summary = None
    if v31:
        # Calibration yes/no records are chosen first, so development cannot use up the scarce
        # positives (39 of 500 in the Civil Comments validation pool). Development keeps the v3 selection.
        yes_no_pools = {s: (eval_pool(s, "validation"), keep_eval) for s in fit}
        yes_no_pools |= {s: (rows, lambda row: True) for (s, split), rows in sorted(synth.items())
                         if split == "calibration" and s != LONG_CONTEXT}
        calibration_yes_no, composition = balanced_yes_no(yes_no_pools, YES_NO_PER_LABEL, ids, states, "calibration")
        yes_no_summary = check_yes_no(calibration_yes_no, composition)
        splits["calibration"] += calibration_yes_no
    other = (lambda row: keep_eval(row) and not yes_no(row)) if v31 else keep_eval
    for source in fit:
        pool = eval_pool(source, "validation")
        splits["dev"] += select(pool, DEV_PER_SOURCE, ids, states, source + "/dev", keep_eval)
        count = CALIBRATION_PER_SOURCE if not v31 or any(not yes_no(r) for r in pool) else 0
        splits["calibration"] += select(pool, count, ids, states, source + "/calibration", other)
    for (source, split), rows in sorted(synth.items()):
        target = {"dev": "dev", "calibration": "calibration", "transfer": "transfer"}.get(split)
        if target == "calibration" and v31:
            rows = [r for r in rows if not yes_no(r)]
        if target and source != LONG_CONTEXT:
            splits[target] += select(rows, len(rows), ids, states, f"{source}/{split}")

    # Training may reuse earlier training records, never any evaluation state.
    forbidden_ids = set(prior["eval"][0]) | {r["id"] for s in splits.values() for r in s}
    forbidden_states = set(prior["eval"][1]) | {r["state_sha256"] for s in splits.values() for r in s}
    for (source, split), rows in pools.items():
        if split != "train":
            forbidden_ids |= {r["id"] for r in rows}
            forbidden_states |= {r["state_sha256"] for r in rows}
    # 13-gram contamination: synthetic data must not overlap evaluation text at all; natural
    # records are dropped as near duplicates above NEAR_DUPLICATE and otherwise only reported.
    evaluation_grams = set()
    for name in ("test", "ood", "dev", "calibration"):
        for row in splits[name]:
            if not row["source"].startswith("synth_"):
                evaluation_grams |= ngrams(record_text(row))
    for path in args.external:
        for row in read_jsonl(path):
            if "state" in row:
                evaluation_grams |= ngrams(describe(row["state"]))
    # A training response must not share its prompt with any evaluation record.
    forbidden_prompts = {prompt_group(r) for s in splits.values() for r in s} | {
        prompt_group(r) for (source, split), rows in pools.items() if split != "train" for r in rows}
    forbidden_prompts.discard(None)
    reused = 0
    per_source = {}
    def keep_train(row):
        if prompt_group(row) in forbidden_prompts or prompt_chars(row) > TRAIN_PROMPT_CHARS:
            return False
        grams = ngrams(record_text(row))
        return not grams or len(grams & evaluation_grams) / len(grams) <= NEAR_DUPLICATE
    for source, (_, count) in fit.items():
        chosen = select(pools[source, "train"], count, forbidden_ids, forbidden_states, source + "/train", keep_train)
        reused += sum(r["id"] in prior["train"][0] for r in chosen)
        per_source[source] = chosen
    for source, count in synthetic.items():
        rows = synth[source, "train"]
        if len(rows) < count:
            raise ValueError(f"{source}: need {count} synthetic training records, found {len(rows)}")
        per_source[source] = select(rows, count, forbidden_ids, forbidden_states, source + "/train", keep_train)
    splits["train"] = [r for rows in per_source.values() for r in rows]
    pilot = [r for rows in per_source.values() for r in rows[:round(len(rows) * PILOT_FRACTION)]]

    long_splits = {}
    for split, name in (("train", "train"), ("dev", "dev"), ("transfer", "eval")):
        long_splits[name] = select(synth[LONG_CONTEXT, split], len(synth[LONG_CONTEXT, split]), forbidden_ids,
                                   forbidden_states, f"{LONG_CONTEXT}/{split}")

    # Isolation: no ID or state appears in two splits; nothing from earlier evaluation enters training.
    everything = dict(splits, **{"longctx-" + k: v for k, v in long_splits.items()})
    overlap = {}
    names = sorted(everything)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = ({r["state_sha256"] for r in everything[a]} & {r["state_sha256"] for r in everything[b]})
            overlap[f"{a}|{b}"] = len(shared)
    if any(overlap.values()):
        raise ValueError(f"split overlap: {[k for k, v in overlap.items() if v]}")
    prior_eval_hits = sum(r["state_sha256"] in prior["eval"][1] or r["id"] in prior["eval"][0] for r in splits["train"])
    if prior_eval_hits:
        raise ValueError("earlier evaluation record entered training")

    contamination = {}
    for source, rows in list(per_source.items()) + [(LONG_CONTEXT, long_splits["train"])]:
        hits = [r["id"] for r in rows if ngrams(record_text(r)) & evaluation_grams]
        contamination[source] = {"records": len(rows), "13gram_hits": len(hits), "examples": hits[:5]}
    synthetic_hits = {k: v["13gram_hits"] for k, v in contamination.items() if k.startswith("synth_") and v["13gram_hits"]}
    if synthetic_hits:
        raise ValueError(f"synthetic contamination: {synthetic_hits}")

    args.output.mkdir(parents=True)
    for name, rows in splits.items():
        write_jsonl(args.output / f"{name}.jsonl", rows)
    (args.output / "longctx").mkdir()
    for name, rows in long_splits.items():
        write_jsonl(args.output / "longctx" / f"{name}.jsonl", rows)
    (args.output / "contamination.json").write_text(json.dumps(contamination, indent=2) + "\n")
    manifest = {"seed": SEED, "licenses_sha256": sha256(licenses_path),
                "transformation": {"repository": JEV_REPO, "revision": JEV_REVISION},
                "synthetic": {"repository": "kortexa-ai/shingi-synthetic", "revision": args.synthetic_revision,
                              "manifest": synth_manifest},
                "downloads": files, "prior_files": prior_files,
                "counts": {name: len(rows) for name, rows in everything.items()},
                "by_source": {name: dict(sorted(Counter(r["source"] for r in rows).items())) for name, rows in everything.items()},
                "pilot_records": len(pilot),
                "synthetic_record_share": sum(r["source"].startswith("synth_") for r in splits["train"]) / len(splits["train"]),
                "reused_earlier_training_records": reused, "overlap_state_sha256": overlap,
                "limits": {"train_prompt_chars": TRAIN_PROMPT_CHARS, "eval_state_chars": EVAL_STATE_CHARS},
                "pretraining_contamination": "Unknown for natural sources."}
    extra = None
    if v31:
        extra = {"data_version": "v3.1"}
        manifest.update(profile="v3.1", fit=dict(fit), synthetic_counts=dict(synthetic),
                        gsm8k_judge=gsm8k_stats, calibration_yes_no=yes_no_summary)
    for name in splits:
        manifest[name + "_sha256"] = sha256(args.output / f"{name}.jsonl")
    for name in long_splits:
        manifest["longctx-" + name + "_sha256"] = sha256(args.output / "longctx" / f"{name}.jsonl")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    parent = sha256(args.output / "manifest.json")
    profiles = {"pilot": write_profile(args.output / "pilot", "v3-pilot", pilot, splits["dev"], splits["calibration"], parent, extra),
                "full": write_profile(args.output / "full", "v3-full", splits["train"], splits["dev"], splits["calibration"], parent, extra)}
    print(json.dumps({"counts": manifest["counts"], "synthetic_record_share": manifest["synthetic_record_share"],
                      "pilot": profiles["pilot"]["train_records"], "reused": reused,
                      "natural_13gram_hits": {k: v["13gram_hits"] for k, v in contamination.items() if not k.startswith("synth_")}},
                     indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="raw download cache, reused across builds")
    parser.add_argument("--synthetic", type=Path, required=True, help="kortexa-ai/shingi-synthetic data directory")
    parser.add_argument("--synthetic-revision", required=True)
    parser.add_argument("--prior", type=Path, nargs="+", required=True, help="earlier artifact roots")
    parser.add_argument("--external", type=Path, nargs="*", default=[], help="external benchmark record files")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v3")
    parser.add_argument("--licenses", type=Path, help="default: results/data-<profile>/licenses.json")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
