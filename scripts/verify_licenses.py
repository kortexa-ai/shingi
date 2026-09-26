"""Freeze license evidence for data v3 and v3.1 before any fitting data is prepared.

The registry covers the fitting sources of every data profile. The output path
defaults to the v3 record; v3.1 writes `--output results/data-v3.1/licenses.json`.

Fetches each dataset card at its pinned revision, records the stated license
and the card checksum, and fails when a fitting source lacks an explicit
permissive license. Extra evidence documents cover cards without a license
field and the reasons for exclusions. Publisher-stated terms are provenance,
not a legal conclusion.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from shingi.sources_v3 import JEV_REPO, JEV_REVISION, OOD, PROFILES, license_source

PERMISSIVE = {"cc-by-4.0", "cc-by-3.0", "cc0-1.0", "mit", "apache-2.0"}
HF = "https://huggingface.co/datasets"

# name -> (role, card repository, revision, extra evidence [(url, required regex)], note)
REGISTRY = {
    # Fitting sources.
    "banking77": ("fit", "PolyAI/banking77", "90d4e2ee5521c04fc1488f065b8b083658768c57", [], "via jev-bench"),
    "clinc150": ("fit", "clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", [], "via jev-bench"),
    "mmlu": ("fit", "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe",
             [("https://raw.githubusercontent.com/hendrycks/test/4450500f923c49f1fb1dd3d99108a0bd9717b660/LICENSE", r"MIT License")],
             "via jev-bench"),
    "helpsteer2": ("fit", "nvidia/HelpSteer2", "990b2711a36180dd19d9c94b8627844866f8982a", [],
                   "helpfulness and verbosity via jev-bench; correctness, coherence and complexity converted here"),
    "measuring_hate_speech": ("fit", "ucberkeley-dlab/measuring-hate-speech", "5468f6e118396646b02a2f691e771f6b6d9502ea", [], "via jev-bench"),
    "civil_comments": ("fit", "google/civil_comments", "f2970eb3a55777454c94069077cc8d9b5866312d", [], "via jev-bench"),
    "go_emotions": ("fit", "google-research-datasets/go_emotions", "add492243ff905527e67aeb8b80c082af02207c3", [], "via jev-bench; moved from holdout"),
    "ledgar": ("fit", "coastalcph/lex_glue", "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c", [],
               "via jev-bench; moved from holdout; license stated for LexGLUE as a whole"),
    "massive": ("fit", "AmazonScience/massive", "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740", [],
                "via jev-bench from the mteb/amazon_massive_intent mirror"),
    "helpsteer": ("fit", "nvidia/HelpSteer", "3ca5d59c1bc1080af195b4254e7407db60b6f450", [], "converted here"),
    "commonsense_qa": ("fit", "tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", [],
                       "converted here; questions are crowd-written around ConceptNet (CC BY-SA) concepts"),
    "winogrande": ("fit", "allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                   [("https://raw.githubusercontent.com/allenai/winogrande/727e837f77521ef38bcc56df3b275c8da43f45af/README.md",
                     r"The dataset is licensed under CC-BY")],
                   "converted here; the Hub card has no license field, upstream README states CC-BY"),
    "gsm8k": ("fit", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", [],
              "data v3.1; converted here into strict judging records with computed labels"),
    # Evaluation-only sources.
    "mnli": ("ood", "nyu-mll/multi_nli", "da70db2af9d09693783c3320c4249840212ee221", [], "mixed incl. share-alike"),
    "chaosnli": ("ood", None, None, [], "CC BY-SA 4.0 per jev-bench manifest"),
    "sst5": ("ood", "SetFit/sst5", "e51bdcd8cd3a30da231967c1a249ba59361279a3", [], "no stated license"),
    "sms_spam": ("ood", "ucirvine/sms_spam", "cae486f927c250fe1d4a5b55f11357964ed1646c", [], "unknown license"),
    "boolq": ("ood", "google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5", [], "share-alike"),
    "fever_evidence": ("ood", "fever/fever", "2a74f2909caf2b8656343aeb8203e50bf84dcb56", [], "share-alike"),
    "arc_challenge": ("ood", "allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", [], "share-alike"),
    "paws": ("ood", "google-research-datasets/paws", "161ece9501cf0a11f3e48bd356eaa82de46d6a09", [],
             "permissive per jev-bench; kept out of distribution by design"),
    "stsb": ("ood", None, None, [], "CC BY-SA 4.0 per jev-bench manifest"),
    "strategyqa": ("ood", "ChilleD/StrategyQA", "705562638fe1d8ca6bb98c66fc8f94d45fda8c83", [],
                   "MIT; closed and grounded variants kept out of distribution by design"),
    # Excluded from all v3 use.
    "hellaswag": ("excluded", "Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                  [("https://raw.githubusercontent.com/github/dmca/master/2026/09/2026-09-14-wikihow.md", r"rowanz/hellaswag")],
                  "contexts from wikiHow; upstream repository blocked by a wikiHow DMCA notice on 2026-09-14"),
    "wanli": ("excluded", "alisawuffles/WANLI", "61c95318fd71c55b6ba355d76253254615f387ec", [],
              "generated from MNLI seed examples; would contaminate MNLI and ChaosNLI out-of-distribution evaluation"),
    "ultrafeedback": ("excluded", "openbmb/UltraFeedback", "40b436560ca83a8dba36114c22ab3c66e43f6d5e", [],
                      "labels generated by GPT-4; provider terms risk"),
    "openbookqa": ("excluded", "allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", [],
                   "card license unknown; repository code license does not clearly cover data"),
    "piqa": ("excluded", "ybisk/piqa", "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0", [], "card license unknown"),
    "ag_news": ("excluded", "fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", [], "card license unknown"),
    "yelp5": ("excluded", None, None, [], "Yelp dataset license restricts use; excluded from evaluation too"),
    "super_glue": ("excluded", "aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", [], "mixed per-task licenses"),
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "shingi-license-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def card_license(text):
    front = text.split("---", 2)[1] if text.startswith("---") else ""
    match = re.search(r"^license:[ \t]*(.*?)[ \t]*$", front, re.M)
    if not match:
        return []
    value = match.group(1).strip()
    if value:
        return [value.strip("'\"").lower()]
    items = re.findall(r"^license:[ \t]*\n((?:[ \t]*-[ \t]*.+\n)+)", front, re.M)
    return [x.strip("-'\" ").lower() for x in items[0].splitlines() if x.strip()] if items else []


def audit(name, spec):
    role, repo, revision, evidence, note = spec
    entry = {"role": role, "note": note, "repository": repo, "revision": revision}
    if repo:
        url = f"{HF}/{repo}/raw/{revision}/README.md"
        data = fetch(url)
        entry.update(card_url=url, card_sha256=hashlib.sha256(data).hexdigest(),
                     stated_license=card_license(data.decode("utf-8", "replace")))
    entry["evidence"] = []
    for url, pattern in evidence:
        data = fetch(url)
        found = re.search(pattern, data.decode("utf-8", "replace")) is not None
        entry["evidence"].append({"url": url, "sha256": hashlib.sha256(data).hexdigest(), "pattern": pattern, "found": found})
    stated = set(entry.get("stated_license", []))
    evidenced = bool(entry["evidence"]) and all(e["found"] for e in entry["evidence"])
    if role == "fit":
        entry["cleared"] = bool(stated) and stated <= PERMISSIVE or (not stated and evidenced)
        if stated and not stated <= PERMISSIVE:
            entry["cleared"] = False
    elif role == "excluded" and entry["evidence"]:
        entry["cleared"] = evidenced
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/data-v3/licenses.json"))
    args = parser.parse_args()
    fit_names = {license_source(n) for fit, _ in PROFILES.values() for n in fit}
    if fit_names != {n for n, s in REGISTRY.items() if s[0] == "fit"}:
        raise SystemExit("fitting registry does not match the sources_v3 profiles")
    ood_names = {"strategyqa" if n.startswith("strategyqa_") else n for n in OOD}
    if ood_names != {n for n, s in REGISTRY.items() if s[0] == "ood"}:
        raise SystemExit("evaluation registry does not match sources_v3.OOD")
    jev_card = fetch(f"{HF}/{JEV_REPO}/raw/{JEV_REVISION}/README.md")
    result = {"audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "policy": "Fit only on explicit CC BY, CC0, MIT or Apache-2.0 sources. Share-alike and unclear "
                        "licenses are evaluation-only; restricted or disputed sources are excluded. Raw data "
                        "is not redistributed.",
              "transformation": {"repository": JEV_REPO, "revision": JEV_REVISION,
                                 "card_sha256": hashlib.sha256(jev_card).hexdigest(),
                                 "terms": "Repackaging under upstream source licenses."},
              "synthetic": {"repository": "kortexa-ai/shingi-synthetic", "terms": "Proprietary to Kortexa AI; private."},
              "sources": {name: audit(name, spec) for name, spec in REGISTRY.items()}}
    failed = [n for n, e in result["sources"].items() if e["role"] in ("fit", "excluded") and e.get("cleared") is False]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({n: (e["role"], e.get("stated_license"), e.get("cleared")) for n, e in result["sources"].items()}, indent=1))
    if failed:
        raise SystemExit(f"license verification failed: {failed}")


if __name__ == "__main__":
    main()
