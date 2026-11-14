"""Filter (slop -> human) pairs.

  python data/filter.py --inp data/gen/pilot_out/pairs_*.jsonl --outdir data/pairs/pilot

Checks (each rejection reason is counted and logged):
  * empty / refusal / meta-commentary output
  * length ratio slop_words / human_words in [0.6, 2.5]
  * numbers (incl. dates, years, %) in human all appear in slop, and slop adds none
  * named entities (spaCy) in human appear in slop (fuzzy: entity text found in slop), allowing <=1 miss for long texts
  * code spans, URLs and inline code in human preserved verbatim in slop
  * NLI entailment (DeBERTa-v3 MNLI/FEVER/ANLI), both directions scored and stored:
      slop => human >= --nli-s2h (default 0.8): the target must be fully supported by the input (no hallucination)
      human => slop >= --nli-h2s (default 0.0 = log only): slop adds fluff claims on purpose, so this direction is
      naturally low; added *facts* are caught by the number/entity checks instead.
  * near-duplicate slop texts (MinHash, Jaccard >= 0.8) removed
Outputs <outdir>/{train,val,test_in,test_heldout}.jsonl + filter_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whittle.facts import entities_preserved, extract_numbers, protected_spans  # noqa: E402

HELDOUT = {"gpt-oss-20b"}
REFUSAL = re.compile(r"^(i'?m sorry|i cannot|i can'?t|as an ai|sure, here|here'?s the rewritten|here is the rewritten|rewritten (text|passage))", re.I)


def basic_checks(r, nlp=None):
    s, h = r["slop"], r["human"]
    if not s or len(s.split()) < 20:
        return "empty"
    if REFUSAL.search(s.strip()[:80]):
        return "refusal_or_meta"
    ratio = len(s.split()) / max(1, len(h.split()))
    if not 0.6 <= ratio <= 2.5:
        return "length_ratio"
    nh, ns = extract_numbers(h), extract_numbers(s)
    if not nh <= ns:
        return "number_dropped"
    if ns - nh:
        return "number_added"
    for span in protected_spans(h):
        if span not in s:
            return "protected_span"
    if nlp is not None and not entities_preserved(h, s, nlp):
        return "entity_dropped"
    return None


def minhash_dedup(rows, threshold=0.8):
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    keep = []
    for i, r in enumerate(rows):
        m = MinHash(num_perm=64)
        toks = r["slop"].lower().split()
        for j in range(len(toks) - 2):
            m.update(" ".join(toks[j:j + 3]).encode())
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        keep.append(r)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nli-s2h", type=float, default=0.8)
    ap.add_argument("--nli-h2s", type=float, default=0.0)
    ap.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--no-ner", action="store_true")
    args = ap.parse_args()

    files = [f for pat in args.inp for f in glob.glob(pat)]
    rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
    print(f"{len(rows)} pairs from {len(files)} files")
    nlp = None
    if not args.no_ner:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    reasons = Counter()
    survivors = []
    for r in rows:
        why = basic_checks(r, nlp)
        if why:
            reasons[why] += 1
        else:
            survivors.append(r)
    print("after rule checks:", len(survivors), dict(reasons))

    if not args.no_nli and survivors:
        from whittle.nli import NLIScorer
        scorer = NLIScorer(args.nli_model)
        fwd = scorer.entail([r["human"] for r in survivors], [r["slop"] for r in survivors])
        bwd = scorer.entail([r["slop"] for r in survivors], [r["human"] for r in survivors])
        kept = []
        for r, a, b in zip(survivors, fwd, bwd):
            r["nli_h2s"], r["nli_s2h"] = round(a, 4), round(b, 4)
            if b >= args.nli_s2h and a >= args.nli_h2s:
                kept.append(r)
            else:
                reasons["nli"] += 1
        survivors = kept
        print("after NLI:", len(survivors))

    before = len(survivors)
    survivors = minhash_dedup(survivors)
    reasons["near_dup"] += before - len(survivors)

    os.makedirs(args.outdir, exist_ok=True)
    splits = Counter()
    outs = {k: open(os.path.join(args.outdir, f"{k}.jsonl"), "w", encoding="utf-8")
            for k in ("train", "val", "test_in", "test_heldout")}
    for r in survivors:
        sp = r.get("split", "train")
        if sp == "test":
            sp = "test_heldout" if r["generator"] in HELDOUT else "test_in"
        elif r["generator"] in HELDOUT:
            continue  # held-out generator never enters train/val
        outs[sp].write(json.dumps(r, ensure_ascii=False) + "\n")
        splits[(sp, r["generator"])] += 1
    for f in outs.values():
        f.close()
    by_gen = Counter(r["generator"] for r in rows)
    report = {"input": len(rows), "input_by_generator": dict(by_gen), "rejected": dict(reasons),
              "kept": len(survivors), "splits": {f"{a}/{b}": n for (a, b), n in sorted(splits.items())}}
    json.dump(report, open(os.path.join(args.outdir, "filter_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
