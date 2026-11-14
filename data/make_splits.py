"""Assign human chunks to splits (by document) and to generators.

  python data/make_splits.py --inp data/raw/human_pilot.jsonl --outdir data/gen/pilot

Writes one JSONL per generator job: <outdir>/to_<gen>.jsonl (each line = human chunk + split field).
* train/val docs: one training generator each (alternating by hash) -> in-distribution pairs
* test docs: ALL training generators + the held-out generator     -> test_in / test_heldout
"""
import argparse
import hashlib
import json
import os
from collections import Counter

TRAIN_GENS = ["qwen3-30b-a3b", "mistral-small-3.2"]
HELDOUT_GENS = ["gpt-oss-20b"]


def h(s, salt=""):
    return int(hashlib.sha1((salt + s).encode()).hexdigest(), 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.03)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    jobs = {g: [] for g in TRAIN_GENS + HELDOUT_GENS}
    counts = Counter()
    for line in open(args.inp, encoding="utf-8"):
        r = json.loads(line)
        u = (h(r["doc_id"], "split") % 10_000) / 10_000
        split = "test" if u < args.test_frac else "val" if u < args.test_frac + args.val_frac else "train"
        r["split"] = split
        if split == "test":
            for g in TRAIN_GENS + HELDOUT_GENS:
                jobs[g].append(r)
                counts[(split, g)] += 1
        else:
            g = TRAIN_GENS[h(r["id"], "gen") % len(TRAIN_GENS)]
            jobs[g].append(r)
            counts[(split, g)] += 1
    for g, rows in jobs.items():
        with open(os.path.join(args.outdir, f"to_{g}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(counts.items()):
        print(k, v)


if __name__ == "__main__":
    main()
