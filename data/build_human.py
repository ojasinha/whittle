"""Stream pre-ChatGPT human text, chunk into 60-300 word passages, filter, write JSONL.

Usage:
  python data/build_human.py --out data/raw/human_pilot.jsonl --n 20000
Each line: {id, doc_id, source, license, url, dump, text, n_words, n_paras}
Resumable: re-running appends only until --n chunks exist (counts existing lines, skips seen doc_ids).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset  # noqa: E402

from whittle.slop import slop_hits, words  # noqa: E402

CUTOFF_DUMPS = [  # all before ChatGPT release (2022-11-30)
    "CC-MAIN-2019-30", "CC-MAIN-2019-47", "CC-MAIN-2020-16", "CC-MAIN-2020-34", "CC-MAIN-2020-50",
    "CC-MAIN-2021-10", "CC-MAIN-2021-25", "CC-MAIN-2021-43", "CC-MAIN-2022-05", "CC-MAIN-2022-21",
    "CC-MAIN-2022-33", "CC-MAIN-2022-40",
]

SOURCES = {
    # name: (weight, license)
    "fineweb-edu": (0.40, "ODC-By-1.0 (HuggingFaceFW/fineweb-edu); underlying CommonCrawl terms"),
    "fineweb-blog": (0.30, "ODC-By-1.0 (HuggingFaceFW/fineweb); underlying CommonCrawl terms"),
    "wikipedia": (0.20, "CC-BY-SA-3.0 / GFDL (legacy-datasets/wikipedia 20220301.en)"),
    "gutenberg": (0.10, "Public-domain texts via sedthh/gutenberg_english (MIT dataset card)"),
}

BLOG_URL = re.compile(r"(blogspot\.|wordpress\.|medium\.com|substack\.com|/blog/|/posts?/|tumblr\.com|livejournal|/\d{4}/\d{2}/)", re.I)
BOILER = re.compile(r"(cookie|click here|subscribe|newsletter|all rights reserved|©|privacy policy|terms of (use|service)|"
                    r"sign up|log ?in|add to cart|free shipping|posted by|share this|related posts|comments? \(\d+\)|"
                    r"read more|advertisement|javascript|\bhttps?://)", re.I)


def stream(source: str, seed: int):
    rng = random.Random(seed)
    if source in ("fineweb-edu", "fineweb-blog"):
        repo = "HuggingFaceFW/fineweb-edu" if source == "fineweb-edu" else "HuggingFaceFW/fineweb"
        dumps = CUTOFF_DUMPS[:]
        rng.shuffle(dumps)
        for dump in dumps:
            files = f"hf://datasets/{repo}/data/{dump}/*.parquet"
            ds = load_dataset("parquet", data_files=files, split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
            for ex in islice(ds, 400_000):
                if source == "fineweb-edu" and ex.get("int_score", 3) < 3:
                    continue
                if source == "fineweb-blog" and not BLOG_URL.search(ex.get("url", "")):
                    continue
                yield {"doc_id": ex["id"], "url": ex.get("url"), "dump": dump, "text": ex["text"]}
    elif source == "wikipedia":
        ds = load_dataset("parquet", data_files="hf://datasets/legacy-datasets/wikipedia/data/20220301.en/*.parquet",
                          split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
        for ex in ds:
            yield {"doc_id": f"wiki-{ex['id']}", "url": ex.get("url"), "dump": "20220301", "text": ex["text"]}
    elif source == "gutenberg":
        ds = load_dataset("parquet", data_files="hf://datasets/sedthh/gutenberg_english/data/*.parquet",
                          split="train", streaming=True).shuffle(seed=seed, buffer_size=50)
        for ex in ds:
            txt = ex["TEXT"]
            meta = ex.get("METADATA") or "{}"
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
            title = str(meta.get("title", ""))[:40] or hashlib.md5(txt[:2000].encode()).hexdigest()[:10]
            if int(str(meta.get("issued", "1900"))[:4] or 1900) > 2021:
                continue
            # skip front matter; sample a few windows from the body
            body = txt[len(txt) // 10: -len(txt) // 10]
            paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
            joined = "\n\n".join(p for p in paras if p)
            for k in range(6):
                start = rng.randrange(max(1, len(joined) - 6000))
                seg = joined[start:start + 6000]
                seg = seg[seg.find("\n\n") + 2:] if "\n\n" in seg else seg
                yield {"doc_id": f"pg-{title}-{k}", "url": None, "dump": "gutenberg", "text": seg}
    else:
        raise ValueError(source)


CITATION = re.compile(r"(\(\d{4}\)[:.]|\d{4}\)?: ?\d+[–-]\d+|\bpp?\. ?\d|\bet al\.|\bISBN\b|\bdoi\b|\bvol\. ?\d|\[\^?\d*\]|\^)", re.I)


def good_para(p: str) -> bool:
    w = p.split()
    if len(w) < 12:
        return False
    if not re.match(r"[A-Z0-9\"'“‘(]", p):
        return False
    if CITATION.search(p) or len(re.findall(r"[.!?](?:\s|$)", p)) < 2:
        return False
    if BOILER.search(p):
        return False
    letters = sum(c.isalpha() for c in p)
    if letters / max(1, len(p)) < 0.72:
        return False
    if p.upper() == p:
        return False
    if not re.search(r"[.!?\"')\]]\s*$", p):
        return False
    return True


def good_chunk(text: str) -> bool:
    ws = words(text)
    n = len(ws)
    if n < 60 or n > 320:
        return False
    sents = re.split(r"(?<=[.!?])\s+", text)
    avg = n / max(1, len(sents))
    if not 7 <= avg <= 45:
        return False
    hits = sum(slop_hits(text).values())
    if hits / n * 100 > 1.5:  # already sloppy human text makes a bad target
        return False
    if len(set(w.lower() for w in ws)) / n < 0.4:  # repetitive
        return False
    return True


def chunk_doc(text: str, rng: random.Random, max_chunks: int = 2) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    paras = [p for p in paras if good_para(p)]
    chunks, cur, cur_w = [], [], 0
    multi = rng.random() < 0.2
    limit = 300 if multi else 220
    for p in paras:
        pw = len(p.split())
        if cur and (cur_w + pw > limit or (not multi and cur_w >= 60)):
            chunks.append("\n\n".join(cur))
            cur, cur_w = [], 0
        if pw > 300:
            continue
        cur.append(p)
        cur_w += pw
    if cur:
        chunks.append("\n\n".join(cur))
    chunks = [c for c in chunks if good_chunk(c)]
    rng.shuffle(chunks)
    return chunks[:max_chunks]


def norm_key(t: str) -> str:
    return hashlib.md5(re.sub(r"\W+", "", t.lower())[:200].encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--exclude", nargs="*", default=[], help="JSONL files whose doc_ids must not be reused (leakage guard)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    names = args.sources.split(",")
    weights = [SOURCES[s][0] for s in names]
    total_w = sum(weights)
    targets = {s: int(args.n * w / total_w) for s, w in zip(names, weights)}

    seen_docs, seen_keys, counts = set(), set(), {s: 0 for s in names}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seen_docs.add(r["doc_id"]); seen_keys.add(norm_key(r["text"])); counts[r["source"]] = counts.get(r["source"], 0) + 1
    for ex_path in args.exclude:
        with open(ex_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seen_docs.add(r["doc_id"]); seen_keys.add(norm_key(r["text"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for s in names:
            if counts[s] >= targets[s]:
                continue
            print(f"[{s}] need {targets[s] - counts[s]}", flush=True)
            for doc in stream(s, args.seed + len(seen_docs)):
                if doc["doc_id"] in seen_docs:
                    continue
                for i, c in enumerate(chunk_doc(doc["text"], rng)):
                    k = norm_key(c)
                    if k in seen_keys:
                        continue
                    seen_keys.add(k)
                    rec = {"id": f"{doc['doc_id']}#{i}", "doc_id": doc["doc_id"], "source": s, "license": SOURCES[s][1],
                           "url": doc["url"], "dump": doc["dump"], "text": c, "n_words": len(c.split()),
                           "n_paras": c.count("\n\n") + 1}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[s] += 1
                seen_docs.add(doc["doc_id"])
                if counts[s] % 500 == 0:
                    f.flush()
                if counts[s] >= targets[s]:
                    break
            print(f"[{s}] done: {counts[s]}", flush=True)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
