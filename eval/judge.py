"""LLM-judge quality eval with free open-weight judges (vLLM on a CUDA GPU, or any OpenAI-compatible server).

  python eval/judge.py --runs runs/eval --systems identity rule ckpt_X ckpt_Y --set test_in \
      --judge Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --judge-tag qwen3-30b --backend vllm --limit 300

1. Absolute: each output scored 1-10 for coherence, style, fluency; weakest = min of the three.
   The "identity" system's outputs are the INPUT, so "ours vs input" is just a comparison of the two rows.
2. Pairwise (blind): for each item and each pair (ours vs other), both orders are asked; a win needs both orders
   to agree (position-debiased), otherwise tie.
Two judges from different families than the system under test are recommended (e.g. Qwen3-30B and gpt-oss-20b).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
from collections import defaultdict
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))

ABS_PROMPT = """You are an expert editor evaluating the quality of a piece of English writing.

Rate the text on three criteria, each an integer from 1 (very poor) to 10 (excellent):
- coherence: logical flow, clear structure, ideas connect, nothing contradictory or confusing.
- style: how well written it is: precise word choice, natural voice, varied rhythm, no filler, no clichés, no padding.
- fluency: grammatical, idiomatic, reads smoothly.

Judge only the writing quality. Do not reward length.

Text:
\"\"\"
{text}
\"\"\"

Respond with JSON only, e.g. {{"coherence": 7, "style": 6, "fluency": 8}}"""

PAIR_PROMPT = """You are an expert editor. Two versions of a text convey the same information.
Which version is better writing (clearer, more natural, more precise, free of filler and clichés)?

Version A:
\"\"\"
{a}
\"\"\"

Version B:
\"\"\"
{b}
\"\"\"

Answer with exactly one letter: A, B, or T (tie)."""


def parse_abs(s):
    m = re.search(r"\{.*?\}", s, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        v = {k: int(d[k]) for k in ("coherence", "style", "fluency")}
        if all(1 <= x <= 10 for x in v.values()):
            return v
    except Exception:  # noqa: BLE001
        pass
    return None


def parse_pair(s):
    s = s.strip().upper()
    m = re.search(r"\b([ABT])\b", s)
    return m.group(1) if m else "T"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "eval"))
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--ours", default=None, help="system tag compared pairwise against all others")
    ap.add_argument("--set", default="test_in")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--judge-tag", required=True)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "openai"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "judge"))
    args = ap.parse_args()

    from slopify import OpenAIBackend, VLLMBackend, clean_output

    outs = {}
    for s in args.systems:
        rows = [json.loads(l) for l in open(os.path.join(args.runs, s, f"{args.set}.jsonl"), encoding="utf-8")]
        outs[s] = {r["id"]: r["out"] for r in rows}
    ids = sorted(set.intersection(*(set(v) for v in outs.values())))
    random.Random(0).shuffle(ids)
    ids = ids[:args.limit]
    if "reference" not in outs:  # the human reference is a useful ceiling
        rows = [json.loads(l) for l in open(os.path.join(args.runs, args.systems[0], f"{args.set}.jsonl"), encoding="utf-8")]
        if rows and rows[0].get("ref"):
            outs["reference"] = {r["id"]: r["ref"] for r in rows}

    if args.backend == "vllm":
        be = VLLMBackend(args.judge, 0.0, 512 if "gpt-oss" in args.judge else 64, max_model_len=4096,
                         gpu_memory_utilization=args.gpu_mem)
    else:
        be = OpenAIBackend(args.base_url, args.judge, 0.0, 512, 8)

    systems = list(outs)
    prompts, keys = [], []
    for s in systems:
        for i in ids:
            prompts.append(ABS_PROMPT.format(text=outs[s][i]))
            keys.append((s, i))
    res = be.generate(prompts)
    absolute = defaultdict(list)
    for (s, i), r in zip(keys, res):
        v = parse_abs(clean_output(r))
        if v:
            v["weakest"] = min(v.values())
            absolute[s].append(v)

    ours = args.ours or args.systems[-1]
    pair_prompts, pair_keys = [], []
    for other in systems:
        if other == ours:
            continue
        for i in ids:
            a, b = outs[ours][i], outs[other][i]
            pair_prompts += [PAIR_PROMPT.format(a=a, b=b), PAIR_PROMPT.format(a=b, b=a)]
            pair_keys.append((other, i))
    pres = [parse_pair(clean_output(x)) for x in be.generate(pair_prompts)]
    pairwise = defaultdict(lambda: {"win": 0, "loss": 0, "tie": 0})
    for k, (other, i) in enumerate(pair_keys):
        first, second = pres[2 * k], pres[2 * k + 1]  # first: ours=A ; second: ours=B
        if first == "A" and second == "B":
            pairwise[other]["win"] += 1
        elif first == "B" and second == "A":
            pairwise[other]["loss"] += 1
        else:
            pairwise[other]["tie"] += 1

    summary = {"judge": args.judge_tag, "set": args.set, "n_items": len(ids), "ours": ours, "absolute": {}, "pairwise": {}}
    for s, vs in absolute.items():
        summary["absolute"][s] = {k: round(mean(v[k] for v in vs), 3) for k in ("coherence", "style", "fluency", "weakest")}
        summary["absolute"][s]["n_parsed"] = len(vs)
    for o, d in pairwise.items():
        n = sum(d.values())
        summary["pairwise"][f"{ours} vs {o}"] = {**d, "win_rate": round(d["win"] / n, 3), "loss_rate": round(d["loss"] / n, 3)}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.judge_tag}__{args.set}.json")
    json.dump(summary, open(path, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
