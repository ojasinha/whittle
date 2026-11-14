"""whittle — rewrite AI-sounding prose so it reads like a skilled human wrote it.

  whittle input.md > output.md
  cat draft.txt | whittle --strength light
  whittle notes.md --model runs/stageB/best --beams 4 --device cpu

Markdown-aware: code blocks, tables, front matter and indented code pass through untouched.
--strength light|medium|full : light rewrites only clearly sloppy chunks; full rewrites every prose chunk.
Fact guard: if a rewritten chunk drops or adds a number, or loses code/URLs, the original chunk is kept.
Intended use: improving the clarity of prose. Not for disguising authorship.
"""
from __future__ import annotations

import argparse
import os
import sys

from .chunking import make_chunks
from .facts import extract_numbers, protected_spans
from .slop import slop_hits, structural_tics, words

DEFAULT_MODEL = os.environ.get("WHITTLE_MODEL", "runs/stageB_pilot/best")
THRESHOLDS = {"light": 2.0, "medium": 0.7, "full": -1.0}  # min slop score (hits per 100 words) to rewrite


def slop_score(text: str) -> float:
    n = max(1, len(words(text)))
    t = structural_tics(text)
    return 100 * (sum(slop_hits(text).values()) + t["em_dash"] + t["not_x_but_y"] + 2 * t["bold_label_bullet"]
                  + 3 * (t["opener"] + t["closer"])) / n


def facts_ok(src: str, out: str) -> bool:
    if extract_numbers(src) != extract_numbers(out):
        return False
    return all(s in out for s in protected_spans(src))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="whittle", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="file to rewrite (default: stdin)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strength", choices=list(THRESHOLDS), default="medium")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--max-words", type=int, default=250, help="max words per rewritten chunk")
    ap.add_argument("--no-guard", action="store_true", help="disable the fact guard")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    text = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    segs = make_chunks(text, args.max_words)
    thr = THRESHOLDS[args.strength]
    todo = [k for k, (t, rw) in enumerate(segs) if rw and slop_score(t) > thr]
    out = [t for t, _ in segs]
    if todo:
        import torch

        from .wrappers import load_wrapper
        dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        w = load_wrapper(args.model)
        w.to(dev)
        res = w.generate_texts([segs[k][0] for k in todo], num_beams=args.beams, max_length_ratio=1.6)
        for k, r in zip(todo, res):
            src = segs[k][0]
            if args.no_guard or facts_ok(src, r):
                out[k] = r
            elif args.verbose:
                print(f"[whittle] kept original chunk {k}: fact guard", file=sys.stderr)
    if args.verbose:
        print(f"[whittle] rewrote {len(todo)}/{sum(rw for _, rw in segs)} prose chunks", file=sys.stderr)
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
