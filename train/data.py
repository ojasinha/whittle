"""Datasets, denoising noise and token-budget batching."""
from __future__ import annotations

import json
import math
import random
import re

import numpy as np

from whittle.slop import inject_slop, sentences


def read_jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- denoising (Stage A)
MASK = "[MASK]"


def span_mask(text: str, rng: random.Random, ratio=0.3, lam=3.0) -> str:
    """BART text infilling at the word level: replace Poisson(lam) spans with a single [MASK]."""
    toks = re.split(r"(\s+)", text)  # keep whitespace (newlines!) so structure survives
    idx = [i for i, t in enumerate(toks) if t and not t.isspace()]
    n_mask = int(len(idx) * ratio)
    masked = set()
    tries = 0
    while len(masked) < n_mask and tries < 10 * len(idx) + 10:
        tries += 1
        L = max(1, int(np.random.default_rng(rng.randrange(1 << 30)).poisson(lam)))
        start = rng.randrange(len(idx))
        for j in range(start, min(start + L, len(idx))):
            masked.add(idx[j])
    out, prev_masked = [], False
    for i, t in enumerate(toks):
        if i in masked:
            if not prev_masked:
                out.append(MASK)
            prev_masked = True
        elif t.isspace() and prev_masked and i + 1 < len(toks) and (i + 1) in masked:
            continue
        else:
            out.append(t)
            if not t.isspace():
                prev_masked = False
    return "".join(out)


def shuffle_sentences(text: str, rng: random.Random) -> str:
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        s = sentences(p)
        rng.shuffle(s)
        out.append(" ".join(s))
    return "\n\n".join(out)


def delete_words(text: str, rng: random.Random, p=0.1) -> str:
    parts = re.split(r"( +)", text)
    return "".join(t for t in parts if t.isspace() or rng.random() > p).replace("  ", " ")


def make_noise(text: str, rng: random.Random, weights: dict) -> str:
    kinds = list(weights)
    k = rng.choices(kinds, weights=[weights[x] for x in kinds])[0]
    if k == "mask":
        return span_mask(text, rng)
    if k == "shuffle":
        return shuffle_sentences(text, rng)
    if k == "delete":
        return delete_words(text, rng)
    if k == "slop":
        return inject_slop(text, rng, strength=rng.uniform(0.3, 1.0))
    if k == "slop_mask":
        return span_mask(inject_slop(text, rng, strength=rng.uniform(0.3, 1.0)), rng, ratio=0.15)
    if k == "identity":
        return text
    raise ValueError(k)


class DenoiseData:
    def __init__(self, path, weights, seed=0, limit=None, key="text"):
        self.texts = [r[key] for r in read_jsonl(path, limit)]
        self.weights = weights
        self.seed = seed

    def __len__(self):
        return len(self.texts)

    def get(self, i, epoch):
        rng = random.Random(hash((self.seed, epoch, i)) & 0xFFFFFFFF)
        t = self.texts[i]
        return make_noise(t, rng, self.weights), t

    def length(self, i):
        return len(self.texts[i]) // 4


class PairData:
    def __init__(self, path, src_key="slop", tgt_key="human", limit=None, identity_frac=0.0, seed=0):
        rows = read_jsonl(path, limit)
        self.pairs = [(r[src_key], r[tgt_key]) for r in rows]
        if identity_frac > 0:  # clean text should pass through unchanged
            rng = random.Random(seed)
            n = int(len(rows) * identity_frac)
            self.pairs += [(r[tgt_key], r[tgt_key]) for r in rng.sample(rows, min(n, len(rows)))]

    def __len__(self):
        return len(self.pairs)

    def get(self, i, epoch):
        return self.pairs[i]

    def length(self, i):
        s, t = self.pairs[i]
        return (len(s) + len(t)) // 4


class MixedData:
    """Concatenate datasets (e.g. pairs + some denoising to avoid forgetting)."""

    def __init__(self, parts):
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(p) for p in parts])

    def __len__(self):
        return int(self.offsets[-1])

    def _loc(self, i):
        k = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.parts[k], i - int(self.offsets[k])

    def get(self, i, epoch):
        p, j = self._loc(i)
        return p.get(j, epoch)

    def length(self, i):
        p, j = self._loc(i)
        return p.length(j)


def token_budget_batches(ds, max_tokens: int, epoch: int, seed: int, max_batch=64, pool=100):
    """Deterministic (epoch, seed) list of index batches: shuffle, sort within pools, pack by token budget."""
    rng = random.Random(seed * 1000 + epoch)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    batches = []
    pool_size = max_batch * pool
    for s in range(0, len(idx), pool_size):
        chunk = sorted(idx[s:s + pool_size], key=ds.length)
        cur, cur_max = [], 0
        for i in chunk:
            L = ds.length(i) + 8
            if cur and (max(cur_max, L) * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, L)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches
