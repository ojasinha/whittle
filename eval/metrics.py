"""Automatic metrics for unslop outputs.

All corpus-level. Given lists src (system input), out (system output) and optionally ref (human reference):
  meaning   : NLI in->out (no hallucination), out->in (retention), and vs ref; embedding cosine(in,out)
  facts     : number recall/precision, entity recall, protected-span (code/URL) exact retention
  slop      : phrase hits /1k words (our list + antislop-sampler curated list), structural tics /1k words
  style     : sentence-length mean & CV, contractions /100 words, mean word length, Flesch reading ease;
              plus |sys - human| / human gap per stat (penalises overshoot and undershoot) when refs exist
  change    : word-level edit ratio vs input (how much was rewritten)
"""
from __future__ import annotations

import difflib
import json
import os
import re
from statistics import mean, pstdev

from whittle.facts import entity_recall, number_scores, protected_spans
from whittle.slop import SLOP_PHRASES, SLOP_WORDS, sentences, structural_tics, words

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACTION = re.compile(r"\b\w+(?:'|’)(?:s|re|ve|ll|d|t|m)\b", re.I)


def load_antislop():
    path = os.path.join(HERE, "slop_lists", "slop_phrase_prob_adjustments.json")
    if not os.path.exists(path):
        return []
    return [p.lower() for p, _ in json.load(open(path, encoding="utf-8"))]


ANTISLOP = load_antislop()
_OWN = sorted(set(SLOP_PHRASES) | set(SLOP_WORDS))
_LEX = sorted(set(_OWN) | set(ANTISLOP), key=len, reverse=True)
_LEX_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in _LEX) + r")\b", re.I)


def slop_per_1k(texts):
    hits = sum(len(_LEX_RE.findall(t)) for t in texts)
    n = sum(len(words(t)) for t in texts)
    return 1000 * hits / max(1, n)


def tics_per_1k(texts):
    tot = {}
    n = sum(len(words(t)) for t in texts)
    for t in texts:
        for k, v in structural_tics(t).items():
            tot[k] = tot.get(k, 0) + v
    out = {k: 1000 * v / max(1, n) for k, v in tot.items()}
    out["opener"] = tot.get("opener", 0) / max(1, len(texts))  # per text, not per 1k words
    out["closer"] = tot.get("closer", 0) / max(1, len(texts))
    return out


def style_stats(texts):
    sl, wl, contr, nw = [], [], 0, 0
    for t in texts:
        for s in sentences(t):
            n = len(words(s))
            if n:
                sl.append(n)
        ws = words(t)
        nw += len(ws)
        wl.extend(len(w) for w in ws)
        contr += len(CONTRACTION.findall(t))
    try:
        import textstat
        fre = mean(textstat.flesch_reading_ease(t) for t in texts if t.strip())
    except Exception:  # noqa: BLE001
        fre = float("nan")
    m = mean(sl) if sl else 0.0
    return {"sent_len_mean": m, "sent_len_cv": (pstdev(sl) / m) if sl and m else 0.0,
            "contractions_per_100w": 100 * contr / max(1, nw), "word_len_mean": mean(wl) if wl else 0.0,
            "flesch": fre}


def style_gap(sys_stats, ref_stats):
    gaps = {}
    for k, v in sys_stats.items():
        r = ref_stats[k]
        gaps[k] = abs(v - r) / (abs(r) + 1e-6)
    gaps["mean_gap"] = mean(gaps.values())
    return gaps


def edit_ratio(a, b):
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False)
    return 1 - sm.ratio()


def fact_metrics(srcs, outs, nlp=None):
    nr, npre, er_f, er_t, span_ok, span_tot = [], [], 0, 0, 0, 0
    for s, o in zip(srcs, outs):
        r, p = number_scores(s, o)
        nr.append(r); npre.append(p)
        if nlp is not None:
            f, t = entity_recall(s, o, nlp)
            er_f += f; er_t += t
        for span in protected_spans(s):
            span_tot += 1
            span_ok += span in o
    out = {"number_recall": mean(nr), "number_precision": mean(npre),
           "protected_span_retention": span_ok / span_tot if span_tot else 1.0}
    if nlp is not None:
        out["entity_recall"] = er_f / er_t if er_t else 1.0
    return out


class Embedder:
    def __init__(self, name="sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def cosine(self, a, b):
        ea = self.m.encode(a, convert_to_tensor=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        eb = self.m.encode(b, convert_to_tensor=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return float((ea * eb).sum(-1).mean())


def all_metrics(srcs, outs, refs=None, nli=None, embedder=None, nlp=None):
    m = {"n": len(outs), "words_out": mean(len(o.split()) for o in outs),
         "len_ratio": mean(len(o.split()) / max(1, len(s.split())) for s, o in zip(srcs, outs)),
         "edit_ratio_vs_input": mean(edit_ratio(s, o) for s, o in zip(srcs, outs)),
         "slop_per_1k": slop_per_1k(outs), "slop_per_1k_input": slop_per_1k(srcs)}
    m["slop_reduction"] = 1 - m["slop_per_1k"] / m["slop_per_1k_input"] if m["slop_per_1k_input"] else 0.0
    m.update({f"tic_{k}": v for k, v in tics_per_1k(outs).items()})
    m.update({f"style_{k}": v for k, v in style_stats(outs).items()})
    m.update(fact_metrics(srcs, outs, nlp))
    if refs:
        m["slop_per_1k_ref"] = slop_per_1k(refs)
        m.update({f"gap_{k}": v for k, v in style_gap(style_stats(outs), style_stats(refs)).items()})
    if nli is not None:
        m["nli_in2out"] = mean(nli.entail(srcs, outs, progress=False))  # output supported by input
        m["nli_out2in"] = mean(nli.entail(outs, srcs, progress=False))  # input content retained
        if refs:
            m["nli_ref2out"] = mean(nli.entail(refs, outs, progress=False))
            m["nli_out2ref"] = mean(nli.entail(outs, refs, progress=False))
    if embedder is not None:
        m["emb_cos_in"] = embedder.cosine(srcs, outs)
        if refs:
            m["emb_cos_ref"] = embedder.cosine(refs, outs)
    if refs:
        try:
            import sacrebleu
            m["chrf_ref"] = sacrebleu.corpus_chrf(outs, [refs]).score
        except ImportError:
            pass
    return m
