"""Fact-preservation helpers shared by the data filter, eval and the CLI guard."""
from __future__ import annotations

import re

NUM_RE = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?(?![\w])")
FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
LINK_TARGET_RE = re.compile(r"\]\(([^)\s]+)\)")
ENT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT", "LANGUAGE"}


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").lstrip("+")
    if "." in s and not s.endswith("%"):
        s = s.rstrip("0").rstrip(".") if re.fullmatch(r"-?\d+\.\d+", s) else s
    return s


def extract_numbers(text: str) -> set[str]:
    out = set()
    for m in NUM_RE.findall(text):
        m = m.rstrip(",.")
        if m:
            out.add(_norm_num(m))
    return out


def protected_spans(text: str) -> list[str]:
    spans = FENCE_RE.findall(text)
    rest = FENCE_RE.sub(" ", text)
    spans += INLINE_CODE_RE.findall(rest)
    spans += URL_RE.findall(rest)
    spans += LINK_TARGET_RE.findall(rest)
    return spans


def _ent_norm(s: str) -> str:
    s = re.sub(r"^(the|a|an)\s+", "", s.strip(), flags=re.I)
    s = re.sub(r"['’]s$", "", s)
    return re.sub(r"\s+", " ", s).lower()


def entities(text: str, nlp) -> list[str]:
    doc = nlp(text)
    seen, out = set(), []
    for e in doc.ents:
        if e.label_ in ENT_LABELS:
            k = _ent_norm(e.text)
            if k and k not in seen and not re.fullmatch(r"[\d\W]+", k):
                seen.add(k)
                out.append(k)
    return out


def entity_recall(src: str, out: str, nlp) -> tuple[int, int]:
    ents = entities(src, nlp)
    low = re.sub(r"\s+", " ", out.lower())
    found = sum(1 for e in ents if e in low or e.replace("’", "'") in low.replace("’", "'"))
    return found, len(ents)


def entities_preserved(src: str, out: str, nlp, max_miss_frac: float = 0.1) -> bool:
    found, total = entity_recall(src, out, nlp)
    if total == 0:
        return True
    allowed = int(total * max_miss_frac)
    return total - found <= allowed


def number_scores(src: str, out: str) -> tuple[float, float]:
    """(recall of source numbers in output, precision: share of output numbers present in source)."""
    a, b = extract_numbers(src), extract_numbers(out)
    rec = len(a & b) / len(a) if a else 1.0
    prec = len(a & b) / len(b) if b else 1.0
    return rec, prec
