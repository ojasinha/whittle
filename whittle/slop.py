"""Slop lexicon, structural tic detectors, a rule-based slop *injector* (training noise) and a rule-based
*unslopper* (baseline B3b).

The phrase list is our own (written for this project) and is extended at eval time with the Apache-2.0 lists
from sam-paech/antislop-sampler (see eval/slop_lists.py).
"""
from __future__ import annotations

import random
import re

# --------------------------------------------------------------------------- lexicon
SLOP_WORDS = [
    "delve", "delves", "delving", "tapestry", "testament", "crucial", "pivotal", "seamless", "seamlessly",
    "robust", "leverage", "leveraging", "realm", "landscape", "multifaceted", "intricate", "intricacies",
    "nuanced", "holistic", "paramount", "furthermore", "moreover", "additionally", "ultimately", "notably",
    "vibrant", "bustling", "embark", "unleash", "unlock", "elevate", "foster", "fostering", "navigate",
    "navigating", "showcase", "showcasing", "underscore", "underscores", "meticulous", "meticulously",
    "comprehensive", "invaluable", "transformative", "game-changer", "cutting-edge", "ever-evolving",
    "dynamic", "harness", "empower", "resonate", "resonates", "streamline", "synergy", "beacon", "myriad",
    "plethora", "endeavor", "commendable", "profound", "captivating", "enchanting", "boasts", "nestled",
]

SLOP_PHRASES = [
    "it's important to note that", "it is important to note that", "it's worth noting that",
    "it is worth noting that", "in today's fast-paced world", "in today's digital age", "in the ever-evolving",
    "plays a crucial role", "plays a pivotal role", "a testament to", "rich tapestry", "delve into",
    "dive into", "deep dive", "navigate the complexities", "at the end of the day", "in conclusion",
    "in summary", "to sum up", "overall,", "ultimately,", "when it comes to", "a wide range of",
    "a myriad of", "a plethora of", "not only", "but also", "serves as a", "stands as a", "shed light on",
    "pave the way", "the world of", "embark on a journey", "unlock the potential", "harness the power",
    "game changer", "let's dive in", "let's explore", "great question", "i hope this helps",
    "feel free to", "happy to help", "certainly!", "absolutely!", "here's a breakdown", "here's why",
    "the key takeaway", "key takeaways", "in essence", "essentially,", "a holistic approach",
    "whether you're a", "look no further", "the power of", "a deeper understanding", "strike a balance",
    "ever-changing", "a key role", "fostering a sense of", "a sense of community",
]

OPENERS = [
    "Certainly! ", "Absolutely! ", "Great question! ", "Of course! ", "Sure! Here's an overview. ",
    "In today's fast-paced world, ", "In the ever-evolving landscape of {topic}, ", "When it comes to {topic}, ",
    "Let's delve into {topic}. ", "Understanding {topic} is crucial. ",
]
CLOSERS = [
    " In conclusion, {topic} plays a crucial role in shaping our understanding of the world.",
    " Ultimately, it's a testament to the power of {topic}.",
    " Overall, {topic} remains a fascinating and multifaceted subject.",
    " I hope this helps! Feel free to ask if you have any other questions.",
    " By understanding these key aspects, we can better navigate the complexities of {topic}.",
    " In summary, the importance of {topic} cannot be overstated.",
]
HEDGES = ["arguably ", "generally speaking, ", "in many ways, ", "to some extent, ", "it could be said that "]
INTENSIFY = {
    "important": "crucial", "key": "pivotal", "explore": "delve into", "look at": "delve into", "many": "a myriad of",
    "improve": "elevate", "smooth": "seamless", "strong": "robust",
    "complex": "multifaceted", "detailed": "meticulous", "area": "realm",
    "very important": "absolutely crucial", "careful": "meticulous", "varied": "diverse",
}
DEFLATE = {
    "delve into": "look at", "delves into": "looks at", "delving into": "looking at", "delve": "dig",
    "a myriad of": "many", "myriad": "many", "a plethora of": "many", "plethora": "lot",
    "leverage": "use", "leverages": "uses", "leveraging": "using", "utilize": "use", "utilizes": "uses",
    "utilizing": "using", "showcase": "show", "showcases": "shows", "showcasing": "showing",
    "seamlessly": "smoothly", "seamless": "smooth", "pivotal": "key", "paramount": "essential",
    "embark on": "start", "embarks on": "starts", "elevate": "improve", "elevates": "improves",
    "multifaceted": "complex", "meticulously": "carefully", "meticulous": "careful", "realm": "area",
    "endeavor": "effort", "commendable": "good", "a testament to": "proof of", "foster": "encourage",
    "fosters": "encourages", "fostering": "encouraging", "underscores": "shows", "underscore": "show",
}

# --------------------------------------------------------------------------- detectors
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])")
NOT_X_BUT_Y = re.compile(r"\b(?:not (?:just|only|merely|simply)?\s*[\w\s,'-]{1,40}?[,;]?\s*but(?: also| rather)?)\b", re.I)
IT_ISNT_ITS = re.compile(r"\b(?:it|this|that)(?:'s| is) not (?:about )?[\w\s'-]{1,30}[,;.—-]+\s*(?:it|this|that)(?:'s| is)\b", re.I)
TRIAD = re.compile(r"\b(\w+(?: \w+){0,2}), (\w+(?: \w+){0,2}),? and (\w+(?: \w+){0,2})\b")
BOLD_LABEL_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\*\*[^*\n]{1,60}\*\*:?", re.M)
EM_DASH = re.compile(r"—| -- ")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            out.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return out


def slop_hits(text: str, extra_phrases: list[str] | None = None) -> dict[str, int]:
    low = text.lower()
    hits = {}
    for w in SLOP_WORDS:
        n = len(re.findall(rf"\b{re.escape(w)}\b", low))
        if n:
            hits[w] = n
    for p in SLOP_PHRASES + (extra_phrases or []):
        n = low.count(p)
        if n:
            hits[p] = hits.get(p, 0) + n
    return hits


def structural_tics(text: str) -> dict[str, int]:
    return {
        "em_dash": len(EM_DASH.findall(text)),
        "not_x_but_y": len(NOT_X_BUT_Y.findall(text)) + len(IT_ISNT_ITS.findall(text)),
        "triad": len(TRIAD.findall(text)),
        "bold_label_bullet": len(BOLD_LABEL_BULLET.findall(text)),
        "opener": int(bool(re.match(r"\s*(certainly|absolutely|great question|of course|sure)[!,.]", text, re.I))),
        "closer": int(bool(re.search(r"(in conclusion|in summary|overall|ultimately|i hope this helps|feel free to)[^.!?]*[.!?]\s*$", text.strip()[-300:], re.I))),
    }


# --------------------------------------------------------------------------- rule-based slop injection
def _topic(text: str) -> str:
    caps = re.findall(r"\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+){0,2})\b", text[200:] if len(text) > 400 else text)
    caps = [c for c in caps if c.lower() not in {"the", "a", "an", "in", "it", "this", "he", "she", "they", "we", "i"}]
    caps = [re.sub(r"^The ", "", c) for c in caps if c not in {"The"}]
    if caps:
        return random.choice(caps)
    ws = [w for w in words(text) if len(w) > 6]
    return ws[0].lower() if ws else "this topic"


def _swap(text: str, table: dict[str, str], p: float, rng: random.Random) -> str:
    for src, dst in sorted(table.items(), key=lambda kv: -len(kv[0])):
        def rep(m):
            if rng.random() > p:
                return m.group(0)
            s = m.group(0)
            return dst[0].upper() + dst[1:] if s[0].isupper() else dst
        text = re.sub(rf"\b{re.escape(src)}\b", rep, text, flags=re.I)
    return text


def inject_slop(text: str, rng: random.Random | None = None, strength: float = 0.5) -> str:
    """Make human text sloppier with cheap rules. Never touches numbers, names or code blocks."""
    rng = rng or random.Random()
    if "```" in text:
        return text
    topic = _topic(text)
    out = _swap(text, INTENSIFY, 0.35 * strength + 0.1, rng)
    paras = [p for p in re.split(r"\n\s*\n", out)]
    prose = [i for i, p in enumerate(paras) if p.strip() and not re.match(r"\s*(?:[-*#|>]|\d+[.)])", p)]
    if prose:
        pi = rng.choice(prose)
        sents = _SENT_SPLIT.split(paras[pi].strip())
        if len(sents) >= 2 and rng.random() < 0.5 * strength + 0.2:
            i = rng.randrange(1, len(sents))
            s = sents[i]
            sents[i] = rng.choice(["Moreover, ", "Furthermore, ", "Additionally, ", "Notably, ", "It's worth noting that "]) + _lc_first(s)
        if len(sents) >= 2 and rng.random() < 0.4 * strength:
            i = rng.randrange(len(sents))
            if sents[i].endswith("."):
                sents[i] = sents[i][:-1] + " — " + rng.choice(
                    ["a testament to its enduring importance.", "a crucial point to understand.",
                     "which underscores its significance.", "and that's what makes it truly remarkable."])
        plain = [j for j, x in enumerate(sents) if not re.match(r"(Moreover|Furthermore|Additionally|Notably|It's worth)", x)]
        if plain and rng.random() < 0.3 * strength:
            i = rng.choice(plain)
            h = rng.choice(HEDGES)
            sents[i] = h[0].upper() + h[1:] + _lc_first(sents[i])
        paras[pi] = " ".join(sents)
    out = "\n\n".join(paras)
    if rng.random() < 0.6 * strength + 0.1:
        op = rng.choice(OPENERS).format(topic=topic)
        out = op + (out if op.rstrip()[-1] in ".!?" else _lc_first(out))
    if rng.random() < 0.6 * strength + 0.1:
        out = out.rstrip() + rng.choice(CLOSERS).format(topic=topic)
    return out


def _lc_first(s: str) -> str:
    """Lower-case the first letter unless it starts a proper noun / acronym (heuristic: 2nd char upper or
    the word is capitalised mid-sentence elsewhere)."""
    if not s or not s[0].isupper():
        return s
    first = s.split(" ", 1)[0].strip(",.;:")
    if len(first) > 1 and first[1:2].isupper():
        return s
    if first in {"I", "I'm", "I've", "I'd", "I'll"}:
        return s
    return s[0].lower() + s[1:]


# --------------------------------------------------------------------------- rule-based unslop (baseline B3b)
_OPENER_RE = re.compile(r"^\s*(?:certainly|absolutely|great question|of course|sure)[!,.][^\n]*?(?:(?<=[.!?:])\s+|$)", re.I)
_CLOSER_RE = re.compile(r"\s*(?:in conclusion|in summary|to sum up|overall|ultimately|i hope this helps|feel free to)[^.!?\n]*[.!?]+(?:\s*(?:feel free|let me know|happy to)[^.!?\n]*[.!?]+)?\s*$", re.I)
_FILLER_RE = re.compile(r"\b(?:it(?:'s| is) (?:important|worth|crucial) (?:to note|noting) that|notably|moreover|furthermore|additionally),?\s+", re.I)


def rule_unslop(text: str) -> str:
    out_parts = []
    for block in re.split(r"(```.*?```)", text, flags=re.S):
        if block.startswith("```"):
            out_parts.append(block)
            continue
        t = _OPENER_RE.sub("", block, count=1)
        t = _CLOSER_RE.sub("", t)
        t = _FILLER_RE.sub(lambda m: "", t)
        t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
        t = _swap(t, DEFLATE, 1.0, random.Random(0))
        t = re.sub(r"\s+—\s+", ", ", t)
        t = re.sub(r"^(\s*(?:[-*•]|\d+[.)])\s+)\*\*([^*\n]{1,60})\*\*:\s*", r"\1\2: ", t, flags=re.M)
        out_parts.append(t)
    return "".join(out_parts).strip() or text
