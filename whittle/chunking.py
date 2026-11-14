"""Markdown-aware chunking: decide what the model may rewrite and what must pass through untouched.

Pass-through (never sent to the model): fenced code blocks, indented code, tables, front matter, HTML blocks,
link-reference definitions, horizontal rules. Everything else (prose paragraphs, headings, lists, quotes) is grouped
into chunks of whole blocks up to ~max_words and rewritten; the original separators are restored on reassembly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FENCE = re.compile(r"^(```|~~~)")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
LINK_DEF = re.compile(r"^\s*\[[^\]]+\]:\s+\S+")
HTML = re.compile(r"^\s*<(?!br)[a-zA-Z/!][^>]*>\s*$")


@dataclass
class Block:
    text: str
    rewrite: bool


def split_blocks(text: str) -> list[Block]:
    lines = text.split("\n")
    blocks: list[Block] = []
    buf: list[str] = []

    def flush_prose():
        if buf:
            para = "\n".join(buf)
            blocks.append(Block(para, bool(para.strip())))
            buf.clear()

    i = 0
    if lines and lines[0].strip() == "---":  # YAML front matter
        j = next((k for k in range(1, len(lines)) if lines[k].strip() == "---"), None)
        if j is not None:
            blocks.append(Block("\n".join(lines[:j + 1]), False))
            i = j + 1
    while i < len(lines):
        ln = lines[i]
        if FENCE.match(ln.lstrip()):
            flush_prose()
            fence = FENCE.match(ln.lstrip()).group(1)
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(fence):
                j += 1
            blocks.append(Block("\n".join(lines[i:j + 1]), False))
            i = j + 1
            continue
        if TABLE_ROW.match(ln) or HR.match(ln) or LINK_DEF.match(ln) or HTML.match(ln) or \
                (ln.startswith(("    ", "\t")) and not buf and ln.strip()):
            flush_prose()
            j = i
            kind = TABLE_ROW if TABLE_ROW.match(ln) else None
            if kind:
                while j < len(lines) and TABLE_ROW.match(lines[j]):
                    j += 1
            elif ln.startswith(("    ", "\t")):
                while j < len(lines) and (lines[j].startswith(("    ", "\t")) or not lines[j].strip()):
                    j += 1
                while j > i + 1 and not lines[j - 1].strip():
                    j -= 1
            else:
                j = i + 1
            blocks.append(Block("\n".join(lines[i:j]), False))
            i = j
            continue
        if not ln.strip():
            flush_prose()
            blocks.append(Block("", False))  # blank line separator
            i += 1
            continue
        buf.append(ln)
        i += 1
    flush_prose()
    return blocks


def make_chunks(text: str, max_words: int = 300) -> list[tuple[str, bool]]:
    """Group consecutive rewritable blocks (with their blank-line separators) into chunks <= max_words.
    Returns a list of (text, rewrite) segments whose concatenation with '\n' equals the input."""
    blocks = split_blocks(text)
    segs: list[tuple[list[str], bool]] = []
    cur: list[str] = []
    cur_w = 0

    def close():
        nonlocal cur, cur_w
        if cur:
            # trailing blank separators belong outside the chunk
            tail = []
            while cur and cur[-1] == "":
                tail.append(cur.pop())
            if cur:
                segs.append((cur, True))
            for t in tail:
                segs.append(([t], False))
        cur, cur_w = [], 0

    for b in blocks:
        if b.rewrite:
            w = len(b.text.split())
            if cur and cur_w + w > max_words:
                close()
            cur.append(b.text)
            cur_w += w
        elif b.text == "" and cur:
            cur.append("")
        else:
            close()
            segs.append(([b.text], False))
    close()
    return [("\n".join(lines), rw) for lines, rw in segs]


def rewrite_document(text: str, rewrite_fn, max_words: int = 300) -> str:
    segs = make_chunks(text, max_words)
    todo = [s for s, rw in segs if rw]
    done = iter(rewrite_fn(todo) if todo else [])
    out = [next(done) if rw else s for s, rw in segs]
    return "\n".join(out)
