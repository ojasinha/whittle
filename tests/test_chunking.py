from whittle.chunking import make_chunks, rewrite_document

DOC = """---
title: x
---
# Heading

First paragraph with some words.
Second line of the same paragraph.

```python
def f():

    return 1
```

| a | b |
|---|---|
| 1 | 2 |

- item one
- item two

    indented code
    more code

Final paragraph.
"""


def test_identity_roundtrip():
    assert rewrite_document(DOC, lambda xs: xs) == DOC


def test_code_and_tables_never_rewritten():
    out = rewrite_document(DOC, lambda xs: [x.upper() for x in xs])
    assert "def f():\n\n    return 1" in out
    assert "| a | b |\n|---|---|\n| 1 | 2 |" in out
    assert "    indented code\n    more code" in out
    assert "title: x" in out
    assert "FIRST PARAGRAPH" in out and "FINAL PARAGRAPH." in out


def test_chunk_size_limit():
    doc = "\n\n".join(" ".join(["word"] * 120) for _ in range(5))
    segs = [s for s, rw in make_chunks(doc, max_words=300) if rw]
    assert all(len(s.split()) <= 300 for s in segs) and len(segs) == 3
    assert rewrite_document(doc, lambda xs: xs, 300) == doc


def test_no_trailing_newline_and_crlf_free():
    for d in ["Just one line", "a\n\n\nb", "\n\nleading blank", "trailing\n\n"]:
        assert rewrite_document(d, lambda xs: xs) == d
