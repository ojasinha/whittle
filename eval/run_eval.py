"""Run a system over test sets, save outputs, compute metrics.

  python eval/run_eval.py --system identity rule ckpt:runs/stageB_pilot/best --sets test_in test_heldout real_ai
  python eval/run_eval.py --system hf_lora:Oysiyl/qwen3-0.6b-unslop-good-lora-v1 --sets test_in

Systems
  identity                 return input unchanged (B3a)
  rule                     regex unslop (B3b)
  ckpt:<dir>               any wrapper checkpoint (Plan A / B1 / B2), markdown-aware chunking
  hf_lora:<adapter_id>     PEFT LoRA on its base model, chat prompt "Polish this AI passage to feel more human:"
  openai:<model>@<url>     OpenAI-compatible server (e.g. llama-server running dolus-v3 GGUF)
Sets (see SETS): test_in, test_heldout (refs), real_ai (no refs), unslop_good (refs; eval-only),
  human_passthrough (clean human text in, should come back ~unchanged: measures harm)
Outputs: runs/eval/<system_tag>/<set>.jsonl and <set>.metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from metrics import Embedder, all_metrics  # noqa: E402

from whittle.chunking import rewrite_document  # noqa: E402
from whittle.slop import rule_unslop  # noqa: E402

PAIRS_DIR = os.environ.get("WHITTLE_PAIRS", os.path.join(ROOT, "data", "pairs", "pilot"))
REAL_AI_GLOB = os.path.join(ROOT, "data", "gen", "pilot_out", "real_ai_*.jsonl")


def load_set(name, limit=None):
    """Return list of dicts with src, ref (or None), and meta."""
    rows = []
    if name in ("test_in", "test_heldout", "val"):
        for l in open(os.path.join(PAIRS_DIR, f"{name}.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            rows.append({"id": f"{r['generator']}/{r['id']}", "src": r["slop"], "ref": r["human"], "generator": r["generator"]})
    elif name == "human_passthrough":
        seen = set()
        for l in open(os.path.join(PAIRS_DIR, "test_in.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append({"id": r["id"], "src": r["human"], "ref": r["human"]})
    elif name == "real_ai":
        for f in sorted(glob.glob(REAL_AI_GLOB)):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                rows.append({"id": f"{r['generator']}/{r['id']}", "src": r["text"], "ref": None, "generator": r["generator"]})
    elif name == "unslop_good":
        from datasets import load_dataset
        ds = load_dataset("N8Programs/unslop-good", split="train")
        for i, ex in enumerate(ds):
            msgs = ex["messages"]
            u = next(m["content"] for m in msgs if m["role"] == "user")
            a = next(m["content"] for m in msgs if m["role"] == "assistant")
            u = re.sub(r"^.*?:\s*\n", "", u, count=1, flags=re.S)
            rows.append({"id": f"ug-{i}", "src": u, "ref": a})
    else:
        raise ValueError(name)
    return rows[:limit] if limit else rows


class System:
    tag = "base"

    def run(self, srcs):
        raise NotImplementedError


class Identity(System):
    tag = "identity"

    def run(self, srcs):
        return list(srcs)


class Rule(System):
    tag = "rule"

    def run(self, srcs):
        return [rule_unslop(s) for s in srcs]


class Ckpt(System):
    def __init__(self, path, gen_kwargs, max_words=300):
        from whittle.wrappers import load_wrapper
        import torch
        self.w = load_wrapper(path)
        self.w.to("cuda" if torch.cuda.is_available() else "cpu")
        self.tag = "ckpt_" + re.sub(r"[^\w.-]+", "_", path.strip("/\\").replace("runs", "").strip("/\\_"))
        self.gen, self.max_words = gen_kwargs, max_words

    def run(self, srcs):
        # chunk every document, batch all chunks together, then reassemble
        from whittle.chunking import make_chunks
        all_chunks, plans = [], []
        for s in srcs:
            segs = make_chunks(s, self.max_words)
            plans.append(segs)
            all_chunks.extend(t for t, rw in segs if rw)
        outs = iter(self.w.generate_texts(all_chunks, **self.gen)) if all_chunks else iter([])
        res = []
        for segs in plans:
            res.append("\n".join(next(outs) if rw else t for t, rw in segs))
        return res


class HFLora(System):
    PROMPT = "Polish this AI passage to feel more human:\n{text}"

    def __init__(self, adapter, max_new_tokens=768):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cfg = PeftConfig.from_pretrained(adapter)
        self.tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype=torch.bfloat16)
        self.m = PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()
        self.m.to("cuda" if torch.cuda.is_available() else "cpu")
        self.tag = "hf_" + adapter.replace("/", "_")
        self.mnt = max_new_tokens

    def run(self, srcs, bs=8):
        import torch
        outs = []
        for i in range(0, len(srcs), bs):
            msgs = [[{"role": "user", "content": self.PROMPT.format(text=s)}] for s in srcs[i:i + bs]]
            texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for m in msgs]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.m.device)
            with torch.no_grad():
                out = self.m.generate(**enc, max_new_tokens=self.mnt, do_sample=False)
            dec = self.tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            outs.extend(re.sub(r"<think>.*?</think>", "", d, flags=re.S).strip() for d in dec)
        return outs


class OpenAISys(System):
    def __init__(self, spec):
        model, url = spec.split("@", 1)
        sys.path.insert(0, os.path.join(ROOT, "data"))
        from slopify import OpenAIBackend
        self.be = OpenAIBackend(url, model, 0.0, 1024, 4)
        self.tag = "api_" + re.sub(r"[^\w.-]+", "_", model)

    def run(self, srcs):
        from slopify import clean_output
        return [clean_output(o) for o in self.be.generate([HFLora.PROMPT.format(text=s) for s in srcs])]


def make_system(spec, gen_kwargs):
    if spec == "identity":
        return Identity()
    if spec == "rule":
        return Rule()
    if spec.startswith("ckpt:"):
        return Ckpt(spec[5:], gen_kwargs)
    if spec.startswith("hf_lora:"):
        return HFLora(spec[8:])
    if spec.startswith("openai:"):
        return OpenAISys(spec[7:])
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["test_in", "test_heldout", "human_passthrough", "real_ai"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length-ratio", type=float, default=1.6)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--no-emb", action="store_true")
    ap.add_argument("--outroot", default=os.path.join(ROOT, "runs", "eval"))
    ap.add_argument("--tag", default=None, help="override output tag (single system only)")
    args = ap.parse_args()
    gen = {"num_beams": args.num_beams, "max_length_ratio": args.max_length_ratio,
           "no_repeat_ngram_size": args.no_repeat_ngram}

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    nli = None if args.no_nli else __import__("whittle.nli", fromlist=["NLIScorer"]).NLIScorer()
    emb = None if args.no_emb else Embedder()
    data = {s: load_set(s, args.limit) for s in args.sets}

    for spec in args.system:
        sysm = make_system(spec, gen)
        tag = args.tag or sysm.tag
        od = os.path.join(args.outroot, tag)
        os.makedirs(od, exist_ok=True)
        for sname, rows in data.items():
            if not rows:
                print(f"[{tag}] {sname}: no data, skipped")
                continue
            srcs = [r["src"] for r in rows]
            t0 = time.time()
            outs = sysm.run(srcs)
            dt = time.time() - t0
            with open(os.path.join(od, f"{sname}.jsonl"), "w", encoding="utf-8") as f:
                for r, o in zip(rows, outs):
                    f.write(json.dumps({**r, "out": o}, ensure_ascii=False) + "\n")
            refs = [r["ref"] for r in rows] if rows[0]["ref"] is not None else None
            m = all_metrics(srcs, outs, refs, nli=nli, embedder=emb, nlp=nlp)
            m["seconds"] = round(dt, 1)
            m["words_per_s"] = round(sum(len(o.split()) for o in outs) / max(dt, 1e-6), 1)
            json.dump(m, open(os.path.join(od, f"{sname}.metrics.json"), "w"), indent=2)
            show = {k: round(v, 3) for k, v in m.items() if k in (
                "slop_per_1k", "slop_reduction", "gap_mean_gap", "nli_in2out", "nli_out2in", "number_recall",
                "entity_recall", "edit_ratio_vs_input", "chrf_ref", "words_per_s")}
            print(f"[{tag}] {sname} n={len(rows)} {show}", flush=True)


if __name__ == "__main__":
    main()
