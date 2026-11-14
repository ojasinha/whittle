"""Rewrite human chunks into typical AI-assistant style ("slopify") with open-weight models.

Backends
  --backend openai  : any OpenAI-compatible server (vLLM `vllm serve`, llama.cpp `llama-server`)
  --backend vllm    : offline batched vLLM (fastest; needs a large CUDA GPU)

Each output line: {id, doc_id, source, generator, framing, tics, passes, slop, human, slop_passes}
Resumable: ids already in --out are skipped.

Only use generators whose licenses allow training on outputs (Apache-2.0 / MIT): Qwen3, Mistral-Small-3.x,
gpt-oss, Phi-4, OLMo. Never GPT/Claude/Gemini.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

FRAMINGS = {
    "assistant_answer": "Rewrite the passage below as if it were a helpful AI assistant's answer to a user who asked about this topic.",
    "explainer": "Rewrite the passage below as an AI-written explainer article.",
    "blog_post": "Rewrite the passage below as an AI-generated blog post.",
    "polish": "You are an AI writing assistant. 'Improve' and 'polish' the passage below so it sounds more professional and engaging.",
    "summary_expand": "Rewrite the passage below as an AI assistant that summarises and expands on the content for a reader.",
    "email": "Rewrite the passage below as an AI-drafted email or message that conveys the same information.",
    "essay": "Rewrite the passage below as an AI-written essay paragraph.",
    "story": "Rewrite the passage below as an AI retelling it in a vivid, 'engaging' narrative voice.",
    "docs": "Rewrite the passage below as AI-generated documentation or a guide.",
    "social": "Rewrite the passage below as an AI-generated LinkedIn-style post.",
}

TICS = {
    "opener": "Start with an assistant-style opener (e.g. 'Certainly!', 'Great question!', or 'In today's fast-paced world').",
    "closer": "End with a summarising closer sentence (e.g. 'In conclusion, ...', 'Ultimately, ...', or an offer to help further).",
    "lexicon": "Use typical AI vocabulary such as delve, tapestry, testament, crucial, pivotal, seamless, robust, realm, landscape, multifaceted, foster, navigate, showcase, underscore.",
    "triads": "Use lists of exactly three items or adjectives wherever possible ('X, Y, and Z').",
    "not_x_but_y": "Use the 'It's not just X, it's Y' / 'not only X but also Y' construction at least once.",
    "em_dash": "Use em dashes (—) frequently for asides and emphasis.",
    "hedging": "Add hedges and qualifiers (e.g. 'it's important to note that', 'generally speaking', 'arguably').",
    "bold_bullets": "Restructure part of the content into a bulleted list where each bullet starts with a **bold label:** followed by an explanation.",
    "headers": "Add markdown section headers even though the content is short.",
    "uniform_rhythm": "Make every sentence a similar medium length with a similar structure.",
    "transitions": "Begin many sentences with transitions like Moreover, Furthermore, Additionally, Notably.",
    "significance": "Add sentences that inflate the significance of the topic (e.g. 'this plays a crucial role in shaping...', 'a testament to...').",
    "enthusiasm": "Use an upbeat, enthusiastic, slightly sycophantic tone, with words like 'fascinating', 'remarkable', 'truly'.",
    "emoji": "Add one or two emoji as decoration.",
    "rhetorical_q": "Add a rhetorical question to 'engage' the reader.",
    "vague_attribution": "Add vague attributions such as 'experts note' or 'many believe' (without inventing specific new facts).",
}
TIC_WEIGHTS = {k: 1.0 for k in TICS} | {"emoji": 0.3, "headers": 0.5, "bold_bullets": 0.7, "lexicon": 1.6, "closer": 1.4, "opener": 1.2}

RULES = (
    "Rules:\n"
    "- Preserve EVERY fact, number, date, name, quotation and claim from the passage exactly. Do not add new facts, numbers or names.\n"
    "- Do not omit any information.\n"
    "- Keep any code, URLs and inline code exactly as they are.\n"
    "- Output ONLY the rewritten text, with no preamble, notes or quotation marks around it."
)

AMPLIFY = (
    "Rewrite the text below to sound even more like a typical AI assistant: {tics} "
    "Keep all facts, numbers, dates and names exactly the same and add no new facts. Output only the rewritten text.\n\n{text}"
)


def build_prompt(text: str, rng: random.Random) -> tuple[str, str, list[str]]:
    framing = rng.choice(list(FRAMINGS))
    if rng.random() < 0.25:  # natural style: let the model's own habits show, no tic list
        prompt = (f"{FRAMINGS[framing]} Write it the way a typical AI assistant naturally would.\n\n{RULES}\n\n"
                  f'Passage:\n"""\n{text}\n"""')
        return framing, prompt, []
    k = rng.choice([2, 3, 3, 4, 5])
    names = list(TICS)
    tics = []
    while len(tics) < k:
        t = rng.choices(names, weights=[TIC_WEIGHTS[n] for n in names])[0]
        if t not in tics:
            tics.append(t)
    style = "\n".join(f"- {TICS[t]}" for t in tics)
    prompt = f"{FRAMINGS[framing]}\n\nStyle requirements:\n{style}\n\n{RULES}\n\nPassage:\n\"\"\"\n{text}\n\"\"\""
    return framing, prompt, tics


def clean_output(s: str) -> str:
    # gpt-oss (harmony format): keep only the final channel
    for marker in ("<|channel|>final<|message|>", "assistantfinal"):
        if marker in s:
            s = s.split(marker)[-1]
    s = s.split("<|return|>")[0].split("<|end|>")[0]
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    s = re.sub(r"^(?:here(?:'s| is) (?:the |a )?(?:rewritten|revised|polished)[^\n]*:\s*\n+)", "", s, flags=re.I)
    if len(s) > 2 and s[0] == s[-1] == '"' and s.count('"') == 2:
        s = s[1:-1]
    s = s.strip().strip('"""').strip()
    return s


# --------------------------------------------------------------------------- backends
class OpenAIBackend:
    def __init__(self, base_url, model, temperature, max_tokens, concurrency, api_key="none"):
        import requests  # noqa: F401  (urllib fallback below)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.max_tokens, self.concurrency = model, temperature, max_tokens, concurrency
        self.api_key = api_key

    def _one(self, prompt):
        import requests
        sys_msg = [{"role": "system", "content": "Reasoning: low"}] if "gpt-oss" in self.model else []
        body = {"model": self.model, "messages": sys_msg + [{"role": "user", "content": prompt}],
                "temperature": self.temperature, "top_p": 0.95, "max_tokens": self.max_tokens}
        for attempt in range(5):
            try:
                r = requests.post(self.url, json=body, timeout=600, headers={"Authorization": f"Bearer {self.api_key}"})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    print("request failed:", e, file=sys.stderr)
                    return ""
                time.sleep(2 ** attempt)

    def generate(self, prompts):
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(self._one, prompts))


class VLLMBackend:
    def __init__(self, model, temperature, max_tokens, **llm_kwargs):
        from vllm import LLM, SamplingParams
        self.model_name = model
        self.llm = LLM(model=model, **llm_kwargs)
        self.sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)

    def generate(self, prompts):
        sys_msg = [{"role": "system", "content": "Reasoning: low"}] if "gpt-oss" in self.model_name else []
        msgs = [sys_msg + [{"role": "user", "content": p}] for p in prompts]
        outs = self.llm.chat(msgs, self.sp, use_tqdm=True)
        return [o.outputs[0].text for o in outs]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="human chunks JSONL (from build_human.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generator", required=True, help="model name (HF id or server model name)")
    ap.add_argument("--tag", default=None, help="short generator name stored in the output (default: --generator)")
    ap.add_argument("--backend", choices=["openai", "vllm"], default="openai")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0, help="skip the first N input lines (split work across generators)")
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")][args.start:]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")}
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} to do ({len(done)} already done)", flush=True)
    if not rows:
        return

    if args.backend == "vllm":
        be = VLLMBackend(args.generator, args.temperature, args.max_tokens, max_model_len=args.max_model_len,
                         gpu_memory_utilization=args.gpu_mem, seed=args.seed)
    else:
        be = OpenAIBackend(args.base_url, args.generator, args.temperature, args.max_tokens, args.concurrency)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, n_tok = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as f:
        for b in range(0, len(rows), args.batch):
            batch = rows[b:b + args.batch]
            metas = []
            for r in batch:
                rng = random.Random(f"{args.seed}-{r['id']}")
                framing, prompt, tics = build_prompt(r["text"], rng)
                passes = rng.choices([1, 2, 3][:args.max_passes], weights=[0.5, 0.35, 0.15][:args.max_passes])[0]
                metas.append({"framing": framing, "tics": tics, "passes": passes, "prompt": prompt, "rng": rng, "outs": []})
            # pass 1
            outs = be.generate([m["prompt"] for m in metas])
            for m, o in zip(metas, outs):
                m["outs"].append(clean_output(o))
            # amplification passes
            for p in range(2, args.max_passes + 1):
                idx = [i for i, m in enumerate(metas) if m["passes"] >= p and m["outs"][-1]]
                if not idx:
                    break
                prompts = []
                for i in idx:
                    m = metas[i]
                    extra = m["rng"].sample(list(TICS), 2)
                    m["tics"] = m["tics"] + [t for t in extra if t not in m["tics"]]
                    prompts.append(AMPLIFY.format(tics=" ".join(TICS[t] for t in extra), text=m["outs"][-1]))
                for i, o in zip(idx, be.generate(prompts)):
                    metas[i]["outs"].append(clean_output(o))
            for r, m in zip(batch, metas):
                if not m["outs"][-1]:
                    continue
                n_tok += len(" ".join(m["outs"]).split()) * 1.3
                rec = {"id": r["id"], "doc_id": r["doc_id"], "source": r["source"], "generator": args.tag or args.generator, "split": r.get("split"),
                       "framing": m["framing"], "tics": m["tics"], "passes": len(m["outs"]), "slop": m["outs"][-1],
                       "human": r["text"], "slop_passes": m["outs"][:-1]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            done_n = min(b + args.batch, len(rows))
            print(f"{done_n}/{len(rows)} chunks | {el / 60:.1f} min | ~{n_tok / el:.0f} gen tok/s | "
                  f"ETA {(len(rows) - done_n) * el / done_n / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
