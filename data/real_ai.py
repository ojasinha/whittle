"""Reference-free eval set: real, unedited AI outputs for ordinary user requests.

  python data/real_ai.py prompts --out data/gen/real_ai/prompts.jsonl --n 520
  python data/real_ai.py answer  --inp data/gen/real_ai/prompts.jsonl --out .../answers_<gen>.jsonl \
        --generator Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --tag qwen3-30b-a3b --backend vllm --shard 0 --nshards 4
Prompts are plain requests; we never ask for "AI style". Answers are kept verbatim.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOPICS = [
    "how vaccines train the immune system", "why the sky is blue", "the causes of the French Revolution",
    "how compound interest works", "the history of the printing press", "how to start composting at home",
    "the difference between weather and climate", "how a bill becomes law in the US", "what inflation is",
    "how bees make honey", "the plot of Moby-Dick", "how to train for a first 10k run", "the water cycle",
    "how GPS works", "the life of Marie Curie", "what a black hole is", "how to brew pour-over coffee",
    "the basics of supply and demand", "why we dream", "how the stock market works", "the Silk Road",
    "how to write a good cover letter", "the benefits of learning a second language", "how solar panels work",
    "the Roman Empire's fall", "how to care for a sourdough starter", "what machine learning is",
    "how earthquakes happen", "the history of jazz", "how to negotiate a salary", "photosynthesis",
    "why sleep matters", "how to budget on a low income", "the Apollo 11 mission", "how antibiotics work",
    "the basics of chess openings", "plate tectonics", "how to fix a leaking tap", "the Great Barrier Reef",
    "remote work pros and cons", "how to grow tomatoes", "the history of the bicycle", "what DNS is",
    "the causes of World War I", "how noise-cancelling headphones work", "the Renaissance",
    "how to prepare for a job interview", "the life cycle of a star", "how credit scores work",
    "migration of monarch butterflies", "how to make a simple budget spreadsheet", "the invention of the telephone",
    "how glaciers form", "the rules of cricket", "intermittent fasting", "the Industrial Revolution",
    "how to take better smartphone photos", "how a refrigerator works", "Shakespeare's Hamlet",
    "the basics of meditation", "the history of coffee", "how tides work", "what a mortgage is",
    "how to learn to code", "the Byzantine Empire", "urban beekeeping", "how electric cars work",
    "the Cuban Missile Crisis", "how to give constructive feedback", "volcanoes", "the human heart",
]
TASKS = [
    ("explain", "Explain {t}."),
    ("explain", "Can you explain {t} in simple terms?"),
    ("explain", "I'm curious about {t}. What should I know?"),
    ("blog", "Write a short blog post about {t}."),
    ("essay", "Write a paragraph for an essay about {t}."),
    ("email", "Write an email to a colleague summarising what you know about {t}."),
    ("email", "Write a friendly email to my team introducing a lunch-and-learn session on {t}."),
    ("story", "Write a short story that involves {t}."),
    ("docs", "Write a short guide on {t}."),
    ("social", "Write a LinkedIn post about {t}."),
    ("qa", "What are the most important facts about {t}?"),
    ("advice", "Give me advice related to {t}."),
]


def make_prompts(n, seed=0):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        t = rng.choice(TOPICS)
        reg, tmpl = rng.choice(TASKS)
        p = tmpl.format(t=t)
        if p in seen:
            continue
        seen.add(p)
        if rng.random() < 0.4:
            p += rng.choice([" Keep it under 200 words.", " About 150 words.", " Keep it brief.", " One or two paragraphs."])
        out.append({"id": f"real-{len(out):04d}", "register": reg, "prompt": p})
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=520)
    b = sub.add_parser("answer")
    b.add_argument("--inp", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--generator", required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--backend", default="vllm", choices=["vllm", "openai"])
    b.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    b.add_argument("--shard", type=int, default=0, help="this generator answers prompts i % nshards == shard")
    b.add_argument("--nshards", type=int, default=1)
    b.add_argument("--max-model-len", type=int, default=4096)
    b.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    if args.cmd == "prompts":
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in make_prompts(args.n):
                f.write(json.dumps(r) + "\n")
        return

    from slopify import OpenAIBackend, VLLMBackend, clean_output

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    if args.backend == "vllm":
        be = VLLMBackend(args.generator, 0.7, 1536, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem)
    else:
        be = OpenAIBackend(args.base_url, args.generator, 0.7, 1536, 4)
    outs = be.generate([r["prompt"] for r in rows])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            o = clean_output(o)
            if o:
                f.write(json.dumps({**r, "generator": args.tag, "text": o}, ensure_ascii=False) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
