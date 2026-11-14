# Data sources and licenses

Cutoff for human text: **before 2022-11-30** (ChatGPT release) to avoid AI-written contamination.

## Human text (training targets, Stage A denoising, eval references)

| Source | Dataset | Snapshot / filter | License | Notes |
|---|---|---|---|---|
| Educational web | `HuggingFaceFW/fineweb-edu` | CommonCrawl dumps 2019-30 … 2022-40, `int_score ≥ 3` | ODC-By 1.0 (+ CommonCrawl terms) | Explanatory register |
| Blogs / personal web | `HuggingFaceFW/fineweb` | Same dumps, URL filter (blogspot, wordpress, medium, substack, `/blog/`, dated paths) | ODC-By 1.0 (+ CommonCrawl terms) | Informal register, essays, how-tos |
| Encyclopedia | `legacy-datasets/wikipedia` | `20220301.en` | CC-BY-SA 3.0 / GFDL | **Share-alike.** Commonly accepted for training model weights; flagged on the model card |
| Narrative | `sedthh/gutenberg_english` | Project Gutenberg texts | Texts public domain (US); dataset card MIT | Older register, capped at 10% |

Chunking: paragraphs of 60–300 words (about 20% multi-paragraph), with boilerplate, citation and footnote filters. Chunks whose slop-lexicon density is above 1.5 per 100 words are dropped, since already-sloppy human text makes a bad target. Exact deduplication is by normalized hash.

The Stage A pool (`human_stageA.jsonl`) is built with `--exclude human_pilot.jsonl`, so no pilot (pair/test) document appears in Stage A.

**Not used:** Enron emails (PII; decision 2026-11-15).

## Slopified inputs (generated)

| Generator | HF id (as run) | License | Use |
|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | Apache-2.0 | train / val / test_in |
| Mistral-Small-3.2-24B-Instruct-2506 | `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8` (quantisation of the Apache-2.0 original) | Apache-2.0 | train / val / test_in |
| gpt-oss-20b | `openai/gpt-oss-20b` | Apache-2.0 | **held-out generator: test only** |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 | real-AI eval answers only |

No GPT/Claude/Gemini outputs are used anywhere in training, because their terms prohibit training competing models on their outputs.

## Eval-only resources

| Resource | License | Use |
|---|---|---|
| `N8Programs/unslop-good` | CC0-1.0 | **Eval only.** The generator of its AI side is undocumented, so it is not trained on |
| `sam-paech/antislop-sampler` slop lists (`slop_phrase_prob_adjustments.json`, `slop_regexes.txt`) | Apache-2.0 | Slop metric (copied into `eval/slop_lists/` with LICENSE) |
| `sam-paech/auto-antislop` | **no license** | Not used |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | MIT | NLI filter and metric |
| `sentence-transformers/all-mpnet-base-v2` | Apache-2.0 | Embedding similarity metric |
| `en_core_web_sm` (spaCy) | MIT | NER for fact checks |

## External reference models (inference only)

| Model | License | Note |
|---|---|---|
| `Oysiyl/qwen3-0.6b-unslop-good-lora-v1` | Apache-2.0 | LoRA on Qwen3-0.6B |
| `bingbangboom/dolus-v3-ep1-instruct-GGUF` | CC-BY-NC-SA-4.0 | `dolus-v2` (named in the brief) is not public; v3 is used instead |
