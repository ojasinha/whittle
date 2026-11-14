"""Uniform interface over the three model families we compare.

    w = build_wrapper(cfg)            # cfg: dict with type / init / options
    loss = w.loss(batch)              # batch from w.collate([...])
    texts = w.generate_texts(srcs, num_beams=4)
    w.save(path); w = load_wrapper(path)

Types
  ettin_s2s : Plan A (EttinSeq2Seq, gated cross-attention)
  decoder   : B1 decoder-only Ettin, "[CLS] src [SEP] tgt [SEP]", loss on tgt only
  bart      : B2 facebook/bart-base
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .modeling import EttinSeq2Seq, decode


def chunked_ce(h, labels, head_fn, chunk=4096, label_smoothing=0.0, training=True):
    """Cross-entropy over selected positions without materialising all logits. h:[...,D], labels:[...]"""
    sel = labels != -100
    h, y = h[sel], labels[sel]
    if y.numel() == 0:
        return h.sum() * 0.0
    total = h.new_zeros((), dtype=torch.float32)

    def piece(hc, yc):
        return F.cross_entropy(head_fn(hc).float(), yc, reduction="sum", label_smoothing=label_smoothing)

    for s in range(0, y.numel(), chunk):
        hc, yc = h[s:s + chunk], y[s:s + chunk]
        total = total + (checkpoint(piece, hc, yc, use_reentrant=False) if training else piece(hc, yc))
    return total / y.numel()


class BaseWrapper:
    type: str
    max_src: int = 512
    max_tgt: int = 512

    def __init__(self, model, tok, cfg):
        self.model, self.tok, self.cfg = model, tok, cfg
        self.max_src = cfg.get("max_src", 512)
        self.max_tgt = cfg.get("max_tgt", 512)
        self.label_smoothing = cfg.get("label_smoothing", 0.0)

    # parameter groups: names of newly-initialised params get lr_new
    def new_param_names(self) -> set[str]:
        return set()

    def to(self, device):
        self.model.to(device)
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        with open(os.path.join(path, "wrapper.json"), "w") as f:
            json.dump({**self.cfg, "type": self.type, "init": path}, f, indent=2)

    def n_tokens(self, batch) -> int:
        n = int(batch["attention_mask"].sum())
        if "labels" in batch:
            n += int((batch["labels"] != -100).sum())
        return n


class EttinS2SWrapper(BaseWrapper):
    type = "ettin_s2s"

    def new_param_names(self):
        return self.model.new_parameter_names()

    def collate(self, pairs):
        src = [p[0] for p in pairs]
        tgt = [p[1] for p in pairs]
        enc = self.tok(src, truncation=True, max_length=self.max_src, padding=True, return_tensors="pt")
        t = self.tok(tgt, truncation=True, max_length=self.max_tgt + 1, add_special_tokens=False)["input_ids"]
        eos, pad = self.tok.sep_token_id, self.tok.pad_token_id
        t = [x[:self.max_tgt - 1] + [eos] for x in t]
        L = max(len(x) for x in t)
        labels = torch.full((len(t), L), -100, dtype=torch.long)
        for i, x in enumerate(t):
            labels[i, :len(x)] = torch.tensor(x)
        dec_in = self.model.shift_right(labels)
        dec_mask = (labels != -100).long()
        dec_mask[:, 0] = 1
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labels,
                "decoder_input_ids": dec_in, "decoder_attention_mask": dec_mask}

    def loss(self, batch):
        b = {k: v.to(self.device) for k, v in batch.items()}
        return self.model(**b).loss

    @torch.no_grad()
    def generate_texts(self, srcs, batch_size=16, **gen):
        self.model.eval()
        outs = []
        for i in range(0, len(srcs), batch_size):
            enc = self.tok(srcs[i:i + batch_size], truncation=True, max_length=self.max_src, padding=True,
                           return_tensors="pt").to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                ids = self.model.generate(enc.input_ids, enc.attention_mask, **gen)
            outs.extend(decode(self.tok, ids))
        return [o.strip() for o in outs]

    def gate_values(self):
        return {k: float(torch.tanh(m.gate)) for k, m in self.model.xattn.items()}


class DecoderWrapper(BaseWrapper):
    type = "decoder"

    def collate(self, pairs):
        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        seqs, labs = [], []
        for s, t in pairs:
            si = self.tok(s, add_special_tokens=False, truncation=True, max_length=self.max_src)["input_ids"]
            ti = self.tok(t, add_special_tokens=False, truncation=True, max_length=self.max_tgt - 1)["input_ids"]
            x = [cls] + si + [sep] + ti + [sep]
            seqs.append(x)
            labs.append([-100] * (len(si) + 2) + ti + [sep])
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        am = torch.zeros((len(seqs), L), dtype=torch.long)
        labels = torch.full((len(seqs), L), -100, dtype=torch.long)
        for i, (x, y) in enumerate(zip(seqs, labs)):
            ids[i, :len(x)] = torch.tensor(x)
            am[i, :len(x)] = 1
            labels[i, :len(y)] = torch.tensor(y)
        return {"input_ids": ids, "attention_mask": am, "labels": labels}

    def n_tokens(self, batch):
        return int(batch["attention_mask"].sum())

    def loss(self, batch):
        b = {k: v.to(self.device) for k, v in batch.items()}
        h = self.model.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        labels = b["labels"][:, 1:]
        h = h[:, :-1]
        head = lambda x: self.model.decoder(self.model.lm_head(x))  # noqa: E731
        return chunked_ce(h, labels, head, label_smoothing=self.label_smoothing, training=self.model.training)

    @torch.no_grad()
    def generate_texts(self, srcs, batch_size=16, num_beams=1, max_length_ratio=2.0, max_new_tokens=None,
                       no_repeat_ngram_size=0, repetition_penalty=1.0, do_sample=False, top_p=1.0, temperature=1.0,
                       **_):
        self.model.eval()
        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        outs = []
        for i in range(0, len(srcs), batch_size):
            chunk = srcs[i:i + batch_size]
            seqs = [[cls] + self.tok(s, add_special_tokens=False, truncation=True, max_length=self.max_src)["input_ids"] + [sep]
                    for s in chunk]
            L = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            am = torch.zeros_like(ids)
            for j, x in enumerate(seqs):  # left padding
                ids[j, L - len(x):] = torch.tensor(x)
                am[j, L - len(x):] = 1
            mnt = max_new_tokens or max(16, int(max(len(x) for x in seqs) * max_length_ratio))
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.model.generate(input_ids=ids.to(self.device), attention_mask=am.to(self.device),
                                          max_new_tokens=mnt, num_beams=num_beams, do_sample=do_sample, top_p=top_p,
                                          temperature=temperature, eos_token_id=sep, pad_token_id=pad,
                                          no_repeat_ngram_size=no_repeat_ngram_size,
                                          repetition_penalty=repetition_penalty)
            outs.extend(decode(self.tok, out[:, L:]))
        return [o.strip() for o in outs]


class BartWrapper(BaseWrapper):
    type = "bart"

    def collate(self, pairs):
        src = [p[0] for p in pairs]
        tgt = [p[1] for p in pairs]
        enc = self.tok(src, truncation=True, max_length=self.max_src, padding=True, return_tensors="pt")
        lab = self.tok(text_target=tgt, truncation=True, max_length=self.max_tgt, padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100)
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labels}

    def loss(self, batch):
        from transformers.models.bart.modeling_bart import shift_tokens_right

        b = {k: v.to(self.device) for k, v in batch.items()}
        cfg = self.model.config
        dec_in = shift_tokens_right(b["labels"], cfg.pad_token_id, cfg.decoder_start_token_id)
        out = self.model.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], decoder_input_ids=dec_in)
        h = out.last_hidden_state
        head = lambda x: self.model.lm_head(x) + self.model.final_logits_bias.to(x.dtype)[0]  # noqa: E731
        return chunked_ce(h, b["labels"], head, label_smoothing=self.label_smoothing, training=self.model.training)

    @torch.no_grad()
    def generate_texts(self, srcs, batch_size=16, num_beams=1, max_length_ratio=2.0, max_new_tokens=None,
                       no_repeat_ngram_size=0, repetition_penalty=1.0, do_sample=False, top_p=1.0, temperature=1.0,
                       **_):
        self.model.eval()
        outs = []
        for i in range(0, len(srcs), batch_size):
            enc = self.tok(srcs[i:i + batch_size], truncation=True, max_length=self.max_src, padding=True,
                           return_tensors="pt").to(self.device)
            mnt = max_new_tokens or max(16, int(enc.input_ids.shape[1] * max_length_ratio))
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.model.generate(**enc, max_new_tokens=mnt, num_beams=num_beams, do_sample=do_sample,
                                          top_p=top_p, temperature=temperature,
                                          no_repeat_ngram_size=no_repeat_ngram_size,
                                          repetition_penalty=repetition_penalty)
            outs.extend(self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        return [o.strip() for o in outs]


def build_wrapper(cfg: dict) -> BaseWrapper:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    t = cfg["type"]
    init = cfg["init"]
    if t == "ettin_s2s":
        if os.path.exists(os.path.join(str(init), "config.json")):
            model = EttinSeq2Seq.from_pretrained(init)
            tok = AutoTokenizer.from_pretrained(init)
        else:
            enc_id, dec_id = init["encoder"], init["decoder"]
            model = EttinSeq2Seq.from_ettin(enc_id, dec_id, tie_encoder_embeddings=cfg.get("tie_encoder_embeddings", True),
                                            embedding_source=cfg.get("embedding_source", "decoder"),
                                            xattn_gate_init=cfg.get("xattn_gate_init", 0.0),
                                            xattn_zero_init=cfg.get("xattn_zero_init", "gate"),
                                            label_smoothing=cfg.get("label_smoothing", 0.0))
            tok = AutoTokenizer.from_pretrained(dec_id)
        model.config.label_smoothing = cfg.get("label_smoothing", model.config.label_smoothing)
        return EttinS2SWrapper(model, tok, cfg)
    if t == "decoder":
        model = AutoModelForCausalLM.from_pretrained(init, attn_implementation="sdpa")
        tok = AutoTokenizer.from_pretrained(init)
        return DecoderWrapper(model, tok, cfg)
    if t == "bart":
        model = AutoModelForSeq2SeqLM.from_pretrained(init)
        tok = AutoTokenizer.from_pretrained(init)
        return BartWrapper(model, tok, cfg)
    raise ValueError(t)


def load_wrapper(path: str, **overrides) -> BaseWrapper:
    with open(os.path.join(path, "wrapper.json")) as f:
        cfg = json.load(f)
    cfg.update(overrides)
    cfg["init"] = path
    return build_wrapper(cfg)
