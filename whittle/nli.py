"""Batched NLI entailment scorer (DeBERTa-v3 MNLI/FEVER/ANLI, MIT)."""
from __future__ import annotations

import torch
from tqdm import tqdm


class NLIScorer:
    def __init__(self, name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device=None, max_length=1024,
                 batch_size=8):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ent = lab["entailment"]
        self.max_length, self.batch_size = max_length, batch_size

    @torch.no_grad()
    def entail(self, premises, hypotheses, progress=True):
        order = sorted(range(len(premises)), key=lambda i: len(premises[i]) + len(hypotheses[i]))
        out = [0.0] * len(premises)
        it = range(0, len(order), self.batch_size)
        for s in tqdm(it, disable=not progress, desc="nli"):
            idx = order[s:s + self.batch_size]
            enc = self.tok([premises[i] for i in idx], [hypotheses[i] for i in idx], truncation=True,
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with torch.autocast(self.device if isinstance(self.device, str) else self.device.type,
                                dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
                logits = self.model(**enc).logits.float()
            p = logits.softmax(-1)[:, self.ent].tolist()
            for i, v in zip(idx, p):
                out[i] = v
        return out
