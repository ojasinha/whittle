"""EttinSeq2Seq: a warm-started encoder-decoder built from Ettin encoder + Ettin decoder.

Design
------
* Encoder: ``ModernBertModel`` (bidirectional) from an Ettin encoder checkpoint.
* Decoder: ``ModernBertDecoderForCausalLM`` from an Ettin decoder checkpoint. We re-run its layer loop
  ourselves so we can insert a gated cross-attention sub-layer after each self-attention:

      h = h + SelfAttn(attn_norm(h))                                # pretrained, unchanged
      h = h + tanh(g_l) * XAttn(xattn_norm(h), enc_out, enc_mask)   # new, g_l = 0 at init
      h = h + MLP(mlp_norm(h))                                      # pretrained, unchanged

  With g_l = 0 the model is *exactly* the pretrained decoder (tested).
* Embeddings: encoder input, decoder input and LM head share one matrix (optional for the encoder).
* Loss is computed in token chunks under activation checkpointing so the [tokens x 50k] logits are never
  materialised at once (matters on small GPUs).
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import ModernBertConfig, ModernBertModel, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.modernbert_decoder import ModernBertDecoderConfig, ModernBertDecoderForCausalLM

try:  # transformers >= 5
    from transformers import PreTrainedConfig as _BaseConfig
except ImportError:  # pragma: no cover
    from transformers import PretrainedConfig as _BaseConfig


class EttinSeq2SeqConfig(_BaseConfig):
    model_type = "ettin-seq2seq"

    def __init__(
        self,
        encoder: dict | None = None,
        decoder: dict | None = None,
        tie_encoder_embeddings: bool = True,
        xattn_gate_init: float = 0.0,
        xattn_zero_init: str = "gate",
        xattn_every: int = 1,
        label_smoothing: float = 0.0,
        loss_chunk_tokens: int = 4096,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        kwargs.setdefault("is_encoder_decoder", True)
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)
        self.encoder = dict(encoder or {})
        self.decoder = dict(decoder or {})
        self.tie_encoder_embeddings = tie_encoder_embeddings
        self.xattn_gate_init = xattn_gate_init
        # "gate": gate=xattn_gate_init (0 => exact pretrained decoder), random out-proj  (Flamingo style)
        # "out_proj": out-proj=0 (=> exact pretrained decoder for any gate), gate=xattn_gate_init (e.g. 1.0)
        self.xattn_zero_init = xattn_zero_init
        self.xattn_every = xattn_every
        self.label_smoothing = label_smoothing
        self.loss_chunk_tokens = loss_chunk_tokens
        self.initializer_range = initializer_range
        dec = self.decoder
        self.vocab_size = dec.get("vocab_size", 50368)
        self.pad_token_id = dec.get("pad_token_id", 50283)
        self.bos_token_id = dec.get("bos_token_id", 50281)
        self.eos_token_id = dec.get("eos_token_id", 50282)
        self.decoder_start_token_id = self.bos_token_id


class GatedCrossAttention(nn.Module):
    """Pre-norm multi-head cross-attention with a tanh gate (Flamingo-style)."""

    def __init__(self, d_model: int, d_enc: int, n_heads: int, eps: float, gate_init: float = 0.0, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.norm = nn.LayerNorm(d_model, eps=eps, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_enc, d_model, bias=False)
        self.v_proj = nn.Linear(d_enc, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.dropout = dropout

    def project_kv(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = enc.shape
        k = self.k_proj(enc).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, h: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor], enc_mask: torch.Tensor | None) -> torch.Tensor:
        b, t, d = h.shape
        q = self.q_proj(self.norm(h)).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = kv
        out = F.scaled_dot_product_attention(
            q, k.to(q.dtype), v.to(q.dtype), attn_mask=enc_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, d)
        return torch.tanh(self.gate).to(out.dtype) * self.o_proj(out)


def _sub_config(cls, d: dict):
    d = {k: v for k, v in d.items() if k not in ("transformers_version", "architectures", "_name_or_path")}
    cfg = cls(**d)
    cfg._attn_implementation = "sdpa"
    return cfg


class EttinSeq2Seq(PreTrainedModel):
    config_class = EttinSeq2SeqConfig
    base_model_prefix = "ettin_seq2seq"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    main_input_name = "input_ids"

    def __init__(self, config: EttinSeq2SeqConfig):
        super().__init__(config)
        enc_cfg = _sub_config(ModernBertConfig, config.encoder)
        dec_cfg = _sub_config(ModernBertDecoderConfig, config.decoder)
        self.encoder = ModernBertModel(enc_cfg)
        self.decoder = ModernBertDecoderForCausalLM(dec_cfg)
        d, de = dec_cfg.hidden_size, enc_cfg.hidden_size
        n = dec_cfg.num_hidden_layers
        self.xattn_layers = [i for i in range(n) if i % config.xattn_every == config.xattn_every - 1 or config.xattn_every == 1]
        self.xattn = nn.ModuleDict(
            {str(i): GatedCrossAttention(d, de, dec_cfg.num_attention_heads, dec_cfg.norm_eps, config.xattn_gate_init)
             for i in self.xattn_layers}
        )
        if config.tie_encoder_embeddings:
            if de != d:
                raise ValueError(f"Cannot tie embeddings: encoder width {de} != decoder width {d}")
            self._tied_weights_keys = {
                "encoder.embeddings.tok_embeddings.weight": "decoder.model.embeddings.tok_embeddings.weight"
            }
        else:
            self._tied_weights_keys = {}
        self.gradient_checkpointing = False
        self.post_init()
        self._tie_shared()

    # ----------------------------------------------------------------- weights / ties
    def _tie_shared(self):
        shared = self.decoder.model.embeddings.tok_embeddings.weight
        self.decoder.decoder.weight = shared
        if self.config.tie_encoder_embeddings:
            self.encoder.embeddings.tok_embeddings.weight = shared

    def tie_weights(self, *args, **kwargs):
        out = super().tie_weights(*args, **kwargs)
        self._tie_shared()
        return out

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, GatedCrossAttention):
            std = self.config.initializer_range
            n = max(1, len(self.xattn_layers))
            for lin, s in ((module.q_proj, std), (module.k_proj, std), (module.v_proj, std), (module.o_proj, std / math.sqrt(2 * n))):
                nn.init.trunc_normal_(lin.weight, std=s, a=-2 * s, b=2 * s)
            if self.config.xattn_zero_init == "out_proj":
                nn.init.zeros_(module.o_proj.weight)
            nn.init.ones_(module.norm.weight)
            module.gate.fill_(self.config.xattn_gate_init)
        elif module is self:
            return
        # encoder/decoder submodules initialise themselves in their own constructors

    def get_input_embeddings(self):
        return self.decoder.model.embeddings.tok_embeddings

    def get_output_embeddings(self):
        return self.decoder.decoder

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.encoder.gradient_checkpointing_disable()

    @classmethod
    def from_ettin(
        cls,
        encoder_id: str = "jhu-clsp/ettin-encoder-32m",
        decoder_id: str = "jhu-clsp/ettin-decoder-32m",
        tie_encoder_embeddings: bool = True,
        embedding_source: str = "decoder",
        **config_kwargs,
    ) -> "EttinSeq2Seq":
        enc = ModernBertModel.from_pretrained(encoder_id, attn_implementation="sdpa")
        dec = ModernBertDecoderForCausalLM.from_pretrained(decoder_id, attn_implementation="sdpa")
        cfg = EttinSeq2SeqConfig(
            encoder=enc.config.to_dict(), decoder=dec.config.to_dict(),
            tie_encoder_embeddings=tie_encoder_embeddings, **config_kwargs,
        )
        model = cls(cfg)
        enc_emb = enc.embeddings.tok_embeddings.weight.detach().clone()
        missing, unexpected = model.encoder.load_state_dict(enc.state_dict(), strict=False)
        assert not unexpected, unexpected
        missing, unexpected = model.decoder.load_state_dict(dec.state_dict(), strict=False)
        assert not unexpected, unexpected
        model._tie_shared()
        if tie_encoder_embeddings and embedding_source == "encoder":
            with torch.no_grad():
                model.decoder.model.embeddings.tok_embeddings.weight.copy_(enc_emb)
        return model

    # ----------------------------------------------------------------- parameter report
    def parameter_breakdown(self) -> dict[str, int]:
        seen, out = set(), {}

        def count(name, params):
            n = 0
            for p in params:
                if id(p) not in seen:
                    seen.add(id(p))
                    n += p.numel()
            out[name] = n

        emb = self.decoder.model.embeddings.tok_embeddings.weight
        count("shared_embedding", [emb])
        count("encoder_body", self.encoder.parameters())
        count("decoder_body", [p for n_, p in self.decoder.named_parameters() if not n_.startswith(("lm_head", "decoder."))])
        count("lm_head (dense+norm+bias)", [p for n_, p in self.decoder.named_parameters() if n_.startswith(("lm_head", "decoder."))])
        count("cross_attention", self.xattn.parameters())
        out["total"] = sum(out.values())
        return out

    def new_parameter_names(self) -> set[str]:
        return {n for n, _ in self.named_parameters() if n.startswith("xattn.")}

    # ----------------------------------------------------------------- forward pieces
    def encode(self, input_ids, attention_mask=None):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def cross_kv(self, enc_out):
        return {i: self.xattn[str(i)].project_kv(enc_out) for i in self.xattn_layers}

    @staticmethod
    def _enc_mask(attention_mask, dtype=None):
        if attention_mask is None:
            return None
        return attention_mask[:, None, None, :].bool()

    def _layer(self, i, layer, mask, pe, past, kv, enc_mask, h):
        res = h
        x = layer.attn(hidden_states=layer.attn_norm(h), position_embeddings=pe, attention_mask=mask,
                       past_key_values=past)[0]
        h = res + x
        if kv is not None and i in kv:
            h = h + self.xattn[str(i)](h, kv[i], enc_mask)
        h = h + layer.mlp(layer.mlp_norm(h))
        return h

    def decode(self, decoder_input_ids, decoder_attention_mask=None, kv=None, enc_mask=None,
               past_key_values=None, use_cache=False):
        dm = self.decoder.model
        h = dm.embeddings(input_ids=decoder_input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=dm.config)
        b, t = decoder_input_ids.shape
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        pos = (torch.arange(t, device=h.device) + past_len).unsqueeze(0).expand(b, -1)
        mk = dict(config=dm.config, inputs_embeds=h, attention_mask=decoder_attention_mask,
                  past_key_values=past_key_values, position_ids=pos)
        masks = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
        pes = {lt: dm.rotary_emb(h, pos, lt) for lt in set(dm.config.layer_types)}
        for i, layer in enumerate(dm.layers):
            lt = dm.config.layer_types[i]
            fn = partial(self._layer, i, layer, masks[lt], pes[lt], past_key_values, kv, enc_mask)
            if self.gradient_checkpointing and self.training:
                h = checkpoint(fn, h, use_reentrant=False)
            else:
                h = fn(h)
        return dm.final_norm(h), past_key_values

    def logits_from_hidden(self, h):
        return self.decoder.decoder(self.decoder.lm_head(h))

    def _chunked_loss(self, h, labels):
        sel = labels != -100
        h, y = h[sel], labels[sel]
        if y.numel() == 0:
            return h.sum() * 0.0
        chunk = self.config.loss_chunk_tokens
        total = h.new_zeros((), dtype=torch.float32)

        def piece(hc, yc):
            logits = self.logits_from_hidden(hc).float()
            return F.cross_entropy(logits, yc, reduction="sum", label_smoothing=self.config.label_smoothing)

        for s in range(0, y.numel(), chunk):
            hc, yc = h[s:s + chunk], y[s:s + chunk]
            total = total + (checkpoint(piece, hc, yc, use_reentrant=False) if self.training else piece(hc, yc))
        return total / y.numel()

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                labels=None, encoder_outputs=None, return_logits=None, **kwargs):
        enc = encoder_outputs if encoder_outputs is not None else self.encode(input_ids, attention_mask)
        if decoder_input_ids is None:
            assert labels is not None, "need decoder_input_ids or labels"
            decoder_input_ids = self.shift_right(labels)
        h, _ = self.decode(decoder_input_ids, decoder_attention_mask, self.cross_kv(enc), self._enc_mask(attention_mask))
        loss = self._chunked_loss(h, labels) if labels is not None else None
        if return_logits is None:
            return_logits = labels is None
        logits = self.logits_from_hidden(h) if return_logits else None
        return Seq2SeqLMOutput(loss=loss, logits=logits, encoder_last_hidden_state=enc)

    def shift_right(self, labels):
        start = torch.full_like(labels[:, :1], self.config.decoder_start_token_id)
        x = torch.cat([start, labels[:, :-1]], dim=1)
        return x.masked_fill(x == -100, self.config.pad_token_id)

    # ----------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=None, max_length_ratio=2.0, min_new_tokens=0,
                 num_beams=1, do_sample=False, top_p=1.0, temperature=1.0, length_penalty=1.0,
                 repetition_penalty=1.0, no_repeat_ngram_size=0, use_cache=True, generator=None):
        """Returns generated token ids (B, L) excluding the start token, right-padded with pad_token_id."""
        self.eval()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if max_new_tokens is None:
            src_len = int(attention_mask.sum(1).max())
            max_new_tokens = max(16, int(src_len * max_length_ratio))
        enc = self.encode(input_ids, attention_mask)
        opts = dict(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size, use_cache=use_cache)
        if num_beams > 1:
            return self._beam_search(enc, attention_mask, num_beams=num_beams, length_penalty=length_penalty, **opts)
        return self._sample(enc, attention_mask, do_sample=do_sample, top_p=top_p, temperature=temperature,
                            generator=generator, **opts)

    def _step_logits(self, seqs, kv, enc_mask, past, use_cache):
        if use_cache:
            inp = seqs if past is None or past.get_seq_length() == 0 else seqs[:, -1:]
            h, past = self.decode(inp, kv=kv, enc_mask=enc_mask, past_key_values=past, use_cache=True)
        else:
            h, _ = self.decode(seqs, kv=kv, enc_mask=enc_mask)
        return self.logits_from_hidden(h[:, -1]).float(), past

    def _process(self, logits, seqs, step, min_new_tokens, repetition_penalty, no_repeat_ngram_size):
        eos = self.config.eos_token_id
        logits[:, self.config.pad_token_id] = -float("inf")
        logits[:, self.config.bos_token_id] = -float("inf")
        if step < min_new_tokens:
            logits[:, eos] = -float("inf")
        if repetition_penalty != 1.0:
            prev = seqs
            score = logits.gather(1, prev)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits.scatter_(1, prev, score)
        n = no_repeat_ngram_size
        if n > 0 and seqs.shape[1] >= n:
            s = seqs.tolist()
            for b, row in enumerate(s):
                prefix = tuple(row[-(n - 1):]) if n > 1 else ()
                banned = [row[j + n - 1] for j in range(len(row) - n + 1) if tuple(row[j:j + n - 1]) == prefix]
                if banned:
                    logits[b, banned] = -float("inf")
        return logits

    def _sample(self, enc, attention_mask, max_new_tokens, min_new_tokens, do_sample, top_p, temperature,
                repetition_penalty, no_repeat_ngram_size, use_cache, generator):
        b = enc.shape[0]
        kv, enc_mask = self.cross_kv(enc), self._enc_mask(attention_mask)
        seqs = torch.full((b, 1), self.config.decoder_start_token_id, dtype=torch.long, device=enc.device)
        done = torch.zeros(b, dtype=torch.bool, device=enc.device)
        past = None
        for step in range(max_new_tokens):
            logits, past = self._step_logits(seqs, kv, enc_mask, past, use_cache)
            logits = self._process(logits, seqs, step, min_new_tokens, repetition_penalty, no_repeat_ngram_size)
            if do_sample:
                logits = logits / max(temperature, 1e-5)
                probs = logits.softmax(-1)
                if top_p < 1.0:
                    sp, si = probs.sort(-1, descending=True)
                    keep = sp.cumsum(-1) - sp < top_p
                    sp = sp * keep
                    probs = torch.zeros_like(probs).scatter_(1, si, sp)
                nxt = torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, generator=generator).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.config.pad_token_id), nxt)
            seqs = torch.cat([seqs, nxt[:, None]], 1)
            done |= nxt == self.config.eos_token_id
            if bool(done.all()):
                break
        return self._strip(seqs[:, 1:])

    def _strip(self, out):
        """Replace EOS and everything after it with pad."""
        eos, pad = self.config.eos_token_id, self.config.pad_token_id
        after = (out == eos).int().cumsum(1) > 0
        return out.masked_fill(after, pad)

    def _beam_search(self, enc, attention_mask, num_beams, max_new_tokens, min_new_tokens, length_penalty,
                     repetition_penalty, no_repeat_ngram_size, use_cache):
        k, (b, dev) = num_beams, (enc.shape[0], enc.device)
        eos = self.config.eos_token_id
        enc = enc.repeat_interleave(k, 0)
        am = attention_mask.repeat_interleave(k, 0)
        kv, enc_mask = self.cross_kv(enc), self._enc_mask(am)
        seqs = torch.full((b * k, 1), self.config.decoder_start_token_id, dtype=torch.long, device=dev)
        scores = torch.zeros(b, k, device=dev)
        scores[:, 1:] = -1e9
        finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(b)]
        past = None
        for step in range(max_new_tokens):
            logits, past = self._step_logits(seqs, kv, enc_mask, past, use_cache)
            logits = self._process(logits, seqs, step, min_new_tokens, repetition_penalty, no_repeat_ngram_size)
            logp = logits.log_softmax(-1)
            v = logp.shape[-1]
            cand = (scores.view(-1, 1) + logp).view(b, k * v)
            top_s, top_i = cand.topk(2 * k, dim=1)
            src_beam, tok = top_i // v, top_i % v
            new_scores = torch.full((b, k), -1e9, device=dev)
            new_beam = torch.zeros(b, k, dtype=torch.long, device=dev)
            new_tok = torch.full((b, k), self.config.pad_token_id, dtype=torch.long, device=dev)
            ts, tb, tt = top_s.tolist(), src_beam.tolist(), tok.tolist()
            cur_len = seqs.shape[1]  # includes start token; hypothesis length = cur_len
            for bi in range(b):
                j = 0
                for r in range(2 * k):
                    s, sb, t = ts[bi][r], tb[bi][r], tt[bi][r]
                    if s < -1e8:
                        continue
                    if t == eos:
                        if r < k and len(finished[bi]) < k * 4:
                            hyp = seqs[bi * k + sb, 1:].tolist()
                            finished[bi].append((s / (max(1, len(hyp) + 1) ** length_penalty), hyp))
                        continue
                    if j < k:
                        new_scores[bi, j], new_beam[bi, j], new_tok[bi, j] = s, sb, t
                        j += 1
            # stop for a batch item once it has k finished hyps whose worst beats the best alive possible
            beam_idx = (torch.arange(b, device=dev)[:, None] * k + new_beam).view(-1)
            seqs = torch.cat([seqs[beam_idx], new_tok.view(-1, 1)], 1)
            if past is not None:
                past.reorder_cache(beam_idx)
            scores = new_scores
            all_done = True
            for bi in range(b):
                if len(finished[bi]) < k:
                    all_done = False
                    break
                best_alive = scores[bi].max().item() / (cur_len ** length_penalty)
                if max(f[0] for f in finished[bi]) < best_alive:
                    all_done = False
                    break
            if all_done:
                break
        outs = []
        for bi in range(b):
            if not finished[bi]:
                j = int(scores[bi].argmax())
                finished[bi].append((0.0, seqs[bi * k + j, 1:].tolist()))
            outs.append(max(finished[bi], key=lambda x: x[0])[1])
        width = max(1, max(len(o) for o in outs))
        res = torch.full((b, width), self.config.pad_token_id, dtype=torch.long, device=dev)
        for bi, o in enumerate(outs):
            if o:
                res[bi, :len(o)] = torch.tensor(o, device=dev)
        return res


def load_tokenizer(name: str = "jhu-clsp/ettin-decoder-32m"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    return tok


def decode(tok, ids) -> list[str]:
    """Decode without the tokenizer's default space clean-up (which corrupts text such as ' .')."""
    return tok.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
