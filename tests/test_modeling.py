import pytest
import torch
from transformers import ModernBertDecoderForCausalLM

from whittle.modeling import EttinSeq2Seq, decode, load_tokenizer

ENC, DEC = "jhu-clsp/ettin-encoder-32m", "jhu-clsp/ettin-decoder-32m"


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(DEC)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return EttinSeq2Seq.from_ettin(ENC, DEC).eval()


@pytest.fixture(scope="module")
def batch(tok):
    src = tok(["The quick brown fox jumps over the lazy dog.", "In conclusion, it is crucial to delve deeper."],
              return_tensors="pt", padding=True)
    tgt = tok(["A fox jumped over a dog.", "We should look closer, then."], return_tensors="pt", padding=True)
    return src, tgt


def test_param_budget(model):
    br = model.parameter_breakdown()
    print(br)
    assert 45e6 < br["total"] < 90e6
    # tied: a single embedding matrix
    e = model.decoder.model.embeddings.tok_embeddings.weight
    assert model.decoder.decoder.weight is e
    assert model.encoder.embeddings.tok_embeddings.weight is e


def test_shapes_and_loss(model, batch):
    src, tgt = batch
    labels = tgt.input_ids.masked_fill(tgt.attention_mask == 0, -100)
    dec_in = model.shift_right(labels)
    out = model(input_ids=src.input_ids, attention_mask=src.attention_mask, decoder_input_ids=dec_in,
                labels=labels, return_logits=True)
    assert out.logits.shape == (*dec_in.shape, model.config.vocab_size)
    assert out.encoder_last_hidden_state.shape == (*src.input_ids.shape, 384)
    assert torch.isfinite(out.loss)
    # chunked loss equals plain cross entropy
    ref = torch.nn.functional.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), labels.view(-1),
                                            ignore_index=-100)
    torch.testing.assert_close(out.loss, ref, rtol=1e-5, atol=1e-5)


def test_zero_gate_equals_pretrained_decoder(model, batch):
    src, tgt = batch
    ref = ModernBertDecoderForCausalLM.from_pretrained(DEC, attn_implementation="sdpa").eval()
    ids = tgt.input_ids
    with torch.no_grad():
        want = ref(input_ids=ids).logits
        got = model(input_ids=src.input_ids, attention_mask=src.attention_mask, decoder_input_ids=ids).logits
    assert all(float(model.xattn[k].gate) == 0.0 for k in model.xattn)
    assert torch.equal(got, want), (got - want).abs().max()


def test_zero_outproj_init_equals_pretrained_and_learns(batch):
    src, tgt = batch
    torch.manual_seed(0)
    m = EttinSeq2Seq.from_ettin(ENC, DEC, xattn_zero_init="out_proj", xattn_gate_init=1.0).eval()
    ref = ModernBertDecoderForCausalLM.from_pretrained(DEC, attn_implementation="sdpa").eval()
    with torch.no_grad():
        want = ref(input_ids=tgt.input_ids).logits
        got = m(input_ids=src.input_ids, attention_mask=src.attention_mask, decoder_input_ids=tgt.input_ids).logits
    assert torch.equal(got, want)
    m.train()
    labels = tgt.input_ids.masked_fill(tgt.attention_mask == 0, -100)
    m(input_ids=src.input_ids, attention_mask=src.attention_mask, labels=labels).loss.backward()
    assert m.xattn["0"].o_proj.weight.grad.abs().sum() > 0  # learns from step 1


def test_gradients_reach_gates_and_then_xattn(batch):
    torch.manual_seed(0)
    m = EttinSeq2Seq.from_ettin(ENC, DEC).train()
    src, tgt = batch
    labels = tgt.input_ids.masked_fill(tgt.attention_mask == 0, -100)
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if n.startswith("xattn.")], lr=1e-3)
    for step in range(2):
        opt.zero_grad()
        loss = m(input_ids=src.input_ids, attention_mask=src.attention_mask, labels=labels).loss
        loss.backward()
        g = m.xattn["0"]
        if step == 0:
            assert g.gate.grad is not None and g.gate.grad.abs() > 0  # gate learns first
        else:
            assert g.q_proj.weight.grad.abs().sum() > 0  # then the attention itself
        opt.step()


def test_gradient_checkpointing_matches(batch):
    torch.manual_seed(0)
    m = EttinSeq2Seq.from_ettin(ENC, DEC, xattn_gate_init=0.1).train()
    src, tgt = batch
    labels = tgt.input_ids.masked_fill(tgt.attention_mask == 0, -100)
    l1 = m(input_ids=src.input_ids, attention_mask=src.attention_mask, labels=labels).loss
    l1.backward()
    g1 = m.xattn["3"].k_proj.weight.grad.clone()
    m.zero_grad()
    m.gradient_checkpointing_enable()
    l2 = m(input_ids=src.input_ids, attention_mask=src.attention_mask, labels=labels).loss
    l2.backward()
    torch.testing.assert_close(l1, l2)
    torch.testing.assert_close(g1, m.xattn["3"].k_proj.weight.grad, rtol=1e-4, atol=1e-6)


def test_save_load_roundtrip(tmp_path, batch):
    torch.manual_seed(0)
    m = EttinSeq2Seq.from_ettin(ENC, DEC, xattn_gate_init=0.3).eval()
    m.save_pretrained(tmp_path)
    m2 = EttinSeq2Seq.from_pretrained(tmp_path).eval()
    src, tgt = batch
    with torch.no_grad():
        a = m(input_ids=src.input_ids, attention_mask=src.attention_mask, decoder_input_ids=tgt.input_ids).logits
        b = m2(input_ids=src.input_ids, attention_mask=src.attention_mask, decoder_input_ids=tgt.input_ids).logits
    assert torch.equal(a, b)
    e = m2.decoder.model.embeddings.tok_embeddings.weight
    assert m2.decoder.decoder.weight is e and m2.encoder.embeddings.tok_embeddings.weight is e


def test_untied_encoder_embeddings():
    m = EttinSeq2Seq.from_ettin(ENC, DEC, tie_encoder_embeddings=False)
    assert m.encoder.embeddings.tok_embeddings.weight is not m.decoder.model.embeddings.tok_embeddings.weight
    assert m.parameter_breakdown()["total"] > 65e6


@pytest.fixture(scope="module")
def gated_model():
    torch.manual_seed(0)
    m = EttinSeq2Seq.from_ettin(ENC, DEC, xattn_gate_init=0.5).eval()
    return m


def test_cached_generation_equals_uncached(gated_model, batch):
    src, _ = batch
    # 90 tokens > sliding window (64) to exercise sliding-window cache layers
    kw = dict(input_ids=src.input_ids, attention_mask=src.attention_mask, max_new_tokens=90, min_new_tokens=90)
    a = gated_model.generate(**kw, use_cache=True)
    b = gated_model.generate(**kw, use_cache=False)
    assert a.shape[1] == 90
    assert torch.equal(a, b)


def test_beam_and_sampling_run(gated_model, batch, tok):
    src, _ = batch
    beam = gated_model.generate(src.input_ids, src.attention_mask, max_new_tokens=20, num_beams=4)
    beam_nc = gated_model.generate(src.input_ids, src.attention_mask, max_new_tokens=20, num_beams=4, use_cache=False)
    assert torch.equal(beam, beam_nc)
    g = torch.Generator().manual_seed(0)
    samp = gated_model.generate(src.input_ids, src.attention_mask, max_new_tokens=20, do_sample=True, top_p=0.9,
                                generator=g)
    assert beam.shape[0] == samp.shape[0] == 2
    assert all(isinstance(s, str) for s in decode(tok, samp))


def test_batch_padding_invariance(gated_model, tok):
    """A sequence's output must not change when it is batched with a longer (padded) one."""
    s = "Short input."
    one = tok([s], return_tensors="pt")
    two = tok([s, "A much longer input sentence that forces padding of the first."], return_tensors="pt", padding=True)
    a = gated_model.generate(one.input_ids, one.attention_mask, max_new_tokens=12, min_new_tokens=12)
    b = gated_model.generate(two.input_ids, two.attention_mask, max_new_tokens=12, min_new_tokens=12)
    assert torch.equal(a[0], b[0])


CASES = [
    "Hello world.\n\nNew para\n\n\n  indented\ttab",
    "# Title\n## Sub\n- item\n* star\n1. one\n",
    "`code` and ```python\ndef f(x):\n    return {x: [1,2]}\n```",
    "| a | b |\n|---|---|\n| 1 | 2 |",
    "[link](https://x.com/a_b?q=1) **bold** _it_",
    "“curly” ‘single’ — em – en … , . ; ! ?",
    "emoji \U0001F680\U0001F44D\U0001F3FD \U0001F1FA\U0001F1F8 family \U0001F468‍\U0001F469‍\U0001F467",
    " leading space", "trailing space ", "naïve café Über 東京 3.14e-10 $1,234.56 50% (a) [b] {c}",
    "Wait . . . what ? Yes !  Two  spaces.",
]


@pytest.mark.parametrize("text", CASES)
def test_tokenizer_roundtrip(tok, text):
    ids = tok(text, add_special_tokens=True).input_ids
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    assert decode(tok, [ids])[0] == text
    # word-split path (ModernBERT issue #149 concerns is_split_into_words); our pipeline never uses it,
    # but make sure plain encoding is unaffected by add_prefix_space
    assert tok(text, add_special_tokens=False).input_ids == tok.encode(text, add_special_tokens=False)


def test_default_decode_is_exact(tok):
    # tokenizer_config sets clean_up_tokenization_spaces=True, which strips spaces before punctuation on
    # older transformers. transformers>=5.2x ignores it for BPE; our decode() disables it regardless.
    ids = tok("Wait . . . what ?").input_ids
    assert decode(tok, [ids])[0] == "Wait . . . what ?"
    assert tok.decode(ids, skip_special_tokens=True) == "Wait . . . what ?"
