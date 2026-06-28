from __future__ import annotations

import torch
import torch.nn.functional as F

from ai_unity.text_lm import TinyLMConfig, TinyTransformerLM


def test_tiny_transformer_lm_forward_backward():
    config = TinyLMConfig(vocab_size=128, seq_len=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = TinyTransformerLM(config)
    x = torch.randint(0, config.vocab_size, (3, config.seq_len))
    y = torch.randint(0, config.vocab_size, (3, config.seq_len))

    logits, aux_loss = model(x)
    assert logits.shape == (3, config.seq_len, config.vocab_size)
    assert aux_loss.shape == ()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)

    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) + aux_loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_tiny_moe_transformer_lm_forward_backward():
    config = TinyLMConfig(
        vocab_size=128,
        seq_len=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        model_kind="moe",
        num_experts=3,
        top_k=2,
    )
    model = TinyTransformerLM(config)
    x = torch.randint(0, config.vocab_size, (3, config.seq_len))
    y = torch.randint(0, config.vocab_size, (3, config.seq_len))

    logits, aux_loss = model(x)
    assert logits.shape == (3, config.seq_len, config.vocab_size)
    assert aux_loss.shape == ()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)

    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) + aux_loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert len(model.routing_summaries(x)) == config.n_layers
