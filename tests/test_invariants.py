"""Hard-won invariants that must hold before any gated/fresh run. Pure-CPU, no network.

  python3 tests/test_invariants.py

Covers:
  - D5 gate equivalence: with the G1 gate zero-initialised, sigma(0)=0.5, so the gated
    attention output is EXACTLY 0.5x the ungated output at init (out_proj is linear, no bias).
    This is the equivalence check that must pass before training any gated arm.
  - per-KV-head config sanity (n_heads divisible by n_kv_heads; head_dim is direct ||.||, not a
    pooled proxy).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import VLMConfig
from models.language_model import LanguageModelGroupedQueryAttention, RotaryEmbedding


def test_gate_half_at_init():
    cfg = VLMConfig()
    cfg.lm_attn_gate = True
    torch.manual_seed(0)
    attn = LanguageModelGroupedQueryAttention(cfg).eval()
    torch.nn.init.zeros_(attn.gate_proj.weight)   # exactly as LanguageModel.__init__ does

    B, T = 2, 16
    x = torch.randn(B, T, cfg.lm_hidden_dim)
    cos, sin = RotaryEmbedding(cfg)(torch.arange(T).unsqueeze(0).expand(B, -1))

    y_gated = attn(x, cos, sin)
    attn.use_gate = False
    y_plain = attn(x, cos, sin)

    max_diff = (y_gated - 0.5 * y_plain).abs().max().item()
    assert max_diff < 1e-5, f"gate != 0.5x ungated at init (max diff {max_diff})"

    g = torch.sigmoid(attn.gate_proj(x))
    assert torch.allclose(g, torch.full_like(g, 0.5)), "gate_proj not zero-init -> sigma != 0.5"
    print(f"[ok] gate 0.5x-at-init equivalence (max diff {max_diff:.2e})")


def test_kv_head_config():
    cfg = VLMConfig()
    assert cfg.lm_n_heads % cfg.lm_n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
    assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "hidden must be divisible by n_heads"
    print(f"[ok] per-KV-head config: {cfg.lm_n_heads} heads / {cfg.lm_n_kv_heads} kv "
          f"(head_dim {cfg.lm_hidden_dim // cfg.lm_n_heads})")


if __name__ == '__main__':
    test_gate_half_at_init()
    test_kv_head_config()
    print("ALL INVARIANTS PASS")
