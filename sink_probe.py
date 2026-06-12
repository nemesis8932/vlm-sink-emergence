"""Attention-sink probe for nanoVLM.

Re-walks the decoder manually (eager, fp32, no_grad) using the module weights so the
training path (SDPA, autocast, optional torch.compile) is never touched. Each call
validates its own hidden states against the model's real forward, so the mirrored
math cannot silently drift from the model.

Metrics per probe call (Gu et al. 2410.10781 conventions, fixed T):
  - mean attention to token 0 (first image token) per (layer, head)
  - Sink^eps_1 = fraction of (layer, head) with that mean > eps, eps in {0.2, 0.3, 0.4}
  - attention mass on image segment [0..n_img) vs text segment, per layer
  - mean attention to the first *text* token (per-sample position, left padding aware)
  - argmax key position per (layer, head) ("where the sink sits")
  - value-vector L2 norm at token 0 vs other positions, per layer (decoupling)
  - residual-stream norm at token 0 vs other positions, per layer (massive activations)
"""

import math
import torch

from models.language_model import apply_rotary_pos_embd


@torch.no_grad()
def probe_sinks(model, images, input_ids, attention_mask, validate=True):
    model.eval()
    cfg = model.cfg
    device = next(model.parameters()).device
    images = images.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    image_embd = model.MP(model.vision_encoder(images))
    token_embd = model.decoder.token_embedding(input_ids)
    x = torch.cat((image_embd, token_embd), dim=1)

    B, T, C = x.size()
    n_img = image_embd.size(1)
    img_mask = torch.ones((B, n_img), device=device, dtype=attention_mask.dtype)
    full_mask = torch.cat((img_mask, attention_mask), dim=1)  # (B, T)

    # first real text token per sample (left padding)
    first_text_pos = n_img + attention_mask.float().argmax(dim=1)  # (B,)

    # query validity: real tokens, excluding query position 0 itself
    qvalid = full_mask.bool().clone()
    qvalid[:, 0] = False
    n_qvalid = qvalid.sum()

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    cos, sin = model.decoder.rotary_embd(position_ids)

    # 4D float mask exactly as LanguageModelGroupedQueryAttention builds it
    mask4d = full_mask.unsqueeze(1).unsqueeze(2)
    float_mask = (1.0 - mask4d) * torch.finfo(x.dtype).min
    padding_mask = (mask4d == 0).transpose(-1, -2)
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).view(1, 1, T, T)

    per_layer = []
    for block in model.decoder.blocks:
        attn_mod = block.attn
        H, hd = attn_mod.n_heads, attn_mod.head_dim

        # residual-stream norms at block input
        res_norm = x.norm(dim=-1)  # (B, T)
        h_norm_pos0 = res_norm[:, 0].mean().item()
        h_norm_rest = res_norm[:, 1:][full_mask[:, 1:].bool()].mean().item()

        xn = block.norm1(x)
        q = attn_mod.q_proj(xn).view(B, T, H, hd).transpose(1, 2)
        k = attn_mod.k_proj(xn).view(B, T, attn_mod.n_kv_heads, hd).transpose(1, 2)
        v = attn_mod.v_proj(xn).view(B, T, attn_mod.n_kv_heads, hd).transpose(1, 2)
        q, k = apply_rotary_pos_embd(q, k, cos, sin)
        k = k.repeat_interleave(attn_mod.n_kv_groups, dim=1)
        v = v.repeat_interleave(attn_mod.n_kv_groups, dim=1)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(hd)
        scores = scores.masked_fill(~causal, float('-inf'))
        scores = scores + float_mask
        if attn_mod.attn_impl == 'sigmoid':
            attn = torch.sigmoid(scores)
            # row-normalized view: relative importance per key, comparable across arms
            attn_norm = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            attn = torch.softmax(scores, dim=-1)
            attn_norm = attn

        # --- metrics from attn (B, H, T, T) ---
        qv = qvalid.unsqueeze(1)                                    # (B, 1, T)
        denom = n_qvalid.float()
        attn_to_0 = (attn[..., 0] * qv).sum(dim=(0, 2)) / denom     # (H,)
        img_mass = (attn[..., :n_img].sum(-1) * qv).sum(dim=(0, 2)) / denom
        txt_mass = (attn[..., n_img:].sum(-1) * qv).sum(dim=(0, 2)) / denom
        ft = first_text_pos.view(B, 1, 1, 1).expand(B, H, T, 1)
        attn_to_ft = (attn.gather(3, ft).squeeze(-1) * qv).sum(dim=(0, 2)) / denom
        mean_attn_per_key = (attn * qv.unsqueeze(-1)).sum(dim=(0, 2)) / denom  # (H, T)
        argmax_key = mean_attn_per_key.argmax(dim=-1)               # (H,)

        v_norm = v.norm(dim=-1)                                     # (B, H, T)
        v_norm_pos0 = v_norm[:, :, 0].mean().item()
        kvalid = full_mask[:, 1:].bool().unsqueeze(1).expand(B, H, T - 1)
        v_norm_rest = v_norm[:, :, 1:][kvalid].mean().item()

        attn_to_0_norm = (attn_norm[..., 0] * qv).sum(dim=(0, 2)) / denom

        per_layer.append({
            'attn_to_pos0_norm': attn_to_0_norm.tolist(),
            'attn_to_pos0': attn_to_0.tolist(),
            'attn_to_first_text': attn_to_ft.tolist(),
            'img_mass': img_mass.mean().item(),
            'txt_mass': txt_mass.mean().item(),
            'argmax_key': argmax_key.tolist(),
            'v_norm_pos0': v_norm_pos0,
            'v_norm_rest': v_norm_rest,
            'h_norm_pos0': h_norm_pos0,
            'h_norm_rest': h_norm_rest,
        })

        # --- finish the block exactly as the module does ---
        y = attn @ v
        y = y.masked_fill(padding_mask, 0.0)
        if attn_mod.use_gate:
            gate = torch.sigmoid(attn_mod.gate_proj(xn)).view(B, T, H, hd).transpose(1, 2)
            y = y * gate
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = attn_mod.out_proj(y)
        x = x + y
        x = x + block.mlp(block.norm2(x))

    x = model.decoder.norm(x)

    if validate:
        ref = model.decoder(torch.cat((image_embd, token_embd), dim=1), full_mask)
        valid = full_mask.bool()  # pad rows legitimately differ (SDPA path doesn't zero them)
        err = (x - ref)[valid].abs().max().item()
        scale = ref[valid].abs().max().item()
        assert err / max(scale, 1e-6) < 1e-2, f"probe drifted from model forward: max err {err} vs scale {scale}"

    # summary
    L = len(per_layer)
    H = len(per_layer[0]['attn_to_pos0'])
    flat = [a for layer in per_layer for a in layer['attn_to_pos0_norm']]
    sink_frac = {f"sink_eps{e}": sum(a > e for a in flat) / (L * H) for e in (0.2, 0.3, 0.4)}
    summary = {
        **sink_frac,
        'mean_attn_pos0': sum(flat) / len(flat),
        'max_attn_pos0': max(flat),
        'mean_img_mass': sum(l['img_mass'] for l in per_layer) / L,
        'v_ratio_pos0': sum(l['v_norm_pos0'] / max(l['v_norm_rest'], 1e-8) for l in per_layer) / L,
        'h_ratio_pos0': sum(l['h_norm_pos0'] / max(l['h_norm_rest'], 1e-8) for l in per_layer) / L,
    }
    model.train()
    return {'summary': summary, 'layers': per_layer}


def build_probe_batch(dataset, n=32):
    """Fixed probe batch: first n samples of the given (val) torch dataset via its collator output."""
    items = [dataset[i] for i in range(n)]
    return items
