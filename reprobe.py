"""Run-3 re-probing of SAVED checkpoints (no training). Emits the per-(layer,head) and
per-position arrays that the live summary probe collapses, so the dissociation claims
can be checked un-pooled. Three products per checkpoint, dumped to runs/<name>/reprobe/:

  3a  per-(layer,head) value-norm at pos0 vs rest  -> per-head v_ratio (never pooled)
      plus per-layer residual-stream (h) norm pos0 vs rest (h is per-token, not per-head)
  3c  full per-position RAW attention/sigmoid mass profile, mean over heads and per head
      (for the sigmoid arm this raw, un-row-normalized mass is the PRIMARY concentration
      evidence; row-normalized Sink^eps is secondary)
  3b  image-swap invariance: fix the text, feed N images, dump pos0 value-vectors and
      attention-to-pos0 per image so variance across images can be measured. pos0 is a
      content-bearing image token, so it SHOULD vary; near-zero variance is a red flag.

Usage:
  python reprobe.py --arm baseline --run_dir runs/baseline --ckpts 0,1000,4000,18287
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch

import models.config as config
from models.language_model import apply_rotary_pos_embd
from models.vision_language_model import VisionLanguageModel
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from data.collators import VQACollator
from data.datasets import VQADataset, IterableVQADataset
from data.processors import get_image_processor, get_tokenizer
from datasets import load_dataset, interleave_datasets

ARMS = {
    'baseline': dict(gate=False, attn='softmax', lm_init='random'),
    'g1gate':   dict(gate=True,  attn='softmax', lm_init='random'),
    'sigmoid':  dict(gate=False, attn='sigmoid', lm_init='random'),
    'textinit': dict(gate=False, attn='softmax', lm_init='pretrained'),
}


def build_probe(vlm_cfg, subsets, probe_n):
    """Streaming variant: interleaves subsets, takes first probe_n samples — no shard download.
    NOTE: probe batch differs from the cloud seed-0 run (which used full non-streaming shuffle);
    within this local re-run all arms/seeds use the same streaming batch, preserving comparability."""
    from itertools import islice
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    parts = [load_dataset('HuggingFaceM4/the_cauldron', name, streaming=True)['train']
             for name in subsets.split(',')]
    ds_stream = interleave_datasets(parts, seed=0)
    iterable_ds = IterableVQADataset(ds_stream, tokenizer, image_processor)
    collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    random.seed(0)
    probe_items = list(islice(iterable_ds, probe_n))
    return collator(probe_items)


def build_model(arm_name, vlm_cfg, vit_init):
    arm = ARMS[arm_name]
    vlm_cfg.lm_attn_gate = arm['gate']
    vlm_cfg.lm_attn_impl = arm['attn']
    model = VisionLanguageModel(vlm_cfg, load_backbone=False)
    if vit_init == 'pretrained':
        model.vision_encoder = ViT.from_pretrained(vlm_cfg)
    if arm['lm_init'] == 'pretrained':
        model.decoder = LanguageModel.from_pretrained(vlm_cfg)
    return model


@torch.no_grad()
def reprobe(model, images, input_ids, attention_mask):
    """Re-walk the decoder (eager fp32) dumping un-pooled arrays. Mirrors sink_probe math."""
    model.eval()
    device = next(model.parameters()).device
    images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

    image_embd = model.MP(model.vision_encoder(images))
    token_embd = model.decoder.token_embedding(input_ids)
    x = torch.cat((image_embd, token_embd), dim=1)
    B, T, C = x.size()
    n_img = image_embd.size(1)
    img_mask = torch.ones((B, n_img), device=device, dtype=attention_mask.dtype)
    full_mask = torch.cat((img_mask, attention_mask), dim=1)

    qvalid = full_mask.bool().clone()
    qvalid[:, 0] = False
    n_qvalid = qvalid.sum().clamp_min(1).float()

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    cos, sin = model.decoder.rotary_embd(position_ids)
    mask4d = full_mask.unsqueeze(1).unsqueeze(2)
    float_mask = (1.0 - mask4d) * torch.finfo(x.dtype).min
    padding_mask = (mask4d == 0).transpose(-1, -2)
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).view(1, 1, T, T)

    L = len(model.decoder.blocks)
    H = model.decoder.blocks[0].attn.n_heads
    hd = model.decoder.blocks[0].attn.head_dim

    vn_pos0 = np.zeros((L, H));   vn_rest = np.zeros((L, H))          # 3a per-head v-norm
    hn_pos0 = np.zeros(L);        hn_rest = np.zeros(L)               # per-layer residual norm
    vn_profile = np.zeros((L, H, T))                                  # FULL per-position v-norm (mean over batch)
    hn_profile = np.zeros((L, T))                                     # FULL per-position residual norm (mean over batch)
    raw_profile = np.zeros((L, H, T))                                 # 3c per-head raw mass profile
    raw_to_pos0 = np.zeros((L, H)); norm_to_pos0 = np.zeros((L, H))
    # 3b: per-image pos0 value vector and attention-to-pos0
    v_pos0_img = np.zeros((L, B, H, hd))
    attn0_img = np.zeros((L, B, H))

    qv = qvalid.unsqueeze(1)  # (B,1,T)
    for li, block in enumerate(model.decoder.blocks):
        attn_mod = block.attn
        res_norm = x.norm(dim=-1)
        hn_pos0[li] = res_norm[:, 0].mean().item()
        hn_rest[li] = res_norm[:, 1:][full_mask[:, 1:].bool()].mean().item()
        hn_profile[li] = res_norm.mean(dim=0).cpu().numpy()              # (T,) per-position residual norm

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
            attn_norm = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            attn = torch.softmax(scores, dim=-1)
            attn_norm = attn

        # 3c: mean over valid queries -> (H, T) raw mass profile, and pos0 columns
        prof = (attn * qv.unsqueeze(-1)).sum(dim=(0, 2)) / n_qvalid       # (H,T) raw
        raw_profile[li] = prof.cpu().numpy()
        raw_to_pos0[li] = (attn[..., 0] * qv).sum(dim=(0, 2)).div(n_qvalid).cpu().numpy()
        norm_to_pos0[li] = (attn_norm[..., 0] * qv).sum(dim=(0, 2)).div(n_qvalid).cpu().numpy()

        # 3a: per-head value-norm at pos0 vs valid rest
        v_norm = v.norm(dim=-1)                                          # (B,H,T)
        vn_profile[li] = v_norm.mean(dim=0).cpu().numpy()                # (H,T) per-position v-norm
        vn_pos0[li] = v_norm[:, :, 0].mean(dim=0).cpu().numpy()          # (H,)
        kvalid = full_mask[:, 1:].bool().unsqueeze(1).expand(B, H, T - 1)
        rest = v_norm[:, :, 1:].masked_fill(~kvalid, float('nan'))
        vn_rest[li] = torch.nanmean(rest, dim=(0, 2)).cpu().numpy()      # (H,)

        # 3b: per-image pos0 value vector + attention to pos0 (per query-image)
        v_pos0_img[li] = v[:, :, 0, :].cpu().numpy()                     # (B,H,hd)
        attn0_img[li] = (attn[..., 0] * qv).sum(dim=2).div(qvalid.sum(dim=1, keepdim=True).clamp_min(1)).cpu().numpy()

        # finish block exactly as module does
        y = attn @ v
        y = y.masked_fill(padding_mask, 0.0)
        if attn_mod.use_gate:
            gate = torch.sigmoid(attn_mod.gate_proj(xn)).view(B, T, H, hd).transpose(1, 2)
            y = y * gate
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = attn_mod.out_proj(y)
        x = x + y
        x = x + block.mlp(block.norm2(x))

    model.train()
    return dict(n_img=n_img, T=T, vn_pos0=vn_pos0, vn_rest=vn_rest, hn_pos0=hn_pos0, hn_rest=hn_rest,
                vn_profile=vn_profile, hn_profile=hn_profile,
                raw_profile=raw_profile, raw_to_pos0=raw_to_pos0, norm_to_pos0=norm_to_pos0,
                v_pos0_img=v_pos0_img, attn0_img=attn0_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(ARMS), required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--ckpts', required=True, help='comma-sep step numbers')
    ap.add_argument('--vit_init', default='pretrained')
    ap.add_argument('--probe_n', type=int, default=32)
    ap.add_argument('--swap_n', type=int, default=12, help='# images for the image-swap test')
    ap.add_argument('--subsets', default='vqav2,cocoqa,aokvqa,vsr')
    args = ap.parse_args()

    out = os.path.join(args.run_dir, 'reprobe')
    os.makedirs(out, exist_ok=True)
    vlm_cfg = config.VLMConfig()
    probe = build_probe(vlm_cfg, args.subsets, args.probe_n)
    device = torch.device('cuda' if torch.cuda.is_available()
                           else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # image-swap batch: sample-0 text repeated against swap_n distinct images
    N = min(args.swap_n, probe['image'].size(0))
    swap_img = probe['image'][:N]
    swap_ids = probe['input_ids'][0:1].repeat(N, 1)
    swap_mask = probe['attention_mask'][0:1].repeat(N, 1)

    model = build_model(args.arm, vlm_cfg, args.vit_init).to(device)

    summary = []
    for s in args.ckpts.split(','):
        s = s.strip()
        path = os.path.join(args.run_dir, f'ckpt_step{s}.pt')
        if not os.path.exists(path):
            print(f"[skip] {path} missing", flush=True); continue
        sd = {k: v.float() for k, v in torch.load(path, map_location='cpu').items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert not miss and not unexp, f"{path}: missing={miss} unexpected={unexp}"

        main_r = reprobe(model, probe['image'], probe['input_ids'], probe['attention_mask'])
        swap_r = reprobe(model, swap_img, swap_ids, swap_mask)

        np.savez_compressed(os.path.join(out, f'reprobe_step{s}.npz'),
                            **{k: v for k, v in main_r.items()},
                            swap_v_pos0=swap_r['v_pos0_img'], swap_attn0=swap_r['attn0_img'])

        # compact per-head v_ratio + image-swap variance for the jsonl summary
        v_ratio = main_r['vn_pos0'] / np.clip(main_r['vn_rest'], 1e-8, None)   # (L,H)
        # variance of pos0 value-vector across images: mean over (L,H) of per-component std / mean|.|
        sv = swap_r['v_pos0_img']                                             # (L,B,H,hd)
        v_img_cv = float(np.nanmean(sv.std(axis=1) / (np.abs(sv).mean(axis=1) + 1e-8)))
        attn0_img_std = float(swap_r['attn0_img'].std(axis=1).mean())
        rec = dict(step=int(s),
                   v_ratio_perhead_mean=float(np.nanmean(v_ratio)),
                   v_ratio_perhead_min=float(np.nanmin(v_ratio)),
                   v_ratio_perhead_max=float(np.nanmax(v_ratio)),
                   raw_to_pos0_max=float(main_r['raw_to_pos0'].max()),
                   raw_to_pos0_mean=float(main_r['raw_to_pos0'].mean()),
                   swap_v_pos0_cv=v_img_cv, swap_attn0_std=attn0_img_std)
        summary.append(rec)
        print(f"[reprobe {args.arm}] step {s} | v_ratio[min/mean/max] "
              f"{rec['v_ratio_perhead_min']:.3f}/{rec['v_ratio_perhead_mean']:.3f}/{rec['v_ratio_perhead_max']:.3f}"
              f" | raw_to_pos0 max {rec['raw_to_pos0_max']:.3f} | swap_v_cv {v_img_cv:.3f}"
              f" swap_attn0_std {attn0_img_std:.4f}", flush=True)

    with open(os.path.join(out, 'reprobe_summary.jsonl'), 'w') as f:
        for r in summary:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
