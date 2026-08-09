#!/usr/bin/env python3
"""Re-dump the sigmoid arm's TRUE top sink head (L7H3) for the Figure-3 sink stripe.

The original attn_matrix.npz selected sigmoid heads by RAW (unnormalized) gate score and
picked L8H3/L1H5, missing L7H3 — the arm's actual top head under the row-normalized view
the figure uses. This script re-walks the sigmoid seed-0 checkpoints on mps/cpu (no GPU
box needed) and appends the L7H3 panels, plus raw-vs-normalized diagnostics for the paper.

Usage: python3 analysis/dump_sigmoid_l7h3.py
Writes: analysis/attn_matrix.npz (updated in place, existing keys preserved)
        analysis/sigmoid_raw_vs_norm.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import reprobe as R
import models.config as config

ARM, RUN_DIR, SEED_TAG = 'sigmoid', 'runs/sigmoid', 'sigmoid_s0'
STEPS = [250, 10664]
LAYER, HEAD = 7, 3          # 0-indexed, i.e. "L7H3" as named in the paper
NPZ = 'analysis/attn_matrix.npz'


def main():
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[dump] device={device}', flush=True)
    vlm_cfg = config.VLMConfig()
    probe = R.build_probe(vlm_cfg, 'vqav2,cocoqa,aokvqa,vsr', 32)
    model = R.build_model(ARM, vlm_cfg, 'pretrained').to(device)

    existing = dict(np.load(NPZ))
    diag = {}
    for s in STEPS:
        path = f'{RUN_DIR}/ckpt_step{s}.pt'
        sd = {k: v.float() for k, v in torch.load(path, map_location='cpu').items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert not miss and not unexp, f'{path}: missing={miss} unexpected={unexp}'

        r = R.reprobe(model, probe['image'], probe['input_ids'],
                      probe['attention_mask'], save_matrix=True)
        attn_full = r['attn_full']                      # (L,H,T,T) batch-mean, RAW sigmoid
        M = attn_full[LAYER, HEAD]                      # (T,T)

        # row-normalized pos0 share (what the figure plots) vs raw gate mass (Gu's object)
        rown = M / np.clip(M.sum(1, keepdims=True), 1e-9, None)
        norm_share = float(rown[:, 0].mean())
        raw_share = float(M[:, 0].mean())
        raw_rowsum = float(M.sum(1).mean())
        diag[f'step{s}'] = dict(layer=LAYER, head=HEAD, norm_pos0=round(norm_share, 4),
                                raw_pos0=round(raw_share, 4), raw_rowsum=round(raw_rowsum, 4))
        print(f'[dump] step {s}: L{LAYER}H{HEAD} norm_pos0={norm_share:.3f} '
              f'raw_pos0={raw_share:.3f} raw_rowsum={raw_rowsum:.3f}', flush=True)

        # find the arm's top head under each view, to document the selection bug
        rn_all = attn_full / np.clip(attn_full.sum(-1, keepdims=True), 1e-9, None)
        norm_by_head = rn_all[..., 0].mean(-1)          # (L,H)
        raw_by_head = attn_full[..., 0].mean(-1)        # (L,H)
        tn = np.unravel_index(int(norm_by_head.argmax()), norm_by_head.shape)
        tr = np.unravel_index(int(raw_by_head.argmax()), raw_by_head.shape)
        diag[f'step{s}']['top_head_normalized'] = [int(tn[0]), int(tn[1]),
                                                   round(float(norm_by_head[tn]), 4)]
        diag[f'step{s}']['top_head_raw'] = [int(tr[0]), int(tr[1]),
                                            round(float(raw_by_head[tr]), 4)]
        print(f'         top head (normalized) = L{tn[0]}H{tn[1]} @ {norm_by_head[tn]:.3f}; '
              f'(raw) = L{tr[0]}H{tr[1]} @ {raw_by_head[tr]:.3f}', flush=True)

        existing[f'{SEED_TAG}_l7h3_step{s}'] = M

    existing[f'{SEED_TAG}_l7h3_heads'] = np.array([[LAYER, HEAD]])
    np.savez_compressed(NPZ, **existing)
    json.dump(diag, open('analysis/sigmoid_raw_vs_norm.json', 'w'), indent=1)
    print(f'[dump] wrote {NPZ} + analysis/sigmoid_raw_vs_norm.json', flush=True)


if __name__ == '__main__':
    main()
