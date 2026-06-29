#!/usr/bin/env python3
"""Aggregate reprobe npz -> figure deliverables (entropy curve + sink-stripe).
  per_key_attention.npz : raw_profile (L,H,T) per ckpt, ALL arms/steps  -> entropy-over-training (fig5)
  attn_matrix.npz       : top-2 sink-head (by attn->pos0 at final) T*T batch-mean attn, few ckpts (fig6)
  dump_manifest.json    : exactly what was dumped (git, for Auditor coverage)
raw_profile == sink_probe.py mean_attn_per_key. attn_full present only in --save_matrix npz.
"""
import glob, os, re, json
import numpy as np

# per_key coverage: 4 arms seed-0 + RF, all steps
PER_KEY = {'baseline_s0': 'runs/baseline', 'g1gate_s0': 'runs/g1gate',
           'sigmoid_s0': 'runs/sigmoid', 'textinit_s0': 'runs/textinit',
           'rf': 'runs/rf_fresh_baseline'}
# attn_matrix coverage: 4 arms seed-0, final step (top-2 heads chosen here)
FINAL = {'baseline_s0': ('runs/baseline', 18287), 'g1gate_s0': ('runs/g1gate', 10786),
         'sigmoid_s0': ('runs/sigmoid', 10664), 'textinit_s0': ('runs/textinit', 6244)}

def steps(run_dir):
    return sorted(int(re.search(r'step(\d+)', f).group(1))
                  for f in glob.glob(os.path.join(run_dir, 'reprobe', 'reprobe_step*.npz')))

def npz(run_dir, s):
    return np.load(os.path.join(run_dir, 'reprobe', f'reprobe_step{s}.npz'))

def top2_heads(run_dir, s):
    r2p = npz(run_dir, s)['raw_to_pos0']                       # (L,H)
    idx = np.dstack(np.unravel_index(np.argsort(r2p.ravel())[::-1], r2p.shape))[0][:2]
    return idx                                                 # [[L,H],[L,H]]

def main():
    pk, am, manifest = {}, {}, {'per_key_attention': {}, 'attn_matrix': {}}
    # --- per_key: raw_profile every step ---
    for tag, rd in PER_KEY.items():
        ss = steps(rd)
        if not ss: print(f'[skip per_key] {rd} no npz'); continue
        for s in ss:
            pk[f'{tag}_step{s}'] = npz(rd, s)['raw_profile'].astype(np.float32)
        manifest['per_key_attention'][tag] = ss
        print(f'per_key {tag:12s} steps={ss}')
    # --- attn_matrix: top-2 sink heads, all ckpts that have attn_full ---
    for tag, (rd, fs) in FINAL.items():
        heads = top2_heads(rd, fs)
        am[f'{tag}_heads'] = heads
        dumped = []
        for s in steps(rd):
            d = npz(rd, s)
            if 'attn_full' not in d.files: continue
            mats = np.stack([d['attn_full'][l, h] for l, h in heads]).astype(np.float32)  # (2,T,T)
            am[f'{tag}_step{s}'] = mats
            dumped.append(int(s))
        manifest['attn_matrix'][tag] = {'heads_LH': heads.tolist(), 'steps': dumped}
        print(f'attn_matrix {tag:12s} heads={heads.tolist()} steps={dumped}')
    np.savez_compressed('analysis/per_key_attention.npz', **pk)
    np.savez_compressed('analysis/attn_matrix.npz', **am)
    json.dump(manifest, open('analysis/dump_manifest.json', 'w'), indent=2)
    print(f'\nwrote per_key_attention.npz ({len(pk)} arrays), attn_matrix.npz ({len([k for k in am if "step" in k])} mats), dump_manifest.json')

if __name__ == '__main__':
    main()
