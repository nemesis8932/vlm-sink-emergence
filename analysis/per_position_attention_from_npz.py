#!/usr/bin/env python3
"""Per-position attention-mass profile from reprobe npz (closes open-Q #1, attention half).

DROP-IN: run once `runs/*/reprobe/reprobe_step*.npz` are synced from HF. Zero-GPU.
The npz `raw_profile` (L,H,T) is the ONLY full per-position array reprobe dumps; v/h NORMS
are stored pos0-vs-rest only, so this script answers the *concentration-location* question
(is the attention sink at pos0?) definitively. The per-position NORM profiles need a
reprobe.py patch + re-run (see handoff).

Question answered: for each arm/seed, where does attention mass actually concentrate, and is
pos0 the max-mass position? A "sink" hidden at pos1/pos5/... that pos0-anchored Sink^eps would
miss shows up here.

Output: analysis/per_position_attention.svg + a printed per-arm verdict.
"""
import glob, os, re, json
import numpy as np

RUNS = {
    # seed-0 (session-3): npz from HF, generated on cloud box
    'baseline':       ['runs/baseline'],
    'g1gate':         ['runs/g1gate'],
    'sigmoid':        ['runs/sigmoid'],
    'textinit':       ['runs/textinit'],
    # seed-1/seed-2/RF: npz regenerated locally via streaming probe batch (see reprobe.py note)
    'baseline_seed1': ['runs/baseline_seed1'],
    'g1gate_seed1':   ['runs/g1gate_seed1'],
    'g1gate_seed2':   ['runs/g1gate_seed2'],
    'sigmoid_seed1':  ['runs/sigmoid_seed1'],
    'textinit_seed1': ['runs/textinit_seed1'],
    'textinit_seed2': ['runs/textinit_seed2'],
    'rf_fresh_baseline': ['runs/rf_fresh_baseline'],
}

def latest_npz(run_dir):
    fs = glob.glob(os.path.join(run_dir, 'reprobe', 'reprobe_step*.npz'))
    if not fs:
        return None
    return max(fs, key=lambda f: int(re.search(r'step(\d+)', f).group(1)))

def profile(npz_path, topk=20):
    d = np.load(npz_path)
    raw = d['raw_profile']                  # (L, H, T) raw attention mass per position
    mean_pos = raw.mean(axis=(0, 1))        # (T,) mean over layers+heads
    maxhead_pos = raw.max(axis=(0, 1))      # (T,) max over layers+heads (sink-revealing)
    return mean_pos[:topk], maxhead_pos[:topk]

def main():
    rows = []
    profiles = {}
    for arm, dirs in RUNS.items():
        for rd in dirs:
            p = latest_npz(rd)
            if p is None:
                print(f'[missing npz] {rd}  -> sync from HF first'); continue
            mean_pos, maxhead_pos = profile(p)
            argmax_mean = int(mean_pos.argmax())
            frac_pos0 = float(mean_pos[0] / mean_pos.sum())
            frac_argmax = float(mean_pos[argmax_mean] / mean_pos.sum())
            verdict = 'pos0 IS max' if argmax_mean == 0 else f'MAX AT pos{argmax_mean} (pos0 mislocated)'
            seed = 's' + rd.split('_seed')[-1] if '_seed' in rd else 's0'
            rows.append((arm, seed, argmax_mean, frac_pos0, frac_argmax, verdict,
                         [round(float(x), 5) for x in mean_pos],
                         [round(float(x), 5) for x in maxhead_pos]))
            profiles[f'{arm}/{seed}'] = mean_pos
            print(f'{arm:10s} {seed:6s} argmax_mass=pos{argmax_mean:<3d} '
                  f'mass@pos0={frac_pos0:.3f} mass@max={frac_argmax:.3f}  {verdict}')
    if not profiles:
        print('\nNo npz found. Fetch first (see commands printed by the audit handoff).'); return
    json.dump([{'arm':a,'seed':s,'argmax_mass_pos':am,'mass_pos0':fp,'mass_max':fm,'verdict':v,
                'mean_mass_by_pos':prof,'maxhead_mass_by_pos':mx}
               for a,s,am,fp,fm,v,prof,mx in rows],
              open('analysis/per_position_attention.json','w'), indent=2)
    print('\nwrote analysis/per_position_attention.json — Auditor will read it back + build the figure')

if __name__ == '__main__':
    main()
