#!/usr/bin/env python3
"""Per-position NORM profiles from patched-reprobe npz (closes open-Q #1, the NORM half).

Companion to per_position_attention_from_npz.py (which did the ATTENTION half). reprobe.py now
also dumps FULL per-position arrays `vn_profile` (L,H,T) value-norm and `hn_profile` (L,T)
residual/hidden norm (mean over the probe batch). This script reads them and answers:

  - massive-activation: does residual norm ‖h‖ spike at pos0? at which position does it peak?
  - value-drain: is the value-norm ‖v‖ at pos0 LOW vs rest (<1 ratio = drain)?
  - migration: does the norm peak sit at the SAME position as the attention max-mass (raw_profile)?
    Anchoring h_ratio/v_ratio at the ACTUAL max-mass position vs pos0 resolves whether textinit's
    pos0-anchored h_ratio (5.5 vs 42.5 across seeds) is a lower bound from anchor-mislocation.

"rest" baseline = image-token positions [1, n_img) — all valid (no text padding), apples-to-apples.
Output: analysis/per_position_norms.json (+ npz bundle of the profiles for the figure). Zero-GPU.
"""
import glob, os, re, json
import numpy as np

RUNS = {  # arm -> (run_dir, reprobe arm) ; latest npz per run is used
    'textinit/s0': 'runs/textinit',
    'textinit/s1': 'runs/textinit_seed1',
    'textinit/s2': 'runs/textinit_seed2',
    'rf':          'runs/rf_fresh_baseline',
    'baseline/s0': 'runs/baseline',
    'sigmoid/s0':  'runs/sigmoid',
    'g1gate/s0':   'runs/g1gate',
}


def latest_npz(run_dir):
    fs = glob.glob(os.path.join(run_dir, 'reprobe', 'reprobe_step*.npz'))
    return max(fs, key=lambda f: int(re.search(r'step(\d+)', f).group(1))) if fs else None


def analyze(npz_path):
    d = np.load(npz_path)
    n_img = int(d['n_img'])
    raw = d['raw_profile']                       # (L,H,T) attention mass
    vn = d['vn_profile']                          # (L,H,T) value-norm
    hn = d['hn_profile']                          # (L,T)   residual norm
    mass = raw.mean(axis=(0, 1))                  # (T,)
    vpos = vn.mean(axis=(0, 1))                   # (T,) mean value-norm per position
    hpos = hn.mean(axis=0)                        # (T,) mean residual norm per position
    rest = slice(1, n_img)                        # image-token rest, all valid

    p_attn = int(mass[:20].argmax())             # attention max-mass position
    p_hn = int(hpos[:20].argmax())               # residual-norm peak position
    p_vn_lo = int(vpos[:20].argmin())            # value-norm trough position (drain)
    h_rest = float(np.median(hpos[rest]))
    v_rest = float(np.median(vpos[rest]))
    return dict(
        step=int(re.search(r'step(\d+)', npz_path).group(1)), n_img=n_img,
        argmax_mass_pos=p_attn, argmax_hnorm_pos=p_hn, argmin_vnorm_pos=p_vn_lo,
        h_ratio_pos0=round(float(hpos[0]) / h_rest, 3),
        h_ratio_atmax=round(float(hpos[p_attn]) / h_rest, 3),
        v_ratio_pos0=round(float(vpos[0]) / v_rest, 3),
        v_ratio_atmax=round(float(vpos[p_attn]) / v_rest, 3),
        massive_act_pos0=bool(hpos[0] / h_rest > 2.0),
        value_drain_pos0=bool(vpos[0] / v_rest < 0.8),
        norm_migrates_with_attn=bool(p_hn == p_attn and p_attn != 0),
        hpos20=[round(float(x), 1) for x in hpos[:20]],
        vpos20=[round(float(x), 3) for x in vpos[:20]],
        mass20=[round(float(x), 5) for x in mass[:20]],
    )


def main():
    rows, bundle = {}, {}
    for tag, rd in RUNS.items():
        p = latest_npz(rd)
        if p is None:
            print(f'[missing npz] {rd} -> reprobe first'); continue
        r = analyze(p); rows[tag] = r
        bundle[f'{tag}/hpos'] = np.array(r['hpos20']); bundle[f'{tag}/vpos'] = np.array(r['vpos20'])
        bundle[f'{tag}/mass'] = np.array(r['mass20'])
        ma = 'Y' if r['massive_act_pos0'] else 'n'
        vd = 'Y' if r['value_drain_pos0'] else 'n'
        mig = 'Y' if r['norm_migrates_with_attn'] else 'n'
        print(f"{tag:14s} step{r['step']:<6d} mass@pos{r['argmax_mass_pos']} hnorm-peak@pos{r['argmax_hnorm_pos']} | "
              f"h_ratio pos0={r['h_ratio_pos0']:.2f} atmax={r['h_ratio_atmax']:.2f} | "
              f"v_ratio pos0={r['v_ratio_pos0']:.2f} atmax={r['v_ratio_atmax']:.2f} | "
              f"massive-act@pos0={ma} value-drain@pos0={vd} migrates={mig}")
    if not rows:
        print('no npz; reprobe first'); return
    json.dump(rows, open('analysis/per_position_norms.json', 'w'), indent=2)
    np.savez_compressed('analysis/per_position_norms.npz', **bundle)
    print('\nwrote analysis/per_position_norms.json + .npz')


if __name__ == '__main__':
    main()
