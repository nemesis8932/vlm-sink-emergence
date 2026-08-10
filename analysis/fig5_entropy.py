#!/usr/bin/env python3
"""Entropy-collapse comparison (per_key_attention.npz, dense checkpoints).

For each arm/checkpoint we take the query-averaged attention-per-key marginal (30,9,128),
normalize each head to a distribution over key positions, and compute normalized Shannon
entropy H/log(T_valid) averaged over heads. Low H = attention mass collapsed onto few keys.

Frame: in text LMs, entropy collapse and the sink co-emerge ~step 1k (Gu 2410.10781;
2510.06477). In the VLM the arms SEPARATE: textinit collapses hard (inherited),
sigmoid collapses moderately WITHOUT the norm pathology, while baseline/g1gate/RF
barely move — entropy collapse is not locked to the value/norm signatures.
NOTE: marginal (query-averaged) entropy, not per-query-row entropy (only the marginal
was dumped); the causal tilt is identical across arms so cross-arm comparison holds.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import fig_common as fc

NPZ = 'analysis/per_key_attention.npz'
ARM_KEY = {'baseline': 'baseline_s0', 'g1gate': 'g1gate_s0', 'sigmoid': 'sigmoid_s0',
           'textinit': 'textinit_s0', 'rf': 'rf'}
FLOOR_M = 0.02               # log-x floor for the step-0 probe


def norm_entropy(marg):
    """marg (...,T) -> normalized Shannon entropy over last axis, row-normalized first."""
    p = marg / np.clip(marg.sum(-1, keepdims=True), 1e-12, None)
    T = p.shape[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(-1)
    return h / np.log(T)


def tokens_at(arm):
    """step -> tokens_seen (M), interpolated over the probe log. A fixed tokens/step constant
    is wrong: padding makes the true rate ~9.5K/step, not the 16,384 maximum, so a constant put
    the curves ~1.7x too far right. Probes fire every 100 steps and the reprobe dumps sit at
    other steps, hence the interpolation over an essentially linear accumulator."""
    s = fc.load_summ(arm)
    xs = np.array([r['step'] for r in s], float)
    ys = np.array([r['tokens_seen'] for r in s], float) / 1e6
    return lambda st: float(np.interp(st, xs, ys))


def main():
    d = np.load(NPZ)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    rows = []
    for arm in fc.MAIN_ARMS + ['rf']:
        pref = ARM_KEY[arm]
        steps = sorted(int(re.search(r'step(\d+)', k).group(1))
                       for k in d.files if k.startswith(pref + '_step'))
        tok = tokens_at(arm)
        xs, ys = [], []
        for st in steps:
            a = d[f'{pref}_step{st}']             # (30,9,128)
            H = norm_entropy(a).mean()            # mean over layers+heads
            xs.append(max(tok(st), FLOOR_M)); ys.append(float(H))
        c = fc.ARM_COLORS[arm]
        ax.plot(xs, ys, 'o-', color=c, lw=1.7, ms=4.5, label=fc.ARM_LABELS[arm])
        rows.append((arm, ys[0], ys[-1]))
    ax.set_xscale('log')
    ax.set_xlabel('tokens (millions, log)', fontsize=9)
    ax.set_ylabel('normalized attention entropy  H / log T  (mean over heads)', fontsize=9)
    ax.set_title('Entropy collapse separates by arm — not locked to the norm signatures',
                 fontsize=10.5, weight='bold')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.92)
    ax.axvspan(0.5, 2.0, color='gray', alpha=0.07)
    y0, y1 = ax.get_ylim()
    ax.text(1.0, y0 + 0.035 * (y1 - y0), 'text-LM sink\n+collapse ~step 1k', fontsize=6.8,
            color='gray', ha='center', va='bottom')
    fc.style_ax(ax)
    fig.tight_layout()
    out = 'analysis/fig5_entropy.svg'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=160, bbox_inches='tight')
    print('wrote', out)
    print('\narm        H_init  H_final')
    for a, h0, h1 in rows:
        print(f'  {a:9} {h0:.3f}  {h1:.3f}  (Δ {h1-h0:+.3f})')


if __name__ == '__main__':
    main()
