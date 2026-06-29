#!/usr/bin/env python3
"""Shared infra for the publication figure suite (Auditor, arxiv v1).

Palette reuses analyze_sinks.py arm colors; RF added. All loaders are pure-local
(probes.jsonl per-(layer,head) + the HF reprobe npz). RF is stitched across the
resume@4000 seam (pre_resume4000 segment + main) so its path starts at the origin.
"""
import json, glob, os, re
import numpy as np

# repo root = parent of this file's analysis/ dir; all run paths anchored here
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Palette — matches analyze_sinks.py; RF = purple (fresh-data baseline twin)
ARM_COLORS = {
    'baseline': 'tab:blue', 'textinit': 'tab:orange',
    'g1gate': 'tab:green', 'sigmoid': 'tab:red', 'rf': 'tab:purple',
}
ARM_LABELS = {
    'baseline': 'baseline (softmax, scratch)',
    'textinit': 'textinit (pretrained text LM)',
    'g1gate': 'g1gate (softmax + σ-gate)',
    'sigmoid': 'sigmoid (no softmax)',
    'rf': 'RF (fresh data, 1B tok)',
}
RUN_DIR = {
    'baseline': 'runs/baseline', 'textinit': 'runs/textinit',
    'g1gate': 'runs/g1gate', 'sigmoid': 'runs/sigmoid',
    'rf': 'runs/rf_fresh_baseline',
}
# main-body arm order (4 levers) + RF
MAIN_ARMS = ['baseline', 'g1gate', 'sigmoid', 'textinit']
ALL_ARMS = MAIN_ARMS + ['rf']


def load_summ(arm):
    """step-sorted list of dicts {step, tokens_seen, **summary}. RF stitched over seam."""
    rd = RUN_DIR[arm]
    files = [f'{rd}/probes.jsonl']
    if arm == 'rf':
        pre = f'{rd}/probes.jsonl.pre_resume4000'
        if os.path.exists(pre):
            files = [pre, f'{rd}/probes.jsonl']
    rows = {}
    for fp in files:
        with open(fp) as f:
            for line in f:
                p = json.loads(line)
                s = dict(p['summary']); s['step'] = p['step']; s['tokens_seen'] = p.get('tokens_seen', 0)
                rows[p['step']] = s            # main overrides pre at the duplicated seam step
    return [rows[k] for k in sorted(rows)]


def load_perhead(arm, which='last'):
    """Per-(layer,head) arrays at a probe step. which='last'|'first'|int(step).
    Returns dict of (30,9) arrays: attn_to_pos0, v_ratio, h_ratio(by layer, broadcast), argmax_key."""
    rd = RUN_DIR[arm]
    P = []
    with open(f'{rd}/probes.jsonl') as f:
        for line in f:
            P.append(json.loads(line))
    P.sort(key=lambda p: p['step'])
    if which == 'last':
        p = P[-1]
    elif which == 'first':
        p = P[0]
    else:
        p = min(P, key=lambda q: abs(q['step'] - int(which)))
    L = p['layers']
    nL, nH = len(L), len(L[0]['attn_to_pos0'])
    # NORMALIZED attn = cross-arm-comparable concentration (sigmoid raw = uniform ~0.5 gate, not a sink)
    a0 = np.array([L[i]['attn_to_pos0_norm'] for i in range(nL)])
    vp = np.array([L[i]['v_norm_pos0'] for i in range(nL)])
    vr = np.array([L[i]['v_norm_rest'] for i in range(nL)])
    hp = np.array([L[i]['h_norm_pos0'] for i in range(nL)])     # (nL,) layer-level
    hr = np.array([L[i]['h_norm_rest'] for i in range(nL)])
    amk = np.array([L[i]['argmax_key'] for i in range(nL)])
    return dict(step=p['step'], attn_to_pos0=a0, v_ratio=vp / vr,
                h_ratio=(hp / hr)[:, None].repeat(nH, axis=1), argmax_key=amk,
                v_pos0=vp, v_rest=vr)


def latest_npz(run_dir):
    fs = glob.glob(os.path.join(run_dir, 'reprobe', 'reprobe_step*.npz'))
    return max(fs, key=lambda f: int(re.search(r'step(\d+)', f).group(1))) if fs else None


def load_perhead_npz(arm):
    """TRUE per-(layer,head) arrays from the reprobe npz at the arm's latest ckpt.
    probes.jsonl logs v/h-norm per LAYER only; the npz has them per head.
    Returns (30,9) attn_to_pos0 + v_ratio, plus step. value 'rest' = image tokens."""
    p = latest_npz(RUN_DIR[arm])
    if p is None:
        raise FileNotFoundError(f'no reprobe npz for {arm}')
    d = np.load(p)
    step = int(re.search(r'step(\d+)', p).group(1))
    a0 = d['norm_to_pos0']                        # (30,9) row-normalized concentration (sigmoid-safe)
    vr = d['vn_pos0'] / d['vn_rest']             # (30,9) per-head value-norm ratio
    return dict(step=step, attn_to_pos0=a0, v_ratio=vr)


def style_ax(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)


if __name__ == '__main__':
    for a in ALL_ARMS:
        S = load_summ(a)
        print(f'{a:9} {len(S)} probes, step {S[0]["step"]}→{S[-1]["step"]}, '
              f'final max_attn_pos0={S[-1]["max_attn_pos0"]:.3f} v_ratio={S[-1]["v_ratio_pos0"]:.3f}')
