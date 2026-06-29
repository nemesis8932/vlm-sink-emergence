#!/usr/bin/env python3
"""HERO: decoupling phase-portrait. Each arm is a PATH through (concentration, norm)
space that diverges into its own corner over training. The thesis as motion.

Left  : x = max attn->pos0 (concentration)   y = value-norm ratio @pos0
Right : x = max attn->pos0 (concentration)   y = h-norm ratio @pos0 (massive activation, log)

A point per probe step; markers grow with training; a dot marks the start, an
arrowhead the end (direction of travel). Four levers leave from a shared origin
(no concentration, unit norms) and end in four distinct corners -> the signatures
are independently steerable, not facets of one phenomenon.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fig_common as fc

ARMS = fc.MAIN_ARMS + ['rf']


def smooth(y, k=5):
    """edge-padded moving average — preserves endpoints (no convolve boundary bias)."""
    y = np.asarray(y, float)
    if len(y) < k: return y
    pad = k // 2
    yp = np.pad(y, pad, mode='edge')
    return np.convolve(yp, np.ones(k) / k, mode='valid')


def draw(ax, xs, ys_raw, color, label, logy=False):
    xs = np.asarray(xs, float); ys = smooth(ys_raw); ys_raw = np.asarray(ys_raw, float)
    ax.plot(xs, ys, color=color, lw=1.6, alpha=0.9, zorder=3, label=label)
    # progress markers (growing with training) sampled along the path
    idx = np.linspace(0, len(xs) - 1, min(7, len(xs))).astype(int)[1:-1]
    sizes = np.linspace(14, 55, len(idx))
    ax.scatter(xs[idx], ys[idx], s=sizes, color=color, alpha=0.5, zorder=4, edgecolors='none')
    # start = open circle (raw), end = filled diamond (raw); travel = start→end
    ax.scatter([xs[0]], [ys_raw[0]], s=55, facecolors='white', edgecolors=color, lw=1.8, zorder=6)
    ax.scatter([xs[-1]], [ys_raw[-1]], s=95, color=color, marker='D', edgecolors='black',
               lw=0.6, zorder=7)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8))
    for arm in ARMS:
        S = fc.load_summ(arm)
        x = [s['max_attn_pos0'] for s in S]
        vr = [s['v_ratio_pos0'] for s in S]
        hr = [s['h_ratio_pos0'] for s in S]
        c = fc.ARM_COLORS[arm]
        draw(axL, x, vr, c, fc.ARM_LABELS[arm])
        draw(axR, x, hr, c, fc.ARM_LABELS[arm], logy=True)

    axL.axhline(1.0, color='black', lw=0.6, alpha=0.35, ls='--')
    axL.set_xlabel('attention concentration  (max  attn→pos0)', fontsize=9)
    axL.set_ylabel('value-norm ratio  ‖v‖$_{pos0}$ / ‖v‖$_{rest}$', fontsize=9)
    axL.set_title('Concentration vs value-norm', fontsize=10, weight='bold')
    axL.text(0.015, 0.97, 'value AMPLIFIED ↑', transform=axL.transAxes, fontsize=7, color='gray')
    axL.text(0.015, 0.03, 'value DRAINED ↓', transform=axL.transAxes, fontsize=7, color='gray')

    axR.axhline(1.0, color='black', lw=0.6, alpha=0.35, ls='--')
    axR.set_yscale('log')
    axR.set_xlabel('attention concentration  (max  attn→pos0)', fontsize=9)
    axR.set_ylabel('massive-activation ratio  ‖h‖$_{pos0}$ / ‖h‖$_{rest}$  (log)', fontsize=9)
    axR.set_title('Concentration vs massive-activation', fontsize=10, weight='bold')

    for ax in (axL, axR):
        fc.style_ax(ax); ax.set_xlim(-0.03, 1.03)

    # corner annotations on left panel
    axL.text(0.62, 1.55, 'sigmoid:\nconcentration +\nvalue AMPLIFIED', fontsize=7.5,
             color='tab:red', ha='left', va='top')
    axL.text(0.97, 0.46, 'textinit:\nconcentration +\nsevere DRAIN', fontsize=7.5,
             color='tab:orange', ha='right', va='bottom')
    axL.text(0.30, 0.92, 'baseline / RF / g1gate:\nno concentration', fontsize=7.5,
             color='dimgray', ha='left', va='top')

    handles = [Line2D([0], [0], color=fc.ARM_COLORS[a], lw=2, label=fc.ARM_LABELS[a]) for a in ARMS]
    handles += [Line2D([0], [0], marker='o', color='gray', markerfacecolor='white', lw=0,
                       label='○ init', markersize=7),
                Line2D([0], [0], marker='D', color='gray', markerfacecolor='gray', lw=0,
                       label='◆ final', markersize=7)]
    fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=7.4, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Decoupling phase-portrait: each lever drives the three sink signatures into a different corner',
                 fontsize=11, weight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = 'analysis/fig2_phase_portrait.svg'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.svg', '.png'), dpi=160, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
