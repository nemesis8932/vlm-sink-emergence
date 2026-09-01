#!/usr/bin/env python3
"""HERO: decoupling phase-portrait. Each arm is a PATH through (concentration, norm)
space that diverges into its own corner over training. The thesis as motion.

Left  : x = max attn->pos0 (concentration)   y = value-norm ratio @pos0
Right : x = max attn->pos0 (concentration)   y = h-norm ratio @pos0 (massive activation, log)

Smoothed path per arm; an open circle marks init, a diamond the final checkpoint.
Four levers leave from a shared origin (no concentration, unit norms) and end in four
distinct corners -> the signatures are independently steerable, not facets of one
phenomenon. Data identical to v1; this revision is styling only (palette/markers/grid).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter
import fig_common as fc

ARMS = fc.MAIN_ARMS + ['rf']

# palette + rcParams come from fig_common (shared, CVD-validated)
COLORS = fc.ARM_COLORS


def smooth(y, k=9):
    """edge-padded moving average, endpoints blended back to RAW values so the path
    connects to the init/final markers even when the signal moves fast near an end
    (textinit reorganizes quickly right after init — pure edge-padding leaves a gap)."""
    y = np.asarray(y, float)
    if len(y) < k: return y
    pad = k // 2
    yp = np.pad(y, pad, mode='edge')
    ys = np.convolve(yp, np.ones(k) / k, mode='valid')
    w = np.linspace(0.0, 1.0, pad + 1)          # 0 = raw at the very endpoint
    ys[:pad + 1] = (1 - w) * y[:pad + 1] + w * ys[:pad + 1]
    ys[-(pad + 1):] = (1 - w[::-1]) * y[-(pad + 1):] + w[::-1] * ys[-(pad + 1):]
    return ys


def draw(ax, xs, ys_raw, color, label):
    xs_raw = np.asarray(xs, float)
    xs = smooth(xs, 5); ys = smooth(ys_raw); ys_raw = np.asarray(ys_raw, float)
    # faint RAW probe points behind the smoothed path, so the smoothing (5-point on x,
    # 9-point on y) never hides the underlying scatter
    ax.scatter(xs_raw, ys_raw, s=3.0, color=color, alpha=0.16, zorder=2,
               edgecolors='none', rasterized=True)
    # white underlay = separation from grid + other paths
    ax.plot(xs, ys, color='white', lw=2.4, alpha=0.75, zorder=3, solid_capstyle='round')
    ax.plot(xs, ys, color=color, lw=1.3, alpha=0.95, zorder=4, label=label,
            solid_capstyle='round')
    # a few progress dots (grow with training)
    idx = np.linspace(0, len(xs) - 1, 6).astype(int)[1:-1]
    ax.scatter(xs[idx], ys[idx], s=np.linspace(5, 14, len(idx)), color=color,
               alpha=0.45, zorder=5, edgecolors='none')
    # init = open circle (raw values), final = diamond (raw values)
    ax.scatter([xs[0]], [ys_raw[0]], s=26, facecolors='white', edgecolors=color,
               lw=1.1, zorder=7)
    ax.scatter([xs[-1]], [ys_raw[-1]], s=50, color=color, marker='D',
               edgecolors='white', lw=0.9, zorder=8)


def main():
    # sized to the NeurIPS 5.5in text block so every font prints at its nominal size
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.5, 2.55))
    for arm in ARMS:
        S = fc.load_summ(arm)
        x = [s['max_attn_pos0'] for s in S]
        vr = [s['v_ratio_pos0'] for s in S]
        hr = [s['h_ratio_pos0'] for s in S]
        c = COLORS[arm]
        draw(axL, x, vr, c, fc.ARM_LABELS[arm])
        draw(axR, x, hr, c, fc.ARM_LABELS[arm])

    for ax in (axL, axR):
        ax.axhline(1.0, color='#666666', lw=0.8, alpha=0.55, ls=(0, (5, 4)), zorder=1)
        ax.set_xlim(-0.03, 1.03)
        ax.grid(True, which='major', color='#999999', alpha=0.16, lw=0.5)
        ax.set_axisbelow(True)
        fc.style_ax(ax)
        ax.set_xlabel('attention concentration (max attn→pos0)', fontsize=7)
        ax.tick_params(labelsize=6)

    axL.set_ylabel('value-norm ratio  ‖v‖$_{pos0}$ / ‖v‖$_{rest}$', fontsize=7)
    axL.set_title('Concentration vs value-norm', fontsize=7.5, weight='bold', pad=4)
    axL.text(0.02, 0.965, '↑ value amplified', transform=axL.transAxes, fontsize=5.5,
             color='#777777', style='italic')
    axL.text(0.02, 0.035, '↓ value drained', transform=axL.transAxes, fontsize=5.5,
             color='#777777', style='italic')

    axR.set_yscale('log')
    axR.set_ylabel('residual-norm ratio  ‖h‖$_{pos0}$ / ‖h‖$_{rest}$  (log)',
                   fontsize=7)
    axR.set_title('Concentration vs residual-norm', fontsize=7.5, weight='bold', pad=4)
    axR.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 40]))
    axR.yaxis.set_major_formatter(lambda v, _: f'{v:g}×')
    axR.yaxis.set_minor_formatter(NullFormatter())

    # corner annotations (left panel), colors matched to arms
    box = dict(boxstyle='round,pad=0.32', fc='white', ec='none', alpha=0.75)
    axL.text(0.24, 1.555, 'sigmoid: concentration\n+ value amplified', fontsize=5.6,
             color=COLORS['sigmoid'], ha='left', va='top', weight='bold', bbox=box)
    axL.text(0.985, 0.465, 'textinit: concentration\n+ severe drain', fontsize=5.6,
             color=COLORS['textinit'], ha='right', va='bottom', weight='bold', bbox=box)
    axL.text(0.315, 0.955, 'baseline / RF / g1gate:\nno concentration', fontsize=5.6,
             color='#555555', ha='left', va='top', weight='bold', bbox=box)

    handles = [Line2D([0], [0], color=COLORS[a], lw=2.4, label=fc.ARM_LABELS[a],
                      solid_capstyle='round') for a in ARMS]
    handles += [Line2D([0], [0], marker='o', color='#666666', markerfacecolor='white',
                       lw=0, label='init', markersize=6.5),
                Line2D([0], [0], marker='D', color='#666666', markerfacecolor='#666666',
                       markeredgecolor='white', lw=0, label='final', markersize=6.5)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, -0.06), handlelength=1.5, columnspacing=1.0)
    # no suptitle: the LaTeX caption carries the title
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    out = 'analysis/fig2_phase_portrait.svg'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.svg', '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.svg', '.pdf'), bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
