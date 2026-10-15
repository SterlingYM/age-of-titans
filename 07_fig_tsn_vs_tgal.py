"""Fig. tSN_vs_tGal: E[t_prog] versus host mass-weighted age, and their difference.

Reads results/host_sample_with_tprog.csv (step 02).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from tprog.data import SUBSAMPLE_LABEL, SUBSAMPLE_STYLE, subsamples
from tprog.style import use_paper_style

MARKERS = {   # marker, size, hollow, linewidth
    'Low-mass Q': ('o', 3, False, 0.5),
    'Low-mass SF': ('s', 3, False, 0.5),
    'High-mass Q': ('d', 10, True, 0.3),
    'High-mass SF': ('+', 50, False, 0.5),
}


def main():
    use_paper_style()
    df = pd.read_csv(C.RESULTS_DIR / 'host_sample_with_tprog.csv').dropna(subset=['prog_age_50'])
    t_mwa, t_prog = df['mass_weighted_age_50'], df['prog_age_50']
    dage = t_prog - t_mwa

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    x0, y0, w, h = ax.get_position().bounds
    ax_bot = fig.add_axes([x0, y0 - 0.4, w, 0.4])
    ax_top = fig.add_axes([x0, y0 + h, w, 0.2])
    ax_right = fig.add_axes([x0 + w, y0, 0.2, h])
    ax_botright = fig.add_axes([x0 + w, y0 - 0.4, 0.2, 0.4])

    age_bins, diff_bins = np.linspace(0, 12, 60), np.linspace(-6, 1, 60)
    ax_top.hist(t_mwa, bins=age_bins, color='gray')
    ax_right.hist(t_prog, bins=age_bins, orientation='horizontal', color='gray')
    ax_botright.hist(dage, bins=diff_bins, orientation='horizontal', color='gray')

    for name, mask in subsamples(df).items():
        color, _ = SUBSAMPLE_STYLE[name]
        marker, size, hollow, lw = MARKERS[name]
        kw = dict(s=size, marker=marker, lw=lw, label=SUBSAMPLE_LABEL[name],
                  fc='none' if hollow else color, **({} if marker == '+' else {'ec': color}))
        ax.scatter(t_mwa[mask], t_prog[mask], **kw)
        ax_bot.scatter(t_mwa[mask], dage[mask], **kw)
        ax_top.hist(t_mwa[mask], bins=age_bins, color=color, alpha=0.5, histtype='step')
        ax_right.hist(t_prog[mask], bins=age_bins, orientation='horizontal', color=color, histtype='step')
        ax_botright.hist(dage[mask], bins=diff_bins, orientation='horizontal', color=color,
                         alpha=0.5, histtype='step')

    leg = ax.legend(frameon=False, fontsize=13)
    for handle in leg.legend_handles:
        handle.set_sizes([50])
        handle.set_linewidth(1)

    ax.plot([0, 12], [0, 12], c='gold', ls='--')
    ax_bot.axhline(0, c='gold', ls='--')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.set_xticklabels([])
    ax.set_ylabel(r'$\mathbb{E}[t_\mathrm{prog}]$: PDF-weighted Progenitor Age [Gyr]')
    ax_bot.set_xlim(ax.get_xlim())
    ax_bot.set_ylim(-6, 1)
    ax_bot.set_yticks(np.arange(-6, 1, 2))
    ax_bot.set_xlabel('Mass-Weighted Galaxy Age [Gyr]')
    ax_bot.set_ylabel(r'$\Delta$ age [Gyr]')
    for a in [ax, ax_bot]:
        a.tick_params(which='both', direction='in', top=True, right=True)
    ax_top.set_xlim(ax.get_xlim())
    ax_right.set_ylim(ax.get_ylim())
    ax_botright.set_ylim(ax_bot.get_ylim())
    for a in [ax_top, ax_right, ax_botright]:
        a.axis('off')

    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / 'tSN_vs_tGal.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
