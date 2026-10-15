"""Fig. HR_separation_tests: SF-Q Hubble-residual offsets measured by W26 versus the offsets
predicted from our high-mass SF-Q progenitor-age difference with the C25 and W26 slopes.

Reads results/dtd_variants.csv (step 03).
"""
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from tprog.style import use_paper_style

BETA_COLOR = {-1.00: 'darkgoldenrod', -1.07: 'royalblue', -1.13: 'seagreen'}
BETA_Y = {-1.00: 1.8, -1.07: -0.2, -1.13: -2.2}     # row position of the 40 Myr point
TMIN_STYLE = {40: ('d', None, 0.0), 300: ('o', 'w', -0.6)}   # marker, face colour, y offset
W26_BLOCK_OFFSET = -7.8
LABEL_X = 0.14


def main():
    use_paper_style()
    tab = pd.read_csv(C.RESULTS_DIR / 'dtd_variants.csv')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # top block: W26 measurements
    for y, (val, err, label) in zip([7, 6, 5], C.W26_HR_OFFSETS):
        ax.errorbar(val, y, xerr=err, fmt='o', capsize=5, color='k' if y == 6 else 'gray')
        ax.text(0.065, y, label, fontsize=13, va='center')
    hm_bc = C.W26_HR_OFFSETS[1]
    ax.axvspan(hm_bc[0] - hm_bc[1], hm_bc[0] + hm_bc[1], color='gray', alpha=0.2, lw=0)
    ax.text(-0.03, 6.5, 'W26\nDES low-z', fontsize=14, fontweight='bold')
    ax.axhline(3.8, ls=':', c='k')

    # middle (C25 slope) and bottom (W26 slope) blocks: our predictions
    for tag, y0, title in [('C25', 0.0, 'TITAN hosts\n+ C25 slope'),
                           ('W26', W26_BLOCK_OFFSET, 'TITAN hosts\n+ W26 slope')]:
        ax.text(-0.03, 2.0 + y0 + (0.3 if tag == 'W26' else 0), title, fontsize=14, fontweight='bold')
        for _, r in tab.iterrows():
            marker, mfc, dy = TMIN_STYLE[int(r['t_min_myr'])]
            color = BETA_COLOR[round(r['beta'], 2)]
            eb = ax.errorbar(r[f'dHR_{tag}'], BETA_Y[round(r['beta'], 2)] + dy + y0,
                             xerr=r[f'dHR_{tag}_err'], fmt=marker, capsize=5, ms=8,
                             color=color, mec=color, mfc=mfc)
            if tag == 'W26':
                eb[-1][0].set_linestyle(':')
        for beta, y in BETA_Y.items():
            ax.text(LABEL_X, y - 0.3 + y0, rf'$\beta_\text{{DTD}}={beta:.2f}$', fontsize=13,
                    va='center', ha='center')
    ax.axhline(-4, ls=':', c='k')

    # legend for t_min
    ax.errorbar(-1, 1.1, xerr=0.004, fmt='d', capsize=5, color='k', ms=8, label='40 Myr')
    ax.errorbar(-1, 0.1, xerr=0.004, fmt='o', capsize=5, color='k', mfc='w', ms=8, label='300 Myr')
    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.005, 0.6), frameon=True,
                    title=r'$t_\text{min}$', title_fontsize=13,
                    handletextpad=0.2, borderpad=0.1, labelspacing=0.8)
    leg.get_frame().set_boxstyle('round')
    leg.get_frame().set_edgecolor('gray')

    ax.set_xlim(-0.035, 0.16)
    ax.set_ylim(-11.5, 8.2)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.tick_params(direction='in', top=True, right=True)
    ax.set_xlabel(r'$\Delta_\text{SF-Q}\text{HR}$ [mag]')

    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / 'HR_separation_tests.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
