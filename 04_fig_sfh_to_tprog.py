"""Fig. SFH_to_t_prog: sSFH, P(t_prog) and P(E[t_prog]) for three example hosts (baseline DTD)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde

import config as C
from tprog.data import load_ages, load_sfh
from tprog.dtd import age_grid, progenitor_age_pdf
from tprog.style import use_paper_style

EXAMPLES = [('2017hgz', 'yellowgreen', ':'), ('2019yyv', 'royalblue', '--'), ('2024vjb', 'firebrick', '-')]


def main():
    use_paper_style()
    t_min, beta = C.FIDUCIAL
    props = pd.read_csv(C.HOST_PROPS).set_index('transient')
    cids, ev, _ = load_ages(t_min, beta, cids=[c for c, _, _ in EXAMPLES], pdf=False)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    plt.subplots_adjust(hspace=0)
    for (cid, color, ls), ev_host in zip(EXAMPLES, ev):
        sfh, lookback = load_sfh(cid)
        sfh = np.where(sfh > 0, sfh, np.nan)
        ssfh_lo, ssfh_med, ssfh_hi = np.nanpercentile(
            np.log10(sfh) - props.loc[cid, 'stellar_mass_50'], [16, 50, 84], axis=0)

        # P(t_prog) from the 16/50/84th-percentile sSFH, normalized by the median curve
        ok = np.isfinite(ssfh_med)
        grid = age_grid(t_min, lookback[ok].max(), C.DT_GRID_GYR)
        pdfs = progenitor_age_pdf(10 ** np.array([ssfh_lo, ssfh_med, ssfh_hi])[:, ok], lookback[ok],
                                  grid, beta, t_min, normalize=False)
        pdfs /= trapezoid(pdfs[1], grid)

        axes[0].fill_between(lookback, ssfh_lo, ssfh_hi, color=color, alpha=0.1, lw=0)
        axes[0].plot(lookback, ssfh_med, color=color, ls=ls, lw=2, label=cid)
        axes[1].fill_between(grid, pdfs[0], pdfs[2], color=color, alpha=0.1, lw=0)
        axes[1].plot(grid, pdfs[1], color=color, ls=ls, lw=2, zorder=10)
        kde = gaussian_kde(ev_host)(grid)
        axes[2].fill_between(grid, 0, kde, color=color, alpha=0.3, lw=0)
        axes[2].plot(grid, kde, color=color, ls=ls, lw=2, zorder=10)

    axes[0].set_ylabel(r'$\log \mathrm{sSFH}$ [yr$^{-1}$]')
    axes[0].set_ylim(-12.5, -8.7)
    axes[0].legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    axes[1].set_yscale('log')
    axes[1].set_ylim(5e-4, 2)
    axes[1].set_ylabel(r'$P(t_\mathrm{prog})$')
    axes[2].set_ylim(0)
    axes[2].set_ylabel(r'$P\left(\mathbb{E}[t_\mathrm{prog}]\right)$')
    axes[2].set_xlabel('Lookback Time [Gyr]')
    for ax in axes:
        ax.set_xlim(0, 13.3)
        ax.tick_params(direction='in', which='both', top=True, right=True)
    for ax in axes[:2]:
        ax.set_xticklabels([])
    fig.align_ylabels(axes)

    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / 'SFH_to_t_prog.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
