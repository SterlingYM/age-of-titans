"""Fig. age_vs_redshift: host t_MWA (top) and E[t_prog] (bottom) versus redshift for SF and Q hosts.

Reads results/host_sample_with_tprog.csv (step 02).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM

import config as C
from tprog.style import use_paper_style

COSMO = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
Z_GRID = np.linspace(0, 0.2, 100)
Z_WINDOW = 0.01


def rolling_median(z, y, mask):
    out = []
    for z0 in Z_GRID:
        sel = mask & (z >= z0 - Z_WINDOW / 2) & (z < z0 + Z_WINDOW / 2)
        out.append(np.median(y[sel]) if sel.any() else np.nan)
    return np.array(out)


def main():
    use_paper_style()
    df = pd.read_csv(C.RESULTS_DIR / 'host_sample_with_tprog.csv').dropna(subset=['prog_age_50'])
    z = df['redshift_best'].values
    groups = {   # mask, colour, linestyle of the running median
        'SF': (df['ssfr_50'].gt(C.LOGSSFR_SF).values, 'royalblue', '-'),
        'Q': (df['ssfr_50'].lt(C.LOGSSFR_Q).values, 'firebrick', '-.'),
        'All': (np.ones(len(df), bool), 'k', '--'),
    }

    fig, axes = plt.subplots(2, 1, figsize=(7, 9), dpi=150)
    plt.subplots_adjust(hspace=0)
    pos = [ax.get_position().bounds for ax in axes]
    ax_hist = [fig.add_axes([p[0] + p[2], p[1], 0.15, p[3]]) for p in pos]
    ax_top = fig.add_axes([pos[0][0], pos[0][1] + pos[0][3], pos[0][2], 0.15])

    for ax, axh, col in zip(axes, ax_hist, ['mass_weighted_age', 'prog_age']):
        y = df[f'{col}_50'].values
        yerr = (df[f'{col}_84'] - df[f'{col}_16']).values / 2
        axh.hist(y, bins=30, orientation='horizontal', color='gray')
        for name, (mask, color, ls) in groups.items():
            if name != 'All':
                ax.errorbar(z[mask], y[mask], yerr=yerr[mask], fmt='o', ms=0.5, lw=0.1,
                            c=color, alpha=0.3, rasterized=True)
                axh.hist(y[mask], bins=30, orientation='horizontal', color=color, histtype='step', lw=3)
            ax.plot(Z_GRID, rolling_median(z, y, mask), color=color, ls=ls, lw=2, zorder=10)
        ax.fill_between(Z_GRID, COSMO.age(Z_GRID).value, 15, color='gray', hatch='/', alpha=0.3)
        ax.set_xlim(0, 0.15)
        ax.set_ylim(0, 13.5)
        ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=13)
        axh.set_ylim(0, 13.5)
        axh.axis('off')

    z_bins = np.linspace(0, 0.15, 50)
    ax_top.hist(z, bins=z_bins, color='gray')
    for name in ['SF', 'Q']:
        mask, color, _ = groups[name]
        ax_top.hist(z[mask], bins=z_bins, color=color, histtype='step', lw=3)
    ax_top.set_xlim(0, 0.15)
    ax_top.axis('off')

    axes[0].text(0.11, 9, r'$\textbf{Quiescent}$', c='firebrick', fontsize=15)
    axes[0].text(0.11, 4, r'$\textbf{Star-forming}$', c='royalblue', fontsize=15)
    axes[1].text(0.11, 7, r'$\textbf{Quiescent}$', c='firebrick', fontsize=15)
    axes[1].text(0.11, 0.5, r'$\textbf{Star-forming}$', c='royalblue', fontsize=15)
    axes[1].text(0.12, 3.5, r'$\textbf{All}$', c='k', fontsize=15)
    axes[0].set_ylabel(r'Galaxy Age $t_\mathrm{MWA}$ [Gyr]')
    axes[1].set_ylabel(r'SN Progenitor Age $\mathbb{E}[t_\mathrm{prog}]$ [Gyr]')
    axes[0].set_xticklabels([])
    axes[1].set_xlabel('Redshift')

    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / 'age_vs_redshift.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
