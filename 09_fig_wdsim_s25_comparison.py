"""Fig. WDsim_and_S25_comparison (appendix): minimum WD formation time from binary simulations
(left) and the C25-slope HR offset predicted for each DTD variant (right).

Reads data/inputs/WDsim/*.npy and results/dtd_variants.csv (step 03).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm

import config as C
from tprog.style import use_paper_style

LEVELS = [20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000]
BETA_MARKER = {-1.00: ('o', -8, 10), -1.07: ('d', 0, 12), -1.13: ('*', 8, 14)}   # marker, x offset, size


def load_wd_grid(name):
    """{(M1, M2): tau} dictionary -> (M1 values, M2 values, tau[M2, M1]) [Myr]."""
    d = np.load(C.WDSIM_DIR / f'M1_M2_tau_{name}.npy', allow_pickle=True).item()
    m1, m2, tau = (np.array(v, float) for v in zip(*[(a, b, t) for (a, b), t in d.items()]))
    x, y = np.unique(m1), np.unique(m2)
    grid = np.full((len(y), len(x)), np.nan)
    grid[np.searchsorted(y, m2), np.searchsorted(x, m1)] = tau
    return x, y, grid


def main():
    use_paper_style()
    tab = pd.read_csv(C.RESULTS_DIR / 'dtd_variants.csv')
    norm = BoundaryNorm(LEVELS, ncolors=256)
    cmap = plt.get_cmap('cividis_r')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.40)

    # left: binary-simulation WD formation times
    ax = axes[0]
    ax.pcolor(*load_wd_grid('ONe'), cmap=cmap, norm=norm, shading='nearest', zorder=-1, alpha=0.5,
              hatch='..', rasterized=True, linewidth=0, edgecolor='none')
    im = ax.pcolormesh(*load_wd_grid('CO'), cmap=cmap, norm=norm, shading='nearest', zorder=1)
    p = ax.get_position().bounds
    cax = fig.add_axes([p[0] + p[2] + 0.01, p[1], 0.02, p[3]])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(LEVELS)
    cbar.set_ticklabels([str(l) for l in LEVELS])
    cax.set_title(r'$\mathrm{\tau_{WD,min}~(Myr)}$', fontsize=15, pad=10)
    cax.tick_params(labelsize=13)
    ax.set_xticks(range(1, 12))
    ax.set_yticks(range(1, 11))
    ax.minorticks_off()
    ax.tick_params(direction='in', top=True, right=True, labelsize=13)
    ax.set_xlabel(r'$\mathrm{M_1~(M_{\odot})}$')
    ax.set_ylabel(r'$\mathrm{M_2~(M_{\odot})}$')
    ax.text(1.5, 9.0, 'Minimum WD formation time\nin binary systems', fontsize=15, fontweight='bold')
    ax.text(1.5, 8.0, 'Dotted: ONe WD', fontsize=15)
    ax.text(1.5, 7.5, 'Solid: CO WD', fontsize=15)

    # right: predicted HR offsets vs t_min, on the colour scale of the left panel
    ax = axes[1]
    ax.set_xlim(0, 350)
    ax.set_ylim(0, 4)
    for lo, hi in zip(LEVELS[:-2], LEVELS[1:-1]):
        ax.axvspan(lo, hi, color=cmap(norm(lo)), zorder=-1)
    ax.set_ylabel(r'$\Delta t_\mathrm{prog}\ \mathrm{High\text{-}mass\ Q - High\text{-}mass\ SF}$' + '\n[Gyr]')
    ax.set_xlabel(r'$\tau_\mathrm{min}$ (Myr)')
    ax.tick_params(direction='in', top=True, labelsize=13)
    ax.minorticks_off()

    twin = ax.twinx()
    twin.set_ylim(np.array(ax.get_ylim()) * C.C25_SLOPE)
    twin.set_ylabel(r'$\mathrm{S25\ model\ prediction}$' + '\n' + r'$\Delta$HR [mag]')
    twin.tick_params(direction='in', labelsize=13)
    twin.minorticks_off()
    for _, r in tab.iterrows():
        marker, dx, ms = BETA_MARKER[round(r['beta'], 2)]
        twin.errorbar(r['t_min_myr'] + dx, r['dHR_C25'], yerr=r['dHR_C25_err'], fmt=marker,
                      capsize=0, color='tomato', ms=ms, mec='k')
    for beta in sorted(BETA_MARKER, reverse=True):
        marker = BETA_MARKER[beta][0]
        twin.errorbar(0, 0, yerr=1, c='k', mfc='tomato', fmt=marker, ms=10 if marker == '*' else 8,
                      label=rf'$\beta_\text{{DTD}}={beta:.2f}$')
    twin.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)

    hm_bc, hm_bc_err, _ = C.W26_HR_OFFSETS[1]
    twin.axhline(hm_bc, color='k', ls='--', lw=1.5)
    twin.axhspan(hm_bc - hm_bc_err, hm_bc + hm_bc_err, alpha=0.8, lw=0, hatch='//', zorder=10,
                 ec='k', fc='none', rasterized=True)
    twin.axvspan(100, 200, ymin=0.25, ymax=0.33, color=cmap(norm(110)), lw=0, hatch='//', zorder=10)
    twin.axvspan(200, 305, ymin=0.25, ymax=0.33, color=cmap(norm(300)), lw=0, hatch='//', zorder=10)
    twin.text(200, 0.0325, 'W26 bias-corrected HR separation', fontsize=13, va='bottom', ha='center',
              zorder=100)

    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / 'WDsim_and_S25_comparison.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
