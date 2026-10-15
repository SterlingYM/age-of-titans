"""Fig. progenitor_age_histogram: P(t_prog) and P(E[t_prog]) for the full sample and the four
host subsamples (baseline DTD)."""
import matplotlib.pyplot as plt
import numpy as np

import config as C
from tprog.data import SUBSAMPLE_LABEL, SUBSAMPLE_STYLE, load_ages, load_hosts, subsamples
from tprog.style import use_paper_style


def main():
    use_paper_style()
    df = load_hosts()
    cids, ev, samples = load_ages(*C.FIDUCIAL, cids=df['transient'])
    hosts = df.set_index('transient').loc[cids].reset_index()

    bins = np.linspace(0, 12, 50)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.16)

    # all panels normalized to the full sample, so subsample areas show their fractions
    for ax, data in zip(axes, [samples, ev]):
        counts, edges = np.histogram(data, bins=bins)
        norm = data.size * np.diff(edges)
        ax.stairs(counts / norm, edges, color='gray', fill=True)
        for name, mask in subsamples(hosts).items():
            color, ls = SUBSAMPLE_STYLE[name]
            ax.stairs(np.histogram(data[mask.values], bins=bins)[0] / norm, edges,
                      color=color, ls=ls, lw=3, label=SUBSAMPLE_LABEL[name])
        ax.set_xlim(0, 12)
        ax.set_xlabel(r'$t_\mathrm{prog}$ [Gyr]')
        ax.legend(frameon=False)

    top = axes[0].twiny()
    top.set_xlim(0, 12)
    pct = np.arange(25, 101, 25)
    top.set_xticks(np.percentile(samples, pct))
    top.set_xticklabels([rf'{p}\%' for p in pct])
    top.set_xlabel(r'Percentiles of $t_\mathrm{prog}$')
    top.minorticks_off()
    top.tick_params(direction='in')

    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$P(t_\mathrm{prog})$')
    axes[1].set_ylabel(r'$P\left(\mathbb{E}[t_\mathrm{prog}]\right)$')
    axes[0].tick_params(which='both', direction='in', top=False, right=True)
    axes[1].tick_params(which='both', direction='in', top=True, right=True)

    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / 'progenitor_age_histogram.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
