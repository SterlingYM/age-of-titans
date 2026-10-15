"""Step 03: summary statistics for each DTD variant (paper Table "DTD_variants_stats",
and the SF-Q age difference / predicted HR offsets used in Figs. HR_separation_tests and
WDsim_and_S25_comparison).

Writes results/dtd_variants.csv and prints the LaTeX table rows.
"""
import numpy as np
import pandas as pd

import config as C
from tprog.data import load_ages, load_hosts, subsamples


def group_mean(ev, mask):
    """Mean E[t_prog] over the hosts in `mask`, per SFH draw -> (N_SFH_DRAWS,)."""
    return ev[mask].mean(axis=0)


def main():
    df = load_hosts()
    rows = []
    for t_min, beta in C.DTD_VARIANTS:
        cids, ev, samples = load_ages(t_min, beta, cids=df['transient'])
        hosts = df.set_index('transient').loc[cids].reset_index()
        masks = {k: m.values for k, m in subsamples(hosts).items()}

        row = dict(t_min_myr=round(t_min * 1e3), beta=beta, n_hosts=len(cids))
        row.update(zip(['pdf_16', 'pdf_50', 'pdf_84'], np.percentile(samples, [16, 50, 84])))
        row.update(zip(['ev_16', 'ev_50', 'ev_84'], np.percentile(ev, [16, 50, 84])))
        row['mean'] = samples.mean()
        for name, m in masks.items():
            lo, med, hi = np.percentile(group_mean(ev, m), [16, 50, 84])
            row[f'{name}_mean'], row[f'{name}_err'] = med, (hi - lo) / 2

        diff = row['High-mass Q_mean'] - row['High-mass SF_mean']
        diff_err = np.hypot(row['High-mass Q_err'], row['High-mass SF_err'])
        row['dage_HM_Q_SF'], row['dage_HM_Q_SF_err'] = diff, diff_err
        for tag, slope, slope_err in [('C25', C.C25_SLOPE, C.C25_SLOPE_ERR),
                                      ('W26', C.W26_SLOPE, C.W26_SLOPE_ERR)]:
            row[f'dHR_{tag}'] = slope * diff
            row[f'dHR_{tag}_err'] = np.hypot(slope_err * diff, slope * diff_err)
        rows.append(row)

    out = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(C.RESULTS_DIR / 'dtd_variants.csv', index=False)
    print(f'wrote {C.RESULTS_DIR / "dtd_variants.csv"}\n')

    for r in rows:
        print(f"{r['t_min_myr']} & ${r['beta']:+.2f}$ & "
              f"{r['pdf_16']:.1f}, {r['pdf_50']:.1f}, {r['pdf_84']:.1f} & "
              f"{r['ev_16']:.1f}, {r['ev_50']:.1f}, {r['ev_84']:.1f} & {r['mean']:.1f} \\\\")
    print()
    cols = ['t_min_myr', 'beta'] + [f'{k}_mean' for k in subsamples(df)] + \
           ['dage_HM_Q_SF', 'dage_HM_Q_SF_err', 'dHR_C25', 'dHR_C25_err']
    print(out[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
