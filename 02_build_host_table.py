"""Step 02: analysis-sample host table with per-host E[t_prog] percentiles (baseline DTD).

Writes results/host_sample_with_tprog.csv: host properties + prog_age_{16,50,84}, the
percentiles of E[t_prog] over SFH posterior draws.
"""
import numpy as np

import config as C
from tprog.data import load_ages, load_hosts


def main():
    df = load_hosts()
    cids, ev, _ = load_ages(*C.FIDUCIAL, cids=df['transient'], pdf=False)
    p16, p50, p84 = np.percentile(ev, [16, 50, 84], axis=1)
    ages = dict(zip(cids, zip(p16, p50, p84)))
    for i, q in enumerate(['16', '50', '84']):
        df[f'prog_age_{q}'] = df['transient'].map(lambda c: ages.get(c, (np.nan,) * 3)[i])

    missing = df['prog_age_50'].isna().sum()
    print(f'{len(df)} hosts in sample; {missing} without SFH posteriors in {C.SFH_H5.name}')
    C.RESULTS_DIR.mkdir(exist_ok=True)
    out = C.RESULTS_DIR / 'host_sample_with_tprog.csv'
    df.to_csv(out, index=False)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
