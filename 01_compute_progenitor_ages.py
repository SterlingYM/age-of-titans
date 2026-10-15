"""Step 01: SFH posteriors -> SN Ia progenitor-age samples, for every DTD variant.

For each host, N_SFH_DRAWS SFH posterior draws are multiplied by the DTD to give
P(t_prog); N_AGE_PER_SFH ages are sampled from each. Saved per host and variant:
  ev       (N_SFH_DRAWS,)                     E[t_prog] of each draw (mean of its sampled ages)
  samples  (N_PDF_DRAWS_SAVED, N_AGE_PER_SFH) the sampled ages of the first draws, for P(t_prog)

The SFH file is read once; all variants are computed in the same pass. Re-running skips
hosts already present in every output file.

    python 01_compute_progenitor_ages.py [--workers 28] [--limit N]
"""
import argparse
import multiprocessing as mp
import zlib

import h5py
import numpy as np
from tqdm import tqdm

import config as C
from tprog.dtd import age_grid, progenitor_age_pdf, sample_ages

_sfh_file = None


def _open_sfh():
    global _sfh_file
    _sfh_file = h5py.File(C.SFH_H5, 'r')


def process_host(cid):
    sfh = _sfh_file[cid]['sfh_samps'][:]
    lookback = _sfh_file[cid]['lookback_times_gyr'][:]
    rng = np.random.default_rng(zlib.crc32(cid.encode()))
    draws = sfh[rng.choice(len(sfh), size=C.N_SFH_DRAWS, replace=True)]
    out = {}
    for t_min, beta in C.DTD_VARIANTS:
        grid = age_grid(t_min, C.T_MAX_GYR, C.DT_GRID_GYR)
        pdf = progenitor_age_pdf(draws, lookback, grid, beta, t_min)
        ages = sample_ages(pdf, grid, C.N_AGE_PER_SFH, rng).astype(np.float32)
        out[(t_min, beta)] = (ages.mean(axis=1), ages[:C.N_PDF_DRAWS_SAVED])
    return cid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=mp.cpu_count())
    ap.add_argument('--limit', type=int, default=None, help='first N hosts only (testing)')
    args = ap.parse_args()

    C.AGES_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(C.SFH_H5, 'r') as f:
        cids = list(f.keys())[:args.limit]
    done = set(cids)
    for variant in C.DTD_VARIANTS:
        path = C.ages_path(*variant)
        if path.exists():
            with h5py.File(path, 'r') as f:
                done &= set(f.keys())
        else:
            done = set()
    todo = [c for c in cids if c not in done]
    print(f'SFH file: {C.SFH_H5}\n{len(cids)} hosts, {len(todo)} to process, {len(C.DTD_VARIANTS)} DTD variants')

    outputs = {v: h5py.File(C.ages_path(*v), 'a') for v in C.DTD_VARIANTS}
    for v, f in outputs.items():
        f.attrs.update(t_min_gyr=v[0], beta=v[1], sfh_file=str(C.SFH_H5),
                       n_sfh_draws=C.N_SFH_DRAWS, n_age_per_sfh=C.N_AGE_PER_SFH)
    try:
        with mp.Pool(args.workers, initializer=_open_sfh) as pool:
            for cid, res in tqdm(pool.imap_unordered(process_host, todo, chunksize=4), total=len(todo)):
                for v, (ev, samples) in res.items():
                    if cid in outputs[v]:
                        del outputs[v][cid]
                    g = outputs[v].create_group(cid)
                    g['ev'] = ev
                    g['samples'] = samples
    finally:
        for f in outputs.values():
            f.close()


if __name__ == '__main__':
    main()
