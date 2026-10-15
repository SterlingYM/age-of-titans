"""Load the host sample, define subsamples, and read progenitor-age products."""
import h5py
import numpy as np
import pandas as pd

import config as C


def load_hosts():
    """Host properties for the analysis sample (paper: 6,983 hosts).

    Cuts: reliable UV-MIR SED fit, d_DLR <= DDLR_MAX, classification flag == CLASS_FLAG.
    """
    df = pd.read_csv(C.HOST_PROPS)
    reliable = pd.read_csv(C.RELIABLE)['objname']
    ddlr = pd.read_csv(C.DDLR).rename(columns={'objname': 'transient'})[['transient', 'd_dlr']]
    snia = pd.read_csv(C.SNIA).rename(columns={'objname': 'transient'})
    df = (df.merge(ddlr, on='transient', how='left')
            .merge(snia[['transient', 'class_flag', 'redshift_best']], on='transient', how='left'))
    keep = df['transient'].isin(reliable) & df['d_dlr'].le(C.DDLR_MAX) & df['class_flag'].eq(C.CLASS_FLAG)
    return df[keep].reset_index(drop=True)


def subsamples(df):
    """Boolean masks for the four host types used throughout the paper (ordered for plotting)."""
    lowmass = df['stellar_mass_50'].le(C.LOGM_SPLIT)
    sf = df['ssfr_50'].gt(C.LOGSSFR_SF)
    q = df['ssfr_50'].lt(C.LOGSSFR_Q)
    return {
        'Low-mass Q': lowmass & q,
        'Low-mass SF': lowmass & sf,
        'High-mass Q': ~lowmass & q,
        'High-mass SF': ~lowmass & sf,
    }


SUBSAMPLE_STYLE = {   # colour, linestyle
    'Low-mass Q': ('orange', ':'),
    'Low-mass SF': ('royalblue', '--'),
    'High-mass Q': ('firebrick', '-.'),
    'High-mass SF': ('yellowgreen', '-'),
}

SUBSAMPLE_LABEL = {
    'Low-mass Q': r'Low-mass Q ($\log\mathrm{sSFR} < -11$)',
    'Low-mass SF': r'Low-mass SF ($\log\mathrm{sSFR} > -10.5$)',
    'High-mass Q': r'High-mass Q ($\log\mathrm{sSFR} < -11$)',
    'High-mass SF': r'High-mass SF ($\log\mathrm{sSFR} > -10.5$)',
}


def load_ages(t_min, beta, cids=None, pdf=True):
    """Read step-01 output for one DTD variant.

    Returns (cids, ev, samples):
      ev      (N_host, N_SFH_DRAWS)  E[t_prog] for each SFH posterior draw [Gyr]
      samples (N_host, N_PDF_DRAWS_SAVED, N_AGE_PER_SFH)  individual t_prog draws [Gyr], or None
    Hosts are returned in the order of `cids` (default: all hosts in the file).
    """
    with h5py.File(C.ages_path(t_min, beta), 'r') as f:
        keys = list(f.keys()) if cids is None else [c for c in cids if c in f]
        ev = np.stack([f[k]['ev'][:] for k in keys])
        samples = np.stack([f[k]['samples'][:] for k in keys]) if pdf else None
    return np.array(keys), ev, samples


def load_sfh(cid):
    with h5py.File(C.SFH_H5, 'r') as f:
        return f[cid]['sfh_samps'][:], f[cid]['lookback_times_gyr'][:]
