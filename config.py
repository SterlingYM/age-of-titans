"""Paths and analysis choices shared by all numbered scripts.

Edit this file (or set the environment variables below) to point at a new SED-fit run.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- inputs
# Bagpipes Iyer+2019 SFH posteriors, one group per CID with
# `sfh_samps` (N_samp, N_t) [Msun/yr] and `lookback_times_gyr` (N_t,)
SFH_H5 = Path(os.environ.get(
    'TPROG_SFH_H5',
    '/project2/rkessler/SURVEYS/ATLAS/USERS/ymurakami/combined_iyer2019_sfhs_enlarged_apertures.h5'))

# Global host properties (stellar mass, sSFR, MWA, colours) from the same SED-fit run
HOST_PROPS = Path(os.environ.get('TPROG_HOST_PROPS', ROOT / 'data/inputs/host_props.csv'))

RELIABLE = ROOT / 'data/inputs/all_parameters_reliable.csv'   # hosts with reliable UV-MIR fits
DDLR = ROOT / 'data/inputs/titan_ddlr.csv'
SNIA = ROOT / 'data/inputs/titan_snia.csv'                    # redshift, classification flag
WDSIM_DIR = ROOT / 'data/inputs/WDsim'                        # binary-evolution WD formation times

# ---------------------------------------------------------------- outputs
AGES_DIR = Path(os.environ.get('TPROG_AGES_DIR', ROOT / 'data/ages'))   # step 01 output
RESULTS_DIR = ROOT / 'results'                                          # tables (steps 02-03)
FIG_DIR = ROOT / 'figures'

# ---------------------------------------------------------------- DTD
# (t_min [Gyr], beta); the first entry is the paper baseline
DTD_VARIANTS = [
    (0.04, -1.07),
    (0.04, -1.00),
    (0.04, -1.13),
    (0.30, -1.00),
    (0.30, -1.07),
    (0.30, -1.13),
]
FIDUCIAL = DTD_VARIANTS[0]
T_MAX_GYR = 14.0
DT_GRID_GYR = 0.01
N_SFH_DRAWS = 1000      # SFH posterior draws per host
N_AGE_PER_SFH = 100     # progenitor ages sampled from each draw's P(t_prog)
N_PDF_DRAWS_SAVED = 100 # draws whose individual age samples are kept (for P(t_prog) histograms)

# ---------------------------------------------------------------- sample and subsamples
DDLR_MAX = 10
CLASS_FLAG = 1          # spectroscopically secure SNe Ia
LOGM_SPLIT = 10.0
LOGSSFR_SF = -10.5      # star-forming: log sSFR > -10.5
LOGSSFR_Q = -11.0       # quiescent:    log sSFR < -11.0 (intermediate range excluded)

# ---------------------------------------------------------------- literature HR inputs
C25_SLOPE, C25_SLOPE_ERR = 0.030, 0.004     # HR-age slope [mag/Gyr], Chung+2025 / Son+2025
W26_SLOPE, W26_SLOPE_ERR = 0.007, 0.012     # HR-t_MWA slope after corrections, Wiseman+2026
# Wiseman+2026 DES low-z SF-Q HR offsets [mag]: (value, error, label)
W26_HR_OFFSETS = [
    (0.037, 0.019, 'All, after mass-step'),
    (0.028, 0.022, 'High-mass, after bias correction'),
    (0.037, 0.022, 'High-mass, before bias correction'),
]


def ages_path(t_min, beta):
    return AGES_DIR / f'tprog_tmin{t_min*1e3:.0f}Myr_beta{beta:+.2f}.h5'
