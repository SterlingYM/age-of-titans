"""SFH x DTD -> SN Ia progenitor-age PDF, and sampling from it."""
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


def dtd(t_gyr, beta, t_min):
    """Power-law delay-time distribution (arbitrary normalization), zero below t_min."""
    t_gyr = np.asarray(t_gyr, dtype=float)
    out = np.zeros_like(t_gyr)
    ok = t_gyr >= t_min
    out[ok] = t_gyr[ok] ** beta
    return out


def age_grid(t_min, t_max=14.0, dt=0.01):
    return np.arange(t_min, t_max, dt)


def progenitor_age_pdf(sfh, lookback_gyr, grid, beta, t_min, normalize=True):
    """P(t_prog) on `grid` for one SFH (N_t,) or many SFHs (N_draw, N_t).

    The SFH is linearly interpolated onto `grid`, held constant beyond the ends of
    `lookback_gyr` (as np.interp does).
    """
    sfh = np.atleast_2d(sfh)
    i = np.clip(np.searchsorted(lookback_gyr, grid) - 1, 0, len(lookback_gyr) - 2)
    w = np.clip((grid - lookback_gyr[i]) / (lookback_gyr[i + 1] - lookback_gyr[i]), 0, 1)
    sfh_grid = sfh[:, i] * (1 - w) + sfh[:, i + 1] * w
    pdf = sfh_grid * dtd(grid, beta, t_min)
    if normalize:
        pdf /= trapezoid(pdf, grid, axis=1)[:, None]
    return pdf


def sample_ages(pdf, grid, n, rng):
    """Draw `n` ages from each row of `pdf` by inverse-CDF sampling -> (N_draw, n)."""
    cdf = cumulative_trapezoid(pdf, grid, axis=1, initial=0)
    cdf /= cdf[:, -1:]
    u = rng.random((len(pdf), n))
    return np.array([np.interp(u_i, cdf_i, grid) for u_i, cdf_i in zip(u, cdf)])
