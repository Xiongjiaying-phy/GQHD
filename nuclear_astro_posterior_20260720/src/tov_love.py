"""Coupled TOV and l=2 tidal-Love integration in G=c=1 units."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


MEVFM3_TO_KM2 = 1.3234e-6
MSUN_TO_KM = 1.4766


class BarotropicEOS:
    def __init__(self, pressure_MeV_fm3: np.ndarray, energy_MeV_fm3: np.ndarray):
        p = np.asarray(pressure_MeV_fm3, dtype=float)
        e = np.asarray(energy_MeV_fm3, dtype=float)
        good = np.isfinite(p) & np.isfinite(e) & (p >= 0.0) & (e > 0.0)
        p, e = p[good], e[good]
        order = np.argsort(p)
        p, e = p[order], e[order]
        keep = np.r_[True, (np.diff(p) > 0.0) & (np.diff(e) > 0.0)]
        p, e = p[keep], e[keep]
        if len(p) < 8:
            raise ValueError("EOS needs at least eight strictly monotonic points")
        self.p_mev = p
        self.e_mev = e
        self.p = p * MEVFM3_TO_KM2
        self.e = e * MEVFM3_TO_KM2
        self._e_of_p = PchipInterpolator(self.p, self.e, extrapolate=True)
        self._de_dp = self._e_of_p.derivative()

    def epsilon(self, pressure_km2: float) -> float:
        return float(self._e_of_p(pressure_km2))

    def cs2(self, pressure_km2: float) -> float:
        de_dp = float(self._de_dp(pressure_km2))
        return 1.0 / de_dp if de_dp > 0.0 else math.nan


def merge_bps_core(
    bps_path: str | Path,
    core_n_fm3: np.ndarray,
    core_pressure_MeV_fm3: np.ndarray,
    core_energy_MeV_fm3: np.ndarray,
    *,
    match_density_fm3: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-density BPS/core join with strict P-epsilon monotonic cleanup."""
    bps = np.loadtxt(bps_path)
    bn, bp, be = bps[:, 0], bps[:, 1], bps[:, 2]
    cn = np.asarray(core_n_fm3, dtype=float)
    cp = np.asarray(core_pressure_MeV_fm3, dtype=float)
    ce = np.asarray(core_energy_MeV_fm3, dtype=float)
    crust = bn < match_density_fm3
    core = cn >= match_density_fm3
    p = np.r_[bp[crust], cp[core]]
    e = np.r_[be[crust], ce[core]]
    order = np.argsort(p)
    p, e = p[order], e[order]
    keep = np.zeros(len(p), dtype=bool)
    last_p = last_e = -math.inf
    for i, (pi, ei) in enumerate(zip(p, e)):
        if np.isfinite(pi) and np.isfinite(ei) and pi > last_p and ei > last_e:
            keep[i] = True
            last_p, last_e = float(pi), float(ei)
    return p[keep], e[keep]


@dataclass(frozen=True)
class StarLove:
    central_pressure_MeV_fm3: float
    mass_Msun: float
    radius_km: float
    compactness: float
    y_surface: float
    k2: float
    Lambda: float


def _rhs(_r: float, state: np.ndarray, eos: BarotropicEOS) -> np.ndarray:
    r = max(float(_r), 1e-12)
    m, p, y = map(float, state)
    if p <= 0.0 or r <= 2.0 * m:
        return np.zeros(3)
    e = eos.epsilon(p)
    cs2 = eos.cs2(p)
    if not (math.isfinite(e) and math.isfinite(cs2) and cs2 > 0.0):
        return np.array([math.nan, math.nan, math.nan])
    one_minus_2c = 1.0 - 2.0 * m / r
    a = m + 4.0 * math.pi * r**3 * p
    dm = 4.0 * math.pi * r**2 * e
    dp = -(e + p) * a / (r**2 * one_minus_2c)
    f = (1.0 - 4.0 * math.pi * r**2 * (e - p)) / one_minus_2c
    q = (
        4.0 * math.pi * (5.0 * e + 9.0 * p + (e + p) / cs2) / one_minus_2c
        - 6.0 / (r**2 * one_minus_2c)
        - 4.0 * a**2 / (r**4 * one_minus_2c**2)
    )
    dy = -(y * y + y * f + r * r * q) / r
    return np.array([dm, dp, dy])


def _surface(_r: float, state: np.ndarray, _eos: BarotropicEOS) -> float:
    return float(state[1])


_surface.terminal = True
_surface.direction = -1


def love_number_k2(compactness: float, y_surface: float) -> float:
    c = compactness
    y = y_surface
    if not (0.0 < c < 0.5):
        return math.nan
    z = 1.0 - 2.0 * c
    numerator = (8.0 / 5.0) * c**5 * z**2 * (2.0 + 2.0 * c * (y - 1.0) - y)
    denominator = (
        2.0 * c * (6.0 - 3.0 * y + 3.0 * c * (5.0 * y - 8.0))
        + 4.0 * c**3 * (13.0 - 11.0 * y + c * (3.0 * y - 2.0) + 2.0 * c**2 * (1.0 + y))
        + 3.0 * z**2 * (2.0 - y + 2.0 * c * (y - 1.0)) * math.log(z)
    )
    return numerator / denominator


def integrate_star(
    eos: BarotropicEOS,
    central_pressure_MeV_fm3: float,
    *,
    rtol: float = 2e-7,
    atol: float = 1e-10,
    max_step_km: float = 0.03,
) -> StarLove:
    pc = central_pressure_MeV_fm3 * MEVFM3_TO_KM2
    if not (eos.p[0] < pc < eos.p[-1]):
        raise ValueError("central pressure outside EOS table")
    r0 = 1e-5
    ec = eos.epsilon(pc)
    m0 = 4.0 * math.pi * ec * r0**3 / 3.0
    sol = solve_ivp(
        _rhs,
        (r0, 60.0),
        np.array([m0, pc, 2.0]),
        args=(eos,),
        events=_surface,
        rtol=rtol,
        atol=atol,
        max_step=max_step_km,
    )
    if not sol.success or not len(sol.t_events[0]):
        raise RuntimeError(f"TOV-Love integration failed: {sol.message}")
    radius = float(sol.t_events[0][0])
    mass_km, _, y_r = map(float, sol.y_events[0][0])
    c = mass_km / radius
    k2 = love_number_k2(c, y_r)
    lam = 2.0 * k2 / (3.0 * c**5)
    return StarLove(central_pressure_MeV_fm3, mass_km / MSUN_TO_KM, radius, c, y_r, k2, lam)
