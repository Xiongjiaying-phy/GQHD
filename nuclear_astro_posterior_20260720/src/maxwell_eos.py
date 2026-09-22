"""Thermodynamically consistent Maxwell matching of a core EOS to BPS.

The transition is found from equality of pressure at common baryon chemical
potential.  The two energy densities at the transition pressure are retained
explicitly; they are never merged by sorting or monotonic point deletion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .tov_love import (
    BarotropicEOS,
    MEVFM3_TO_KM2,
    MSUN_TO_KM,
    StarLove,
    _rhs,
    _surface,
    love_number_k2,
)


@dataclass(frozen=True)
class MaxwellTransition:
    n_minus_fm3: float
    n_plus_fm3: float
    pressure_MeV_fm3: float
    mu_B_MeV: float
    epsilon_minus_MeV_fm3: float
    epsilon_plus_MeV_fm3: float
    delta_n_fm3: float
    delta_epsilon_MeV_fm3: float
    pressure_residual_MeV_fm3: float
    chemical_potential_residual_MeV: float
    latent_heat_residual_MeV_fm3: float


def _strictly_increasing_indices(values: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(values), dtype=bool)
    last = -math.inf
    for index, value in enumerate(values):
        if np.isfinite(value) and value > last:
            keep[index] = True
            last = float(value)
    return keep


def _high_density_stable_branch(
    n: np.ndarray,
    pressure: np.ndarray,
    energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(n) & np.isfinite(pressure) & np.isfinite(energy)
    local = (
        finite
        & (n > 0.0)
        & (pressure > 0.0)
        & (energy > 0.0)
    )
    if not local[-1]:
        raise ValueError("uniform-matter branch is not physical at its high-density endpoint")
    start = len(local) - 1
    while start > 0:
        candidate = start - 1
        if not local[candidate]:
            break
        if not (
            pressure[candidate] < pressure[start]
            and energy[candidate] < energy[start]
        ):
            break
        start = candidate
    n, pressure, energy = n[start:], pressure[start:], energy[start:]
    mu = (energy + pressure) / n
    keep = _strictly_increasing_indices(mu)
    n, pressure, energy, mu = n[keep], pressure[keep], energy[keep], mu[keep]
    if len(n) < 8:
        raise ValueError("too few points on the stable high-density core branch")
    if not (np.all(np.diff(pressure) > 0.0) and np.all(np.diff(energy) > 0.0)):
        raise ValueError("core branch is not monotonic after chemical-potential filtering")
    return n, pressure, energy


def find_maxwell_transition(
    bps_n_fm3: np.ndarray,
    bps_pressure_MeV_fm3: np.ndarray,
    bps_energy_MeV_fm3: np.ndarray,
    core_n_fm3: np.ndarray,
    core_pressure_MeV_fm3: np.ndarray,
    core_energy_MeV_fm3: np.ndarray,
    *,
    bps_search_min_fm3: float = 0.01,
    bps_search_max_fm3: float = 0.08,
    root_scan_points: int = 2049,
) -> tuple[MaxwellTransition, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Find the BPS-to-core phase switch on the stable P(mu_B) envelope."""
    bn = np.asarray(bps_n_fm3, dtype=float)
    bp = np.asarray(bps_pressure_MeV_fm3, dtype=float)
    be = np.asarray(bps_energy_MeV_fm3, dtype=float)
    cn = np.asarray(core_n_fm3, dtype=float)
    cp = np.asarray(core_pressure_MeV_fm3, dtype=float)
    ce = np.asarray(core_energy_MeV_fm3, dtype=float)
    if not (len(bn) == len(bp) == len(be) and len(cn) == len(cp) == len(ce)):
        raise ValueError("inconsistent EOS-array lengths")

    crust_search = (
        np.isfinite(bn) & np.isfinite(bp) & np.isfinite(be)
        & (bn >= bps_search_min_fm3) & (bn <= bps_search_max_fm3)
        & (bp > 0.0) & (be > 0.0)
    )
    bn_s, bp_s, be_s = bn[crust_search], bp[crust_search], be[crust_search]
    bmu_s = (be_s + bp_s) / bn_s
    keep = _strictly_increasing_indices(bmu_s)
    bn_s, bp_s, be_s, bmu_s = bn_s[keep], bp_s[keep], be_s[keep], bmu_s[keep]
    if len(bn_s) < 5:
        raise ValueError("BPS search interval lacks a monotonic chemical-potential branch")

    cn_s, cp_s, ce_s = _high_density_stable_branch(cn, cp, ce)
    cmu_s = (ce_s + cp_s) / cn_s
    mu_low = max(float(bmu_s[0]), float(cmu_s[0]))
    mu_high = min(float(bmu_s[-1]), float(cmu_s[-1]))
    if not mu_low < mu_high:
        raise ValueError("BPS and core branches do not overlap in baryon chemical potential")

    bp_of_mu = PchipInterpolator(bmu_s, bp_s, extrapolate=False)
    bn_of_mu = PchipInterpolator(bmu_s, bn_s, extrapolate=False)
    cp_of_mu = PchipInterpolator(cmu_s, cp_s, extrapolate=False)
    cn_of_mu = PchipInterpolator(cmu_s, cn_s, extrapolate=False)

    def pressure_difference(mu: float) -> float:
        return float(cp_of_mu(mu) - bp_of_mu(mu))

    grid = np.linspace(mu_low, mu_high, root_scan_points)
    difference = np.asarray([pressure_difference(mu) for mu in grid])
    candidates: list[float] = []
    for index in range(len(grid) - 1):
        left, right = difference[index], difference[index + 1]
        if left <= 0.0 and right >= 0.0 and (left < 0.0 or right > 0.0):
            candidates.append(brentq(pressure_difference, grid[index], grid[index + 1]))
    if not candidates:
        raise ValueError("no stable BPS-to-core Maxwell crossing found")

    selected: MaxwellTransition | None = None
    for mu_t in candidates:
        below = difference[grid < mu_t]
        above = difference[grid > mu_t]
        scale = max(float(np.max(np.abs(difference))), 1.0)
        tolerance = 1.0e-10 * scale
        if (
            (len(below) and np.any(below > tolerance))
            or (len(above) and np.any(above < -tolerance))
        ):
            continue
        n_minus = float(bn_of_mu(mu_t))
        n_plus = float(cn_of_mu(mu_t))
        p_minus = float(bp_of_mu(mu_t))
        p_plus = float(cp_of_mu(mu_t))
        # Construct the interface endpoints from the exact zero-temperature
        # identity epsilon = mu_B n - P.  Independent interpolation of all
        # three quantities would introduce a small artificial mismatch.
        e_minus = mu_t * n_minus - p_minus
        e_plus = mu_t * n_plus - p_plus
        delta_n = n_plus - n_minus
        delta_e = e_plus - e_minus
        if not (delta_n > 0.0 and delta_e > 0.0):
            continue
        latent_residual = delta_e - mu_t * delta_n
        selected = MaxwellTransition(
            n_minus,
            n_plus,
            0.5 * (p_minus + p_plus),
            mu_t,
            e_minus,
            e_plus,
            delta_n,
            delta_e,
            p_plus - p_minus,
            ((e_plus + p_plus) / n_plus) - ((e_minus + p_minus) / n_minus),
            latent_residual,
        )
        break
    if selected is None:
        raise ValueError("crossings exist but none has the physical density ordering")
    return selected, (cn_s, cp_s, ce_s)


class MaxwellEOS:
    """Piecewise barotrope retaining a sharp energy-density discontinuity."""

    def __init__(
        self,
        bps_path: str | Path | np.ndarray,
        core_n_fm3: np.ndarray,
        core_pressure_MeV_fm3: np.ndarray,
        core_energy_MeV_fm3: np.ndarray,
        **transition_options: float | int,
    ):
        bps = (
            np.asarray(bps_path, dtype=float)
            if isinstance(bps_path, np.ndarray)
            else np.loadtxt(bps_path)
        )
        if bps.ndim != 2 or bps.shape[1] < 3:
            raise ValueError("BPS input must have columns n, P, and epsilon")
        transition, stable_core = find_maxwell_transition(
            bps[:, 0], bps[:, 1], bps[:, 2],
            core_n_fm3, core_pressure_MeV_fm3, core_energy_MeV_fm3,
            **transition_options,
        )
        self.transition = transition
        cn, cp, ce = stable_core
        crust = bps[:, 1] < transition.pressure_MeV_fm3
        core = cp > transition.pressure_MeV_fm3
        crust_p = np.r_[bps[crust, 1], transition.pressure_MeV_fm3]
        crust_e = np.r_[bps[crust, 2], transition.epsilon_minus_MeV_fm3]
        core_p = np.r_[transition.pressure_MeV_fm3, cp[core]]
        core_e = np.r_[transition.epsilon_plus_MeV_fm3, ce[core]]
        self.crust = BarotropicEOS(crust_p, crust_e)
        self.core = BarotropicEOS(core_p, core_e)
        self.p_transition = transition.pressure_MeV_fm3 * MEVFM3_TO_KM2
        self.p_mev = np.r_[self.crust.p_mev, self.core.p_mev[1:]]
        self.p = np.r_[self.crust.p, self.core.p[1:]]

    def branch(self, pressure_km2: float) -> BarotropicEOS:
        return self.core if pressure_km2 > self.p_transition else self.crust

    def epsilon(self, pressure_km2: float) -> float:
        return self.branch(pressure_km2).epsilon(pressure_km2)

    def cs2(self, pressure_km2: float) -> float:
        return self.branch(pressure_km2).cs2(pressure_km2)


def love_y_jump(
    radius_km: float,
    mass_km: float,
    pressure_km2: float,
    delta_energy_km2: float,
) -> float:
    """Outward jump y(r+)-y(r-) at a finite-pressure density discontinuity."""
    denominator = mass_km + 4.0 * math.pi * radius_km**3 * pressure_km2
    if not denominator > 0.0:
        raise ValueError("invalid interface denominator in tidal jump")
    return -4.0 * math.pi * radius_km**3 * delta_energy_km2 / denominator


def _transition_event(transition_pressure_km2: float):
    def event(_r: float, state: np.ndarray, _eos: BarotropicEOS) -> float:
        return float(state[1] - transition_pressure_km2)
    event.terminal = True
    event.direction = -1
    return event


def integrate_star_maxwell(
    eos: MaxwellEOS,
    central_pressure_MeV_fm3: float,
    *,
    rtol: float = 2e-7,
    atol: float = 1e-10,
    max_step_km: float = 0.03,
) -> StarLove:
    pc = central_pressure_MeV_fm3 * MEVFM3_TO_KM2
    if not (eos.p[0] < pc < eos.p[-1]):
        raise ValueError("central pressure outside EOS table")
    r0 = 1.0e-5
    active = eos.branch(pc)
    ec = active.epsilon(pc)
    state0 = np.array([4.0 * math.pi * ec * r0**3 / 3.0, pc, 2.0])

    if pc > eos.p_transition:
        interface_event = _transition_event(eos.p_transition)
        inner = solve_ivp(
            _rhs, (r0, 60.0), state0, args=(eos.core,),
            events=interface_event, rtol=rtol, atol=atol, max_step=max_step_km,
        )
        if not inner.success or not len(inner.t_events[0]):
            raise RuntimeError(f"core integration failed: {inner.message}")
        radius_i = float(inner.t_events[0][0])
        mass_i, pressure_i, y_i = map(float, inner.y_events[0][0])
        delta_energy = eos.transition.delta_epsilon_MeV_fm3 * MEVFM3_TO_KM2
        y_i += love_y_jump(radius_i, mass_i, pressure_i, delta_energy)
        state0 = np.array([mass_i, pressure_i, y_i])
        r0 = radius_i

    outer = solve_ivp(
        _rhs, (r0, 60.0), state0, args=(eos.crust,),
        events=_surface, rtol=rtol, atol=atol, max_step=max_step_km,
    )
    if not outer.success or not len(outer.t_events[0]):
        raise RuntimeError(f"crust integration failed: {outer.message}")
    radius = float(outer.t_events[0][0])
    mass_km, _, y_r = map(float, outer.y_events[0][0])
    compactness = mass_km / radius
    k2 = love_number_k2(compactness, y_r)
    tidal = 2.0 * k2 / (3.0 * compactness**5)
    return StarLove(
        central_pressure_MeV_fm3,
        mass_km / MSUN_TO_KM,
        radius,
        compactness,
        y_r,
        k2,
        tidal,
    )
