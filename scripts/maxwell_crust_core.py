"""Maxwell construction between the bundled BPS crust and an RMF core EOS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _row(point) -> dict:
    if isinstance(point, dict):
        return dict(point)
    if hasattr(point, "to_dict"):
        return point.to_dict()
    return {
        "rho_b": point.rho_b,
        "pressure_total_mev_fm3": point.pressure_total_mev_fm3,
        "epsilon_total_mev_fm3": point.epsilon_total_mev_fm3,
    }


def _phase_rows(rows, *, bps: bool) -> list[dict]:
    out = []
    for point in rows:
        r = _row(point)
        n = float(r["rho_b"])
        pkey = "pressure_mev_fm3" if bps else "pressure_total_mev_fm3"
        ekey = "energy_density_mev_fm3" if bps else "epsilon_total_mev_fm3"
        p, e = float(r[pkey]), float(r[ekey])
        if n > 0.0 and p >= 0.0 and e > 0.0 and np.isfinite((n, p, e)).all():
            out.append({"rho_b": n, "pressure": p, "epsilon": e, "mu_b": (e + p) / n, "extra": r})
    return sorted(out, key=lambda x: x["rho_b"])


def _monotone_branch(rows: list[dict]) -> list[dict]:
    """Keep the increasing thermodynamic envelope in density order."""
    kept = []
    for r in rows:
        if not kept or (r["mu_b"] > kept[-1]["mu_b"] + 1e-10 and r["pressure"] > kept[-1]["pressure"] + 1e-12):
            kept.append(r)
    return kept


def _interp(branch: list[dict], key: str, mu: float) -> float:
    mus = np.asarray([r["mu_b"] for r in branch])
    vals = np.asarray([r[key] for r in branch])
    return float(np.interp(mu, mus, vals))


class MaxwellEOSInterpolator:
    """Piecewise epsilon(P), with an explicit density jump at P_transition."""

    def __init__(self, low_table, high_table, transition_pressure_mev_fm3, eos_interpolator_cls, conversion):
        self.low = eos_interpolator_cls(
            [r["pressure_total_mev_fm3"] for r in low_table],
            [r["epsilon_total_mev_fm3"] for r in low_table],
        )
        self.high = eos_interpolator_cls(
            [r["pressure_total_mev_fm3"] for r in high_table],
            [r["epsilon_total_mev_fm3"] for r in high_table],
        )
        self.transition_pressure_mev_fm3 = float(transition_pressure_mev_fm3)
        self.transition_pressure_km2 = self.transition_pressure_mev_fm3 * conversion

    def energy_density_from_pressure_km2(self, p_km2: float) -> float:
        # At exactly P_t choose the high-density phase, appropriate when the
        # integration approaches the interface from the stellar centre.
        if p_km2 < self.transition_pressure_km2:
            return self.low.energy_density_from_pressure_km2(p_km2)
        return self.high.energy_density_from_pressure_km2(p_km2)


@dataclass
class MaxwellResult:
    eos: object
    low_table: list[dict]
    high_table: list[dict]
    merged_table: list[dict]
    active_core_rows: list[dict]
    metadata: dict


def build_maxwell_crust_core_eos(
    bps_rows,
    core_rows,
    *,
    eos_interpolator_cls,
    mevfm3_to_km2: float,
    crust_search=(1.0e-5, 0.12),
    core_search=(3.0e-4, 0.20),
) -> MaxwellResult:
    """Find P_BPS(mu)=P_core(mu) and build a discontinuous Maxwell EOS."""
    bps_all = _phase_rows(bps_rows, bps=True)
    core_all = _phase_rows(core_rows, bps=False)
    bps_match = _monotone_branch([r for r in bps_all if crust_search[0] <= r["rho_b"] <= crust_search[1]])
    core_match = _monotone_branch([r for r in core_all if core_search[0] <= r["rho_b"] <= core_search[1]])
    if len(bps_match) < 3 or len(core_match) < 3:
        raise RuntimeError("insufficient points on a Maxwell matching branch")

    mu_lo = max(bps_match[0]["mu_b"], core_match[0]["mu_b"])
    mu_hi = min(bps_match[-1]["mu_b"], core_match[-1]["mu_b"])
    if not mu_hi > mu_lo:
        raise RuntimeError("BPS and core chemical-potential ranges do not overlap")

    knots = [mu_lo, mu_hi]
    knots += [r["mu_b"] for r in bps_match if mu_lo < r["mu_b"] < mu_hi]
    knots += [r["mu_b"] for r in core_match if mu_lo < r["mu_b"] < mu_hi]
    knots += np.linspace(mu_lo, mu_hi, 1201).tolist()
    mus = np.asarray(sorted(set(knots)))

    def delta(mu):
        return _interp(core_match, "pressure", mu) - _interp(bps_match, "pressure", mu)

    ds = np.asarray([delta(mu) for mu in mus])
    candidates = []
    for i in range(len(mus) - 1):
        # Stable BPS below the transition and stable uniform core above it.
        if ds[i] <= 0.0 and ds[i + 1] >= 0.0 and (ds[i] < 0.0 or ds[i + 1] > 0.0):
            lo, hi = float(mus[i]), float(mus[i + 1])
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                if delta(mid) >= 0.0:
                    hi = mid
                else:
                    lo = mid
            mu = 0.5 * (lo + hi)
            pc = 0.5 * (_interp(core_match, "pressure", mu) + _interp(bps_match, "pressure", mu))
            nc = _interp(core_match, "rho_b", mu)
            nb = _interp(bps_match, "rho_b", mu)
            eb, ec = mu * nb - pc, mu * nc - pc
            if nc > nb and ec > eb and pc > 0.0:
                candidates.append((pc, mu, nb, nc, eb, ec))
    if not candidates:
        raise RuntimeError("no physical BPS-to-core Maxwell crossing")

    pc, mu, nb, nc, eb, ec = max(candidates, key=lambda x: x[0])
    low = []
    for r in bps_all:
        if r["mu_b"] < mu and r["pressure"] < pc:
            low.append({"segment": "bps", "rho_b": r["rho_b"], "pressure_total_mev_fm3": r["pressure"], "epsilon_total_mev_fm3": r["epsilon"]})
    high = []
    active = []
    for r in core_all:
        if r["mu_b"] > mu and r["pressure"] > pc and r["rho_b"] > nc:
            row = {"segment": "core", **r["extra"]}
            row["rho_b"] = r["rho_b"]
            row["pressure_total_mev_fm3"] = r["pressure"]
            row["epsilon_total_mev_fm3"] = r["epsilon"]
            high.append(row)
            active.append(dict(row))
    low.sort(key=lambda x: x["pressure_total_mev_fm3"])
    high.sort(key=lambda x: x["pressure_total_mev_fm3"])
    low.append({"segment": "bps_transition", "rho_b": nb, "pressure_total_mev_fm3": pc, "epsilon_total_mev_fm3": eb})
    high.insert(0, {"segment": "core_transition", "rho_b": nc, "pressure_total_mev_fm3": pc, "epsilon_total_mev_fm3": ec})
    if len(low) < 2 or len(high) < 2 or len(active) < 3:
        raise RuntimeError("Maxwell branches are too short for TOV interpolation")

    eos = MaxwellEOSInterpolator(low, high, pc, eos_interpolator_cls, mevfm3_to_km2)
    metadata = {
        "method": "Maxwell: equal baryon chemical potential and pressure",
        "mu_transition_MeV": mu,
        "pressure_transition_MeV_fm3": pc,
        "rho_crust_fm3": nb,
        "rho_core_fm3": nc,
        "density_jump_fm3": nc - nb,
        "epsilon_crust_MeV_fm3": eb,
        "epsilon_core_MeV_fm3": ec,
        "energy_density_jump_MeV_fm3": ec - eb,
        "relative_density_jump": nc / nb - 1.0,
    }
    merged = low + high
    return MaxwellResult(eos=eos, low_table=low, high_table=high, merged_table=merged, active_core_rows=active, metadata=metadata)


def read_bps(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cols = s.split()
        if len(cols) >= 3:
            rows.append({"rho_b": float(cols[0]), "pressure_mev_fm3": float(cols[1]), "energy_density_mev_fm3": float(cols[2])})
    return rows
