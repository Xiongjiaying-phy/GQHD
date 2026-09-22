#!/usr/bin/env python3
"""Search for a thermodynamically stable parameter set near GQHD1.

Screening objective:
  * positive npe curvature on the beta-equilibrium trajectory;
  * positive symmetry energy;
  * preserve GQHD1 J, L and the high-density beta-equilibrium EOS;
  * minimize normalized parameter displacement.

Final M-R validation is intentionally kept separate from this fast screening.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import signal
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, differential_evolution


ROOT = Path(__file__).resolve().parents[1]
BASE_MODULE = ROOT / "tmp" / "rmf_verified_convention_properties.py"
spec = importlib.util.spec_from_file_location("gqhd_verified", BASE_MODULE)
rmf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rmf
assert spec.loader is not None
spec.loader.exec_module(rmf)

BASE = next(p for p in rmf.PARAMS if p.name == "GQHD1")
M_E = 0.51099895

# Scale factors multiply the corresponding GQHD1 parameters.
NAMES = ("g_rho", "g_a0", "g5", "g6", "b5", "b6", "b7", "b8", "b9")
BOUNDS = (
    (0.90, 1.10),
    (0.90, 1.10),
    (0.85, 1.15),
    (0.85, 1.15),
    (0.80, 1.20),
    (0.80, 1.20),
    (0.80, 1.20),
    (0.74, 0.94),
    (0.80, 1.20),
)

STABILITY_RATIOS = np.array(
    [0.55, 0.75, 0.90, 1.05, 1.22, 1.38, 1.52, 1.65, 1.76,
     1.85, 2.0, 2.3, 3.0, 4.0, 5.0]
)
SYMMETRY_RATIOS = np.linspace(0.7, 5.0, 24)
EOS_RATIOS = np.array([2.10, 2.40, 3.0, 4.0, 5.0, 6.0])


def scaled_parameter_set(scales: np.ndarray):
    updates = {name: getattr(BASE, name) * float(scale) for name, scale in zip(NAMES, scales)}
    return replace(BASE, name="GQHD1-stable-search", **updates)


def chemical_potentials(nn: float, np_: float, p) -> np.ndarray:
    fields = rmf.solve_fields(nn, np_, p)
    rho3, omega, _sigma, _a3 = fields
    mp, mn, _rsp, _rsn, kp, kn = rmf.masses_and_densities(nn, np_, fields, p)
    return np.array(
        [
            math.sqrt(kn * kn + mn * mn) + p.g_omega * omega - p.g_rho * rho3,
            math.sqrt(kp * kp + mp * mp) + p.g_omega * omega + p.g_rho * rho3,
        ]
    )


def electron_state(ne: float) -> tuple[float, float, float]:
    k = rmf.kf(ne)
    mu = math.sqrt(k * k + M_E * M_E)
    eps = rmf.eps_one(k, M_E) / rmf.HBARC**3
    curvature = k * k / (3.0 * ne * mu) if ne > 0.0 else math.inf
    return mu, eps, curvature


def beta_fraction(n: float, p) -> float:
    def residual(y: float) -> float:
        mu = chemical_potentials(n * (1.0 - y), n * y, p)
        mu_e, _eps_e, _ce = electron_state(n * y)
        return float(mu[0] - mu[1] - mu_e)

    lo, hi = 1.0e-12, 0.499999
    flo, fhi = residual(lo), residual(hi)
    if not np.isfinite(flo + fhi) or flo * fhi > 0.0:
        raise RuntimeError("beta-equilibrium root is not bracketed")
    return float(brentq(residual, lo, hi, xtol=2e-13, rtol=2e-13, maxiter=100))


def nuclear_curvature(n: float, y: float, p) -> np.ndarray:
    nn, np_ = n * (1.0 - y), n * y
    h = max(min(1.0e-5, 1.0e-3 * np_), 1.0e-12)
    columns = []
    for dn, dp in ((h, 0.0), (0.0, h)):
        plus = chemical_potentials(nn + dn, np_ + dp, p)
        minus = chemical_potentials(nn - dn, np_ - dp, p)
        columns.append((plus - minus) / (2.0 * h))
    raw = np.column_stack(columns)
    return 0.5 * (raw + raw.T)


def beta_point(n: float, p, with_curvature: bool = False) -> dict[str, float]:
    y = beta_fraction(n, p)
    nn, np_ = n * (1.0 - y), n * y
    mu = chemical_potentials(nn, np_, p)
    mu_e, eps_e, ce = electron_state(np_)
    eps_b, _fields = rmf.epsilon(nn, np_, p)
    pressure = nn * mu[0] + np_ * mu[1] + np_ * mu_e - eps_b - eps_e
    out = {
        "n_fm3": n,
        "Yp": y,
        "epsilon": eps_b + eps_e,
        "pressure": pressure,
        "mu_e": mu_e,
    }
    if with_curvature:
        c = nuclear_curvature(n, y, p)
        c[1, 1] += ce
        eig = np.linalg.eigvalsh(c)
        out["lambda_min"] = float(eig[0])
        out["lambda_max"] = float(eig[1])
    return out


def scalar_polarization(kf: float, mass: float) -> float:
    h = 1.0e-3 * mass
    return 2.0 * (rmf.rho_s_one(kf, mass + h) - rmf.rho_s_one(kf, mass - h)) / (2.0 * h)


def symmetry_energy(n: float, p) -> float:
    _rho, omega, sigma, _a = rmf.solve_fields(n / 2.0, n / 2.0, p)
    nnat = rmf.rho_nat(n)
    kf = rmf.kf(n / 2.0)
    mstar = p.M_N + p.g_sigma * sigma
    ef = math.sqrt(kf * kf + mstar * mstar)
    skin = kf * kf / (6.0 * ef)
    pol = scalar_polarization(kf, mstar)
    mix = p.b8 * omega * sigma + p.b9 * omega
    hmat = np.array(
        [
            [-(p.m_rho**2 + p.c4 * omega**2 - p.g5 * sigma**2 - 2.0 * p.g6 * sigma), -mix],
            [-mix, p.m_a0**2 - p.b6 * sigma**2 - p.b7 * omega**2 - p.b5 * sigma + p.g_a0**2 * pol],
        ]
    )
    source = np.array([-p.g_rho * nnat, -p.g_a0 * nnat * mstar / ef])
    return float(skin - 0.5 * source @ np.linalg.solve(hmat, source) / nnat)


def symmetry_slope(n0: float, p) -> float:
    h = 1.0e-3
    return 3.0 * n0 * (symmetry_energy(n0 + h, p) - symmetry_energy(n0 - h, p)) / (2.0 * h)


def build_targets() -> dict:
    n0 = rmf.golden(BASE)
    return {
        "n0": n0,
        "J": symmetry_energy(n0, BASE),
        "L": symmetry_slope(n0, BASE),
        "eos": {str(r): beta_point(float(r * n0), BASE) for r in EOS_RATIOS},
    }


TARGET = build_targets()


def evaluate(scales: np.ndarray, detailed: bool = False) -> dict:
    p = scaled_parameter_set(scales)
    n0 = TARGET["n0"]  # selected parameters vanish in symmetric matter at delta=0
    result: dict = {"valid": False, "scales": dict(zip(NAMES, map(float, scales)))}
    try:
        j = symmetry_energy(n0, p)
        ell = symmetry_slope(n0, p)
        svals = np.array([symmetry_energy(float(r * n0), p) for r in SYMMETRY_RATIOS])
        stability = [beta_point(float(r * n0), p, with_curvature=True) for r in STABILITY_RATIOS]
        eos = [beta_point(float(r * n0), p) for r in EOS_RATIOS]
    except Exception as exc:
        result["error"] = str(exc)
        result["objective"] = 1.0e12
        return result

    min_lambda = min(x["lambda_min"] for x in stability)
    min_s = float(np.min(svals))
    # Preserve J and L tightly, and high-density P(epsilon) closely enough that
    # final M-R validation is meaningful.
    fit = ((j - TARGET["J"]) / 0.6) ** 2 + ((ell - TARGET["L"]) / 2.5) ** 2
    eos_terms = []
    for ratio, point in zip(EOS_RATIOS, eos):
        target = TARGET["eos"][str(float(ratio))]
        pscale = max(2.0, 0.04 * abs(target["pressure"]))
        escale = max(5.0, 0.015 * abs(target["epsilon"]))
        eos_terms.append(((point["pressure"] - target["pressure"]) / pscale) ** 2)
        eos_terms.append(((point["epsilon"] - target["epsilon"]) / escale) ** 2)
    fit += float(np.mean(eos_terms))

    widths = np.array([(hi - lo) / 2.0 for lo, hi in BOUNDS])
    distance = float(np.mean(((np.asarray(scales) - 1.0) / widths) ** 2))
    # Smooth hard-constraint surrogates.  Finalists are accepted only after a
    # denser validation scan, irrespective of this screening objective.
    stability_penalty = 200.0 * max(0.0, (10.0 - min_lambda) / 50.0) ** 2
    symmetry_penalty = 200.0 * max(0.0, (1.0 - min_s) / 5.0) ** 2
    objective = fit + 0.15 * distance + stability_penalty + symmetry_penalty
    result.update(
        {
            "valid": True,
            "objective": float(objective),
            "fit_penalty": float(fit),
            "distance_penalty": distance,
            "stability_penalty": float(stability_penalty),
            "symmetry_penalty": float(symmetry_penalty),
            "J_MeV": float(j),
            "L_MeV": float(ell),
            "min_Esym_MeV": min_s,
            "min_lambda_MeV_fm3": float(min_lambda),
            "parameters": asdict(p),
        }
    )
    if detailed:
        result["stability_points"] = stability
        result["eos_points"] = eos
        result["Esym_points"] = [
            {"n_over_n0": float(r), "Esym_MeV": float(s)}
            for r, s in zip(SYMMETRY_RATIOS, svals)
        ]
    return result


def objective(scales: np.ndarray) -> float:
    def timeout_handler(_signum, _frame):
        raise TimeoutError("candidate evaluation timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    try:
        return evaluate(scales)["objective"]
    except TimeoutError:
        return 1.0e12
    finally:
        signal.alarm(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "output" / "gqhd1_stable_search")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--evaluate-base", action="store_true")
    ap.add_argument("--no-polish", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.evaluate_base:
        report = evaluate(np.ones(len(NAMES)), detailed=True)
    else:
        updating = "immediate" if args.workers == 1 else "deferred"
        result = differential_evolution(
            objective,
            BOUNDS,
            seed=args.seed,
            popsize=args.popsize,
            maxiter=args.maxiter,
            workers=args.workers,
            updating=updating,
            polish=not args.no_polish,
            tol=2e-4,
            atol=1e-4,
            disp=True,
        )
        report = evaluate(result.x, detailed=True)
        report["optimizer"] = {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
        }
    path = args.output / ("base_report.json" if args.evaluate_base else f"candidate_seed_{args.seed}.json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
