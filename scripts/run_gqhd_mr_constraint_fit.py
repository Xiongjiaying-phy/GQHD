#!/usr/bin/env python3
"""Random-search plus projected SPSA fitting of GQHD M-R constraints.

The fast objective uses the same beta-equilibrium RMF convention as the
GQHD1 stability search, attaches the bundled BPS crust at 0.08 fm^-3, and
solves TOV.  The winning point is recomputed on dense stability/EOS/M-R grids.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = ROOT / "scripts" / "search_gqhd1_stable_neighbor.py"
spec = importlib.util.spec_from_file_location("stable_search_general", SEARCH_PATH)
search = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = search
assert spec.loader is not None
spec.loader.exec_module(search)

TOOLKIT = Path(os.environ.get("RMF_TOOLKIT", str(ROOT / "vendor" / "rmf")))
sys.path.insert(0, str(TOOLKIT / "TOVtools"))
from hybrid_eos import build_hybrid_eos_with_bundled_crust  # noqa: E402
from tov_solver import build_eos_interpolator, build_mr_curve  # noqa: E402


BASES = {p.name: p for p in search.rmf.PARAMS if p.name in {"GQHD1", "GQHD2"}}
NAMES = search.NAMES

# Median and equal-tailed 68% marginal intervals.  The likelihood below
# profiles over mass along the calculated stable M-R branch and uses split
# normal errors.  This is documented as an approximation to the full 2-D
# posterior, which is not bundled with this project.
OBSERVATIONS = {
    "6620": {
        "label": "PSR J0740+6620",
        "mass": 2.073, "mass_minus": 0.069, "mass_plus": 0.069,
        "radius": 12.49, "radius_minus": 0.88, "radius_plus": 1.28,
        "source": "Salmi et al. 2024, arXiv:2406.14466",
    },
    "4715": {
        "label": "PSR J0437-4715",
        "mass": 1.418, "mass_minus": 0.037, "mass_plus": 0.037,
        "radius": 11.36, "radius_minus": 0.63, "radius_plus": 0.95,
        "source": "Choudhury et al. 2024, arXiv:2407.06789",
    },
    "3329": {
        "label": "PSR J0614-3329",
        "mass": 1.44, "mass_minus": 0.07, "mass_plus": 0.06,
        "radius": 10.29, "radius_minus": 0.86, "radius_plus": 1.01,
        "source": "Mauviard et al. 2025, arXiv:2506.14883",
    },
    "347": {
        "label": "HESS J1731-347",
        "mass": 0.77, "mass_minus": 0.17, "mass_plus": 0.20,
        "radius": 10.40, "radius_minus": 0.78, "radius_plus": 0.86,
        "source": "Doroshenko et al. measurement as quoted in arXiv:2306.12326",
    },
}


def split_sigma(value: np.ndarray | float, center: float, minus: float, plus: float):
    arr = np.asarray(value)
    return np.where(arr < center, minus, plus)


def parameter_set(anchor: str, scales: np.ndarray):
    base = BASES[anchor]
    updates = {name: getattr(base, name) * float(scale) for name, scale in zip(NAMES, scales)}
    return replace(base, name=f"{anchor}-MR-fit", **updates)


def stable_mr_arrays(mr):
    masses = np.asarray(mr.masses_msun, dtype=float)
    radii = np.asarray(mr.radii_km, dtype=float)
    pcs = np.asarray(mr.central_pressures_mev_fm3, dtype=float)
    good = np.isfinite(masses) & np.isfinite(radii) & np.isfinite(pcs)
    masses, radii, pcs = masses[good], radii[good], pcs[good]
    imax = int(np.argmax(masses))
    return masses[: imax + 1], radii[: imax + 1], pcs[: imax + 1]


def radius_at_mass(masses: np.ndarray, radii: np.ndarray, target: float):
    order = np.argsort(masses)
    m, r = masses[order], radii[order]
    if target < m[0] or target > m[-1]:
        return None
    return float(np.interp(target, m, r))


def observation_term(masses: np.ndarray, radii: np.ndarray, obs: dict) -> dict:
    order = np.argsort(masses)
    m, r = masses[order], radii[order]
    lo = max(float(m[0]), obs["mass"] - 4.0 * obs["mass_minus"])
    hi = min(float(m[-1]), obs["mass"] + 4.0 * obs["mass_plus"])
    if hi <= lo:
        return {"chi2": 1.0e5, "profile_mass": None, "profile_radius": None}
    grid = np.linspace(lo, hi, 181)
    rr = np.interp(grid, m, r)
    sm = split_sigma(grid, obs["mass"], obs["mass_minus"], obs["mass_plus"])
    sr = split_sigma(rr, obs["radius"], obs["radius_minus"], obs["radius_plus"])
    chi = ((grid - obs["mass"]) / sm) ** 2 + ((rr - obs["radius"]) / sr) ** 2
    idx = int(np.argmin(chi))
    return {
        "chi2": float(chi[idx]),
        "profile_mass": float(grid[idx]),
        "profile_radius": float(rr[idx]),
        "radius_at_median_mass": radius_at_mass(m, r, obs["mass"]),
    }


def make_core(p, points: int):
    densities = np.linspace(0.08, 1.25, points)
    rows = []
    for density in densities:
        q = search.beta_point(float(density), p)
        rows.append({
            "rho_b": float(density),
            "Yp": q["Yp"],
            "pressure_total_mev_fm3": q["pressure"],
            "epsilon_total_mev_fm3": q["epsilon"],
        })
    return rows


def invalid_report(anchor: str, scales: np.ndarray, message: str):
    return {
        "loss": 1.0e12,
        "valid_numerics": False,
        "anchor": anchor,
        "scales_vector": np.asarray(scales, dtype=float).tolist(),
        "error": message,
    }


def evaluate_candidate(payload, final: bool = False):
    anchor, scales_list, constraint_ids = payload
    scales = np.asarray(scales_list, dtype=float)
    p = parameter_set(anchor, scales)
    try:
        n0 = float(search.rmf.golden(BASES[anchor]))
        stability_ratios = np.linspace(0.5, min(6.0, 1.25 / n0), 260 if final else 23)
        stability = []
        for ratio in stability_ratios:
            q = search.beta_point(float(ratio * n0), p, with_curvature=True)
            q["n_over_n0"] = float(ratio)
            q["Esym_MeV"] = search.symmetry_energy(float(ratio * n0), p)
            stability.append(q)

        core = make_core(p, 520 if final else 112)
        eps = np.asarray([x["epsilon_total_mev_fm3"] for x in core])
        prs = np.asarray([x["pressure_total_mev_fm3"] for x in core])
        deps = np.diff(eps)
        dprs = np.diff(prs)
        cs2 = dprs / deps
        min_dp = float(np.min(dprs))
        min_cs2 = float(np.min(cs2))
        max_cs2 = float(np.max(cs2))

        hybrid = build_hybrid_eos_with_bundled_crust(core, rho_min_core=0.08)
        eos = build_eos_interpolator(hybrid.merged_table)
        positive = prs[prs > 0.0]
        if len(positive) < 3:
            return invalid_report(anchor, scales, "insufficient positive core pressure")
        pcs = np.geomspace(max(float(positive.min()) * 4.0, 0.025), float(positive.max()) * 0.985, 320 if final else 82)
        mr = build_mr_curve(eos, pcs.tolist(), dr_km=0.002 if final else 0.018, progress=False)
        masses, radii, central_pressures = stable_mr_arrays(mr)
        if len(masses) < 8 or float(np.max(masses)) < 0.5:
            return invalid_report(anchor, scales, "invalid stable M-R branch")

        obs_terms = {cid: observation_term(masses, radii, OBSERVATIONS[cid]) for cid in constraint_ids}
        obs_chi2 = float(sum(v["chi2"] for v in obs_terms.values()))
        min_lambda = float(min(q["lambda_min"] for q in stability))
        min_esym = float(min(q["Esym_MeV"] for q in stability))
        mmax = float(np.max(masses))
        j = float(search.symmetry_energy(n0, p))
        ell = float(search.symmetry_slope(n0, p))

        # Strong smooth barriers enforce thermodynamic stability, monotone
        # pressure, and causality.  The small displacement term selects the
        # nearest solution when observational fits are comparable.
        stability_penalty = 2500.0 * max(0.0, -min_lambda / 20.0) ** 2 + 15.0 * max(0.0, (5.0 - min_lambda) / 5.0) ** 2
        symmetry_penalty = 2500.0 * max(0.0, -min_esym / 5.0) ** 2 + 10.0 * max(0.0, (2.0 - min_esym) / 2.0) ** 2
        thermo_penalty = 5000.0 * max(0.0, -min_cs2 / 0.02) ** 2 + 5000.0 * max(0.0, (max_cs2 - 1.0) / 0.05) ** 2
        thermo_penalty += 5000.0 * max(0.0, -min_dp / 0.2) ** 2
        mass_penalty = 800.0 * max(0.0, (2.02 - mmax) / 0.05) ** 2
        nuclear_penalty = ((j - 31.0) / 2.5) ** 2 + ((ell - 60.0) / 18.0) ** 2
        distance_penalty = 0.20 * float(np.mean(((scales - 1.0) / 0.20) ** 2))
        loss = obs_chi2 + stability_penalty + symmetry_penalty + thermo_penalty + mass_penalty + nuclear_penalty + distance_penalty

        report = {
            "loss": float(loss),
            "valid_numerics": True,
            "anchor": anchor,
            "scales_vector": scales.tolist(),
            "scales": dict(zip(NAMES, map(float, scales))),
            "parameters": asdict(p),
            "constraints": list(constraint_ids),
            "observation_chi2": obs_chi2,
            "observation_terms": obs_terms,
            "penalties": {
                "stability": float(stability_penalty),
                "symmetry": float(symmetry_penalty),
                "thermodynamic_causality": float(thermo_penalty),
                "maximum_mass": float(mass_penalty),
                "nuclear": float(nuclear_penalty),
                "parameter_distance": float(distance_penalty),
            },
            "diagnostics": {
                "n0_fm3": n0, "J_MeV": j, "L_MeV": ell,
                "min_lambda_MeV_fm3": min_lambda,
                "min_Esym_MeV": min_esym,
                "min_delta_pressure_MeV_fm3": min_dp,
                "min_cs2": min_cs2, "max_cs2": max_cs2,
                "Mmax_Msun": mmax,
                "R_Mmax_km": float(radii[int(np.argmax(masses))]),
                "R14_km": radius_at_mass(masses, radii, 1.4),
                "R20_km": radius_at_mass(masses, radii, 2.0),
                "crust_match_density_fm3": float(hybrid.match_density),
                "crust_match_pressure_MeV_fm3": float(hybrid.match_pressure),
            },
        }
        if final:
            report["stability_grid"] = stability
            report["core_eos"] = core
            report["mr_curve"] = [
                {"central_pressure_MeV_fm3": float(pc), "mass_Msun": float(m), "radius_km": float(r)}
                for pc, m, r in zip(central_pressures, masses, radii)
            ]
        return report
    except Exception as exc:
        return invalid_report(anchor, scales, f"{type(exc).__name__}: {exc}")


def random_payloads(mode: str, local_anchor: str, constraint_ids: tuple[str, ...], count: int, seed: int):
    rng = np.random.default_rng(seed)
    payloads = []
    for _ in range(count):
        if mode == "global":
            anchor = "GQHD1" if rng.random() < 0.5 else "GQHD2"
            scales = rng.uniform(0.65, 1.35, len(NAMES))
        else:
            anchor = local_anchor
            scales = np.clip(1.0 + rng.normal(0.0, 0.085, len(NAMES)), 0.80, 1.20)
        payloads.append((anchor, scales.tolist(), constraint_ids))
    # Always include the unmodified local baselines.
    for anchor in (("GQHD1", "GQHD2") if mode == "global" else (local_anchor,)):
        payloads.append((anchor, np.ones(len(NAMES)).tolist(), constraint_ids))
    return payloads


def compact(report: dict) -> dict:
    return {k: v for k, v in report.items() if k not in {"stability_grid", "core_eos", "mr_curve"}}


def atomic_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        json.dump(obj, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        tmp = Path(stream.name)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--constraints", required=True, help="comma-separated: 6620,4715,3329,347")
    ap.add_argument("--mode", choices=("global", "local"), required=True)
    ap.add_argument("--anchor", choices=("GQHD1", "GQHD2"), required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--random-points", type=int, default=4000)
    ap.add_argument("--gradient-starts", type=int, default=6)
    ap.add_argument("--gradient-steps", type=int, default=28)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    args = ap.parse_args()

    constraint_ids = tuple(x.strip() for x in args.constraints.split(",") if x.strip())
    unknown = set(constraint_ids) - set(OBSERVATIONS)
    if unknown:
        raise SystemExit(f"unknown constraints: {sorted(unknown)}")
    payloads = random_payloads(args.mode, args.anchor, constraint_ids, args.random_points, args.seed)
    bounds = (0.65, 1.35) if args.mode == "global" else (0.80, 1.20)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers, maxtasksperchild=50) as pool:
        reports = list(pool.imap_unordered(evaluate_candidate, payloads, chunksize=1))
        reports.sort(key=lambda x: x["loss"])
        checkpoint = {
            "task_id": args.task_id,
            "stage": "random_complete",
            "random_points_requested": args.random_points,
            "random_evaluations": len(reports),
            "best_random": [compact(x) for x in reports[: min(20, len(reports))]],
        }
        checkpoint_path = args.checkpoint_dir / f"{args.task_id}_random.json"
        atomic_json(checkpoint_path, checkpoint)

        states = []
        for rep in reports[: args.gradient_starts]:
            states.append({
                "anchor": rep["anchor"],
                "x": np.asarray(rep["scales_vector"], dtype=float),
                "best": rep,
                "m": np.zeros(len(NAMES)),
                "v": np.zeros(len(NAMES)),
                "history": [{"step": 0, "loss": float(rep["loss"])}],
            })

        rng = np.random.default_rng(args.seed + 1000003)
        for step in range(1, args.gradient_steps + 1):
            c = 0.025 / (step ** 0.101)
            lr = 0.040 / (step ** 0.20)
            perturb = []
            deltas = []
            for state in states:
                delta = rng.choice((-1.0, 1.0), size=len(NAMES))
                deltas.append(delta)
                xp = np.clip(state["x"] + c * delta, *bounds)
                xm = np.clip(state["x"] - c * delta, *bounds)
                perturb.extend([
                    (state["anchor"], xp.tolist(), constraint_ids),
                    (state["anchor"], xm.tolist(), constraint_ids),
                ])
            vals = list(pool.map(evaluate_candidate, perturb, chunksize=1))
            trial_payloads = []
            for i, state in enumerate(states):
                fp, fm = vals[2 * i]["loss"], vals[2 * i + 1]["loss"]
                grad = ((fp - fm) / (2.0 * c)) * deltas[i]
                grad = np.clip(grad, -250.0, 250.0)
                state["m"] = 0.9 * state["m"] + 0.1 * grad
                state["v"] = 0.999 * state["v"] + 0.001 * grad * grad
                mhat = state["m"] / (1.0 - 0.9 ** step)
                vhat = state["v"] / (1.0 - 0.999 ** step)
                trial = np.clip(state["x"] - lr * mhat / (np.sqrt(vhat) + 1e-8), *bounds)
                trial_payloads.append((state["anchor"], trial.tolist(), constraint_ids))
            trials = list(pool.map(evaluate_candidate, trial_payloads, chunksize=1))
            for state, trial_rep in zip(states, trials):
                if trial_rep["loss"] <= state["best"]["loss"]:
                    state["x"] = np.asarray(trial_rep["scales_vector"], dtype=float)
                    state["best"] = trial_rep
                else:
                    # A conservative half step keeps descent robust when the
                    # profiled observational objective changes branch.
                    state["x"] = 0.5 * state["x"] + 0.5 * np.asarray(trial_rep["scales_vector"], dtype=float)
                state["history"].append({"step": step, "loss": float(state["best"]["loss"])})
            if step % 4 == 0 or step == args.gradient_steps:
                atomic_json(args.checkpoint_dir / f"{args.task_id}_gradient.json", {
                    "task_id": args.task_id,
                    "stage": f"gradient_{step}",
                    "states": [{"anchor": s["anchor"], "best": compact(s["best"]), "history": s["history"]} for s in states],
                })

    finalists = sorted((s["best"] for s in states), key=lambda x: x["loss"])
    # Dense validation is intentionally outside the worker pool so that the
    # final file is based on the validated high-resolution calculation.
    dense = []
    for rep in finalists[: min(3, len(finalists))]:
        dense.append(evaluate_candidate((rep["anchor"], rep["scales_vector"], constraint_ids), final=True))
    dense.sort(key=lambda x: x["loss"])
    winner = dense[0]
    final = {
        "schema": "gqhd-mr-constraint-fit-v1",
        "task_id": args.task_id,
        "method": {
            "random_mode": args.mode,
            "requested_local_anchor": args.anchor,
            "random_points": args.random_points,
            "workers": args.workers,
            "gradient": "projected SPSA with Adam moments",
            "gradient_starts": args.gradient_starts,
            "gradient_steps": args.gradient_steps,
            "seed": args.seed,
            "screening": "BPS crust + fast TOV + stability/causality barriers",
            "final_validation": "260-point stability grid, 520-point core EOS, BPS crust, 320-point TOV at dr=0.002 km",
        },
        "likelihood_note": "Profiled split-normal approximation from published marginal 68% M-R intervals; not the full 2-D posterior samples.",
        "observations": {cid: OBSERVATIONS[cid] for cid in constraint_ids},
        "best_random": [compact(x) for x in reports[:20]],
        "gradient_runs": [{"anchor": s["anchor"], "history": s["history"], "screening_best": compact(s["best"])} for s in states],
        "dense_finalists": dense,
        "winner": winner,
        "acceptance": {
            "thermodynamically_stable": bool(winner.get("diagnostics", {}).get("min_lambda_MeV_fm3", -math.inf) > 0.0),
            "positive_symmetry_energy": bool(winner.get("diagnostics", {}).get("min_Esym_MeV", -math.inf) > 0.0),
            "pressure_monotone": bool(winner.get("diagnostics", {}).get("min_delta_pressure_MeV_fm3", -math.inf) > 0.0),
            "causal": bool(0.0 < winner.get("diagnostics", {}).get("min_cs2", -1.0) and winner.get("diagnostics", {}).get("max_cs2", 2.0) <= 1.0),
            "supports_2Msun": bool(winner.get("diagnostics", {}).get("Mmax_Msun", 0.0) >= 2.0),
        },
    }
    atomic_json(args.output, final)
    print(json.dumps({"task_id": args.task_id, "output": str(args.output), "loss": winner["loss"], "acceptance": final["acceptance"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
