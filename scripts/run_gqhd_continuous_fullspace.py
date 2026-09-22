#!/usr/bin/env python3
"""Continuous full-space/local-neighborhood GQHD optimization.

Every round performs independent random sampling, projected SPSA refinement,
and dense BPS+TOV validation.  Validated improvements are saved atomically.
The process continues until PROJECT/STOP exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIT_PATH = ROOT / "scripts" / "run_gqhd_mr_constraint_fit.py"
spec = importlib.util.spec_from_file_location("gqhd_fit_helpers_continuous", FIT_PATH)
fit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fit
assert spec.loader is not None
spec.loader.exec_module(fit)

rmf = fit.search.rmf
from maxwell_crust_core import build_maxwell_crust_core_eos, read_bps
from tov_solver import EOSInterpolator, MEVFM3_TO_KM2

BPS_PATH = Path(os.environ.get("GQHD_BPS", str(ROOT / "data" / "external" / "bps.dat")))
BPS_ROWS = read_bps(BPS_PATH)

# Four masses are fixed by the model convention.  The remaining 21 parameters
# are sampled independently in the global search.  These explicit bounds are
# part of every output file, so the meaning of "full space" is reproducible.
NAMES = (
    "g_sigma", "g_omega", "g_rho", "g_a0", "b3", "b4", "b5", "b6", "b7", "b8", "b9",
    "g2", "g3", "g4", "g5", "g6", "g7", "c3", "c4", "d3", "m_sigma",
)
BOUNDS = {
    "g_sigma": (-12.0, 12.0), "g_omega": (8.0, 15.0), "g_rho": (3.0, 10.0), "g_a0": (-10.0, -3.0),
    "b3": (0.0, 400.0), "b4": (-2.0, 6.0), "b5": (-260.0, 260.0), "b6": (-40.0, 280.0),
    "b7": (-18.0, 18.0), "b8": (-550.0, 150.0), "b9": (-2800.0, 700.0),
    "g2": (-1800.0, 2400.0), "g3": (-5.0, 18.0), "g4": (-12.0, 12.0), "g5": (-80.0, 220.0),
    "g6": (-3300.0, 500.0), "g7": (-250.0, 300.0), "c3": (0.0, 280.0), "c4": (0.0, 320.0),
    "d3": (0.0, 1100.0), "m_sigma": (400.0, 560.0),
}
LOW = np.array([BOUNDS[n][0] for n in NAMES], dtype=float)
HIGH = np.array([BOUNDS[n][1] for n in NAMES], dtype=float)
WIDTH = HIGH - LOW
BASES = {p.name: p for p in rmf.PARAMS if p.name in {"GQHD1", "GQHD2"}}
STOP_REQUESTED = False


def project(x: np.ndarray, preferred_sigma_sign: float | None = None) -> np.ndarray:
    y = np.clip(np.asarray(x, dtype=float), LOW, HIGH)
    i = NAMES.index("g_sigma")
    if abs(y[i]) < 6.0:
        sign = preferred_sigma_sign if preferred_sigma_sign else (1.0 if y[i] >= 0 else -1.0)
        y[i] = 6.0 * sign
    return y


def vector_to_params(x: np.ndarray, name: str):
    # The template only supplies the fixed masses; all 21 sampled fields below
    # overwrite the corresponding template values.
    updates = dict(zip(NAMES, map(float, x)))
    return replace(BASES["GQHD1"], name=name, **updates)


def sample_global(rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(LOW, HIGH)
    # g_sigma has two disjoint sign-convention branches; avoid the unphysical
    # weak-coupling interval around zero while sampling both branches equally.
    x[0] = rng.choice((-1.0, 1.0)) * rng.uniform(6.0, 12.0)
    return x


def sample_local(rng: np.random.Generator, anchor: str) -> np.ndarray:
    base = np.array([getattr(BASES[anchor], n) for n in NAMES], dtype=float)
    sigma = np.maximum(0.08 * np.abs(base), 0.025 * WIDTH)
    x = base + rng.normal(0.0, sigma)
    return project(x, 1.0 if base[0] >= 0 else -1.0)


def invalid(x, mode, anchor, error, loss=1.0e15):
    return {"loss": float(loss), "valid_numerics": False, "mode": mode, "anchor": anchor, "vector": list(map(float, x)), "error": error}


def nuclear_penalty(q: dict) -> float:
    terms = (
        ((q["rho0_fm^-3"] - 0.155) / 0.012) ** 2,
        ((q["Ebind_MeV"] + 16.0) / 2.0) ** 2,
        ((q["K_MeV"] - 240.0) / 80.0) ** 2,
        ((q["J_MeV"] - 31.0) / 4.0) ** 2,
        ((q["L_MeV"] - 60.0) / 30.0) ** 2,
        ((q["Mstar/M"] - 0.65) / 0.15) ** 2,
    )
    barrier = 0.0
    ranges = {
        "rho0_fm^-3": (0.125, 0.185, 0.01), "Ebind_MeV": (-20.0, -12.0, 2.0),
        "K_MeV": (100.0, 450.0, 80.0), "J_MeV": (20.0, 45.0, 5.0),
        "L_MeV": (-30.0, 160.0, 30.0), "Mstar/M": (0.35, 0.95, 0.15),
    }
    for key, (lo, hi, scale) in ranges.items():
        barrier += 250.0 * max(0.0, (lo - q[key]) / scale) ** 2
        barrier += 250.0 * max(0.0, (q[key] - hi) / scale) ** 2
    return float(sum(terms) + barrier)


def make_core_for_maxwell(p, final: bool):
    # Resolve the transition on a dense low-density grid, then retain the
    # high-density reach needed for massive-star central pressures.
    if final:
        densities = np.unique(np.r_[np.geomspace(5.0e-4, 0.02, 100), np.linspace(0.02, 0.18, 241), np.linspace(0.18, 1.25, 430)])
    else:
        densities = np.unique(np.r_[np.geomspace(5.0e-4, 0.02, 28), np.linspace(0.02, 0.18, 54), np.linspace(0.18, 1.25, 78)])
    rows = []
    for density in densities:
        try:
            q = fit.search.beta_point(float(density), p)
        except Exception:
            # Some trial parameter sets do not bracket beta equilibrium at the
            # very lowest densities.  Keep their calculable branch; the
            # Maxwell builder will reject it if no physical crossing remains.
            if density < 0.02:
                continue
            raise
        rows.append({
            "rho_b": float(density), "Yp": q["Yp"],
            "pressure_total_mev_fm3": q["pressure"],
            "epsilon_total_mev_fm3": q["epsilon"],
        })
    return rows


def evaluate(payload, final: bool = False):
    x_list, mode, anchor, constraint_ids = payload
    x = np.asarray(x_list, dtype=float)
    p = vector_to_params(x, f"continuous-{mode}")
    try:
        nq = rmf.props(p)
        npen = nuclear_penalty(nq)
        if not final and npen > 2.0e4:
            return invalid(x, mode, anchor, "early nuclear-matter rejection", npen + 1.0e6)
        n0 = float(nq["rho0_fm^-3"])
        raw_core = make_core_for_maxwell(p, final)
        maxwell = build_maxwell_crust_core_eos(
            BPS_ROWS, raw_core, eos_interpolator_cls=EOSInterpolator,
            mevfm3_to_km2=MEVFM3_TO_KM2,
            # User-requested original optimization strategy. These broad
            # windows admit formal low-density crossings; passing the jump
            # checks alone is not validation of a physical crust-core phase
            # transition. Keep identical windows in before/after comparisons.
            crust_search=(1.0e-5, 0.12),
            core_search=(3.0e-4, 0.20),
        )
        core = maxwell.active_core_rows
        transition_ratio = float(maxwell.metadata["rho_core_fm3"] / n0)
        ratios = np.linspace(max(0.5, transition_ratio), min(6.0, 1.25 / n0), 260 if final else 19)
        stability = []
        for ratio in ratios:
            q = fit.search.beta_point(float(ratio * n0), p, with_curvature=True)
            q["n_over_n0"] = float(ratio)
            q["Esym_MeV"] = fit.search.symmetry_energy(float(ratio * n0), p)
            stability.append(q)
        eps = np.array([q["epsilon_total_mev_fm3"] for q in core])
        prs = np.array([q["pressure_total_mev_fm3"] for q in core])
        deps, dprs = np.diff(eps), np.diff(prs)
        cs2 = dprs / deps
        min_dp, min_cs2, max_cs2 = float(np.min(dprs)), float(np.min(cs2)), float(np.max(cs2))
        eos = maxwell.eos
        positive = prs[prs > 0]
        if len(positive) < 3:
            return invalid(x, mode, anchor, "insufficient positive pressure")
        pcs = np.geomspace(max(float(positive.min()) * 4.0, 0.025), float(positive.max()) * 0.985, 320 if final else 68)
        mr = fit.build_mr_curve(eos, pcs.tolist(), dr_km=0.002 if final else 0.022, progress=False)
        masses, radii, central_pressures = fit.stable_mr_arrays(mr)
        if len(masses) < 8:
            return invalid(x, mode, anchor, "invalid M-R branch")
        obs_terms = {cid: fit.observation_term(masses, radii, fit.OBSERVATIONS[cid]) for cid in constraint_ids}
        obs_chi2 = float(sum(q["chi2"] for q in obs_terms.values()))
        min_lambda = float(min(q["lambda_min"] for q in stability))
        min_esym = float(min(q["Esym_MeV"] for q in stability))
        mmax = float(np.max(masses))
        penalties = {
            "nuclear": npen,
            "stability": 3000.0 * max(0.0, -min_lambda / 20.0) ** 2 + 12.0 * max(0.0, (5.0 - min_lambda) / 5.0) ** 2,
            "symmetry": 3000.0 * max(0.0, -min_esym / 5.0) ** 2 + 10.0 * max(0.0, (2.0 - min_esym) / 2.0) ** 2,
            "thermo": 6000.0 * max(0.0, -min_cs2 / 0.02) ** 2 + 6000.0 * max(0.0, (max_cs2 - 1.0) / 0.05) ** 2 + 6000.0 * max(0.0, -min_dp / 0.2) ** 2,
            "mass": 1000.0 * max(0.0, (2.02 - mmax) / 0.05) ** 2,
            "local_distance": 0.0,
        }
        if mode == "local":
            base = np.array([getattr(BASES[anchor], n) for n in NAMES], dtype=float)
            penalties["local_distance"] = 0.12 * float(np.mean(((x - base) / np.maximum(0.10 * np.abs(base), 0.04 * WIDTH)) ** 2))
        loss = obs_chi2 + float(sum(penalties.values()))
        report = {
            "loss": loss, "valid_numerics": True, "mode": mode, "anchor": anchor,
            "vector": x.tolist(), "parameters": asdict(p), "constraints": list(constraint_ids),
            "observation_chi2": obs_chi2, "observation_terms": obs_terms, "penalties": penalties,
            "nuclear_properties": nq,
            "diagnostics": {
                "min_lambda_MeV_fm3": min_lambda, "min_Esym_MeV": min_esym,
                "min_delta_pressure_MeV_fm3": min_dp, "min_cs2": min_cs2, "max_cs2": max_cs2,
                "Mmax_Msun": mmax, "R_Mmax_km": float(radii[int(np.argmax(masses))]),
                "R14_km": fit.radius_at_mass(masses, radii, 1.4), "R20_km": fit.radius_at_mass(masses, radii, 2.0),
                "crust_core_matching": maxwell.metadata,
            },
        }
        report["acceptance"] = {
            "nuclear_matter": bool(0.125 <= nq["rho0_fm^-3"] <= 0.185 and -20 <= nq["Ebind_MeV"] <= -12 and 100 <= nq["K_MeV"] <= 450 and 20 <= nq["J_MeV"] <= 45 and -30 <= nq["L_MeV"] <= 160 and 0.35 <= nq["Mstar/M"] <= 0.95),
            "thermodynamically_stable": min_lambda > 0, "positive_symmetry_energy": min_esym > 0,
            "pressure_monotone": min_dp > 0, "causal": 0 < min_cs2 and max_cs2 <= 1, "supports_2Msun": mmax >= 2.0,
            "physical_Maxwell_jump": bool(maxwell.metadata["rho_core_fm3"] > maxwell.metadata["rho_crust_fm3"] and maxwell.metadata["energy_density_jump_MeV_fm3"] > 0),
        }
        # Ensemble scans retain the EOS table for every model that passes the
        # complete fast-physics gate.  The approximate M--R objective above is
        # deliberately not part of this gate and is only a screening
        # diagnostic; a separate multimessenger likelihood audit is required
        # before assigning an observationally accepted label.
        report["fast_physical_pass"] = bool(all(report["acceptance"].values()))
        report["astrophysical_status"] = "pending_full_likelihood_audit"
        if report["fast_physical_pass"] and not final:
            report["screening_core_eos"] = core
        if final:
            report["stability_grid"] = stability
            report["core_eos"] = core
            report["maxwell_eos"] = {"metadata": maxwell.metadata, "low_branch": maxwell.low_table, "high_branch": maxwell.high_table}
            report["mr_curve"] = [{"central_pressure_MeV_fm3": float(pc), "mass_Msun": float(m), "radius_km": float(r)} for pc, m, r in zip(central_pressures, masses, radii)]
        return report
    except Exception as exc:
        return invalid(x, mode, anchor, f"{type(exc).__name__}: {exc}")


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno()); tmp = Path(f.name)
    os.replace(tmp, path)


def eval_worker(payload):
    return evaluate(payload, final=False)


def refine(pool, starts, mode, anchor, constraints, rng, steps=8):
    states = [{"x": np.asarray(s["vector"]), "best": s, "m": np.zeros(len(NAMES)), "v": np.zeros(len(NAMES))} for s in starts]
    sign = None if mode == "global" else (1.0 if BASES[anchor].g_sigma >= 0 else -1.0)
    for step in range(1, steps + 1):
        c, lr = 0.018 / step**0.101, 0.025 / step**0.20
        probes, deltas = [], []
        for s in states:
            delta = rng.choice((-1.0, 1.0), len(NAMES)); deltas.append(delta)
            xp = project(s["x"] + c * WIDTH * delta, sign)
            xm = project(s["x"] - c * WIDTH * delta, sign)
            probes.extend([(xp.tolist(), mode, anchor, constraints), (xm.tolist(), mode, anchor, constraints)])
        vals = pool.map(eval_worker, probes, chunksize=1)
        trials = []
        for i, s in enumerate(states):
            grad = ((vals[2*i]["loss"] - vals[2*i+1]["loss"]) / (2*c)) * deltas[i] / WIDTH
            grad = np.clip(grad, -200, 200)
            s["m"] = .9*s["m"] + .1*grad; s["v"] = .999*s["v"] + .001*grad*grad
            move = lr * WIDTH * (s["m"]/(1-.9**step)) / (np.sqrt(s["v"]/(1-.999**step))+1e-8)
            xt = project(s["x"] - move, sign)
            trials.append((xt.tolist(), mode, anchor, constraints))
        reps = pool.map(eval_worker, trials, chunksize=1)
        for s, r in zip(states, reps):
            if r["loss"] < s["best"]["loss"]:
                s["x"] = np.asarray(r["vector"]); s["best"] = r
    return sorted((s["best"] for s in states), key=lambda q: q["loss"])


def all_accepted(report):
    return report.get("valid_numerics") and all(report.get("acceptance", {}).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True); ap.add_argument("--constraints", required=True)
    ap.add_argument("--mode", choices=("global", "local"), required=True)
    ap.add_argument("--anchor", choices=("NONE", "GQHD1", "GQHD2"), required=True)
    ap.add_argument("--workers", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--project", type=Path, required=True); ap.add_argument("--batch-factor", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=0)
    args = ap.parse_args()
    constraints = tuple(x.strip() for x in args.constraints.split(",") if x.strip())
    task = args.project / "tasks" / args.task_id; task.mkdir(parents=True, exist_ok=True)
    config = {"task_id": args.task_id, "constraints": constraints, "mode": args.mode, "anchor": args.anchor,
              "workers": args.workers, "seed": args.seed, "parameter_names": NAMES, "global_bounds": BOUNDS,
              "fixed_masses_MeV": {"m_omega": 783.0, "m_rho": 770.0, "m_a0": 980.0, "M_N": 939.0},
              "crust_core_matching": "Maxwell: P_BPS(mu_B)=P_core(mu_B), explicit epsilon jump in TOV",
              "maxwell_search_density_fm3": {"BPS": [0.01, 0.08], "core": [0.01, 0.20]},
              "stop_file": str(args.project / "STOP")}
    atomic_json(task / "config.json", config)
    rng = np.random.default_rng(args.seed)
    best_loss = math.inf; round_no = 0
    if (task / "best.json").exists():
        old = json.loads((task / "best.json").read_text()); best_loss = float(old["loss"]); round_no = int(old.get("round", 0))
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, maxtasksperchild=40) as pool:
        while not (args.project / "STOP").exists():
            round_no += 1
            samples = []
            if args.mode == "local":
                base_x = np.array([getattr(BASES[args.anchor], n) for n in NAMES], dtype=float)
                samples.append((base_x.tolist(), args.mode, args.anchor, constraints))
            for _ in range(args.workers * args.batch_factor - len(samples)):
                x = sample_global(rng) if args.mode == "global" else sample_local(rng, args.anchor)
                samples.append((x.tolist(), args.mode, args.anchor, constraints))
            reports = list(pool.imap_unordered(eval_worker, samples, chunksize=1)); reports.sort(key=lambda q:q["loss"])
            finite = [q for q in reports if q.get("valid_numerics")]
            starts = finite[: min(3, len(finite))]
            if starts:
                refined = refine(pool, starts, args.mode, args.anchor, constraints, rng)
                candidate = refined[0]
                dense = evaluate((candidate["vector"], args.mode, args.anchor, constraints), final=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                atomic_json(task / "latest_validated.json", {"round": round_no, "screening": candidate, "dense": dense})
                if dense.get("valid_numerics") and dense["loss"] < best_loss:
                    best_loss = float(dense["loss"])
                    payload = {"round": round_no, "timestamp": stamp, "loss": best_loss, "accepted": all_accepted(dense), "config": config, "screening": candidate, "dense": dense}
                    atomic_json(task / "improvements" / f"round_{round_no:06d}_loss_{best_loss:.6g}_{stamp}.json", payload)
                    atomic_json(task / "best.json", payload)
                    print(json.dumps({"task":args.task_id,"round":round_no,"NEW_BEST":best_loss,"accepted":all_accepted(dense)}), flush=True)
            status = {"task": args.task_id, "round": round_no, "evaluated_this_round": len(reports), "finite_this_round": len(finite), "best_loss": best_loss, "updated": time.strftime("%F %T %Z")}
            atomic_json(task / "status.json", status)
            with (task / "history.jsonl").open("a") as f: f.write(json.dumps(status)+"\n")
            if args.max_rounds and round_no >= args.max_rounds: break


if __name__ == "__main__":
    main()
