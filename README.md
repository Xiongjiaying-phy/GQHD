# GQHD1–4: paper-result reproduction

This repository prepares the saved **2026-09-08 HIC + GW170817 refinement** for sharing. Release assembly: 2026-09-22. It is not a new fit, a posterior ensemble, or a proof of global optimality. The latest manuscript PDF was not supplied; implementation details were checked against the saved code, not a PDF.

## What is included

- `models/`: full-precision GQHD1–4 saved results and parameters.
- `scripts/`: frozen objective/refinement modules plus EOS and matching helpers.
- `tmp/rmf_verified_convention_properties.py`: historical model implementation (the path is retained for compatibility; this is required source, not disposable output).
- `vendor/rmf/TOVtools/`: TOV integration and legacy helper dependency.
- `nuclear_astro_posterior_20260720/src/`: coupled TOV–Love integration and jump handling.
- `data/curves/`, `data/beta/`: saved computed curves; no reoptimization needed.
- `audit/`, `docs/Loss.md`: per-model score decomposition and explicit penalties.
- `tables/`, `figures/`: saved paper outputs; these also show the fixed TM1, NL3 and FSU-delta6.7 reference curves. No reference-model refitting or coupling-zero experiments are included.
- `output/gqhd_pressure_refine_20260908/`: historical input seeds and HIC vertices required by the frozen refinement entry point.

## Setup and quick reproduction

Use Python 3.10 or newer. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py check
python scripts/reproduce.py plot
python scripts/reproduce.py nuclear
python scripts/validate.py
```

`check` validates saved loss sums and parameter consistency without observational data. `plot` redraws four-model MR, nuclear, pressure and core sound-speed curves into `generated/` (not observational contours). `nuclear` independently recomputes saturation properties from saved parameters; it is not an optimization. Original observation overlays remain available in `figures/`.

## Re-running the historical refinement

External inputs are intentionally excluded. First read [Data](docs/Data.md), provide the exact BPS table and GW sample file, and validate their SHA-256 checksums against `data/external/Required.json`.

```sh
python scripts/check_external.py
python scripts/run_gqhd_tidal_refine.py --group 1 --workers 4 --steps 8 --out generated/refine1
```

Repeat with groups 2, 3 and 4, using different output directories. The entry point starts from the **saved pre-tidal pressure-refined seeds**, not the final model files. Do not point it at existing result directories. A run may be expensive. Processor counts are user choices; no remote-machine launch scripts are included.

The stored `preflight.json` is historical evidence, not a fresh validation of your environment. Full tidal optimization has not been rerun during packaging. Floating-point libraries and integration details can affect optimizer trajectories.

With the historical BPS table supplied, `python scripts/validate.py --objective` additionally recomputes the four HIC and nuclear scores and the MR profile score on saved MR curves. Packaging test evidence is in `audit/Validation.json`.

## Model groups

| Model | M–R inputs | Original regularization anchor |
|---|---|---|
| GQHD1 | J0740+6620, J0437−4715 | old GQHD2 |
| GQHD2 | base pair + J0614−3329 | old GQHD1 |
| GQHD3 | base pair + HESS J1731−347 | old GQHD1 |
| GQHD4 | all four objects | old GQHD1 |

New labels GQHD1–4 are not the old two model names. All sets share nuclear penalties, HIC and the GW170817 joint-posterior term.

## Read before interpreting

Read [Methods](docs/Methods.md), [Data](docs/Data.md), [Loss](docs/Loss.md), and [Release checklist](docs/Release.md). All central values/widths, penalty weights and acceptance windows must be reported as adopted choices, not uniformly independent experimental standard deviations.

Code and redistributed-data ownership must be confirmed before selecting a license. This preparation does not grant a new license, invent authors, or upload anything to GitHub.
