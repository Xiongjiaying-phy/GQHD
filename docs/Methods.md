# Model and implementation contract

## Physics layers

The model contains nucleons and sigma, omega, rho and a0/delta fields, with the self- and cross-interactions implemented explicitly in `tmp/rmf_verified_convention_properties.py` (`residual`, `epsilon`). That code, including field signs and numerical factors, is authoritative for this saved snapshot. Do not replace negative couplings with their absolute values or transfer same-named parameters to another convention without a term-by-term check.

Masses and dimensional couplings b5, b9, g2, g6, g7 are in MeV; the other tabulated couplings are dimensionless. Uniform-core matter is cold **npe**, with no muons: mu_n-mu_p=mu_e, n_p=n_e. The lower numerical density is positive, not exactly zero.

The paper path is `run_gqhd_tidal_refine -> run_gqhd_pressure_refine -> run_gqhd_continuous_fullspace`. The helper `run_gqhd_mr_constraint_fit.py` also contains an older fixed-density crust workflow; **do not run that helper's main routine to reproduce the final paper**. The active path uses `maxwell_crust_core.py` and retains distinct phases.

Crust matching follows the historical wide-window formal Maxwell construction, not a newly validated inner-crust transition. The TOV–Love path retains the finite-density-jump correction. Neither the historical matching prescription nor its BPS provenance is silently replaced during packaging. Core sound-speed tables describe uniform branches, not a finite difference across a Maxwell jump.

Saved MR arrays contain very-low-mass points with radii reaching the historical 100 km integration guard. These are not validated stellar solutions. They are preserved for traceability, not endorsed as physical predictions; the paper-style MR plot uses R=8.5–15.5 km and M=0–3.35 solar masses. A complete low-mass-sequence audit remains necessary before interpreting those points. Changing or filtering historical data would require a separately versioned recalculation.

## Objective and screening

See `docs/Loss.md` for formulas. M–R terms profile independent asymmetric marginal residuals on 181 mass points within the stable sequence and the observed ±4-width mass window. These are not correlated 2D posterior likelihoods or posterior credible levels. The reported medians serve as the approximate density peaks.

Nuclear target/scale pairs are n0=.155/.012, E0=-16/2, K=240/80, J=31/4, L=60/30, Mstar/M=.65/.15. Separate window penalties have weight 250. They are adopted soft constraints, not direct combined experimental errors. No independent empirical m_sigma target is used; it has parameter bounds and is included in parameter-distance regularization.

The local refinement box is centered on the current seed. Quadratic parameter regularization is centered on the original model anchor, which can differ from the seed. Soft mass penalties vanish at 2.02 solar masses; final acceptance requires at least 2.0. Stability and causality refer only to the sampled domains. Physically failing but numerically evaluable candidates may receive a score before exclusion; it is incorrect to claim all are rejected before evaluation.

HIC is an out-of-band log-pressure penalty, weight 10, on symmetric matter, using a fixed reference density .16 fm^-3. Kaon and chiral EFT overlays do not enter this snapshot's fit. Public MR contours plotted in saved figures do not define the optimizer's MR score.

GW uses a 4D KDE of released low-spin PhenomPNRT samples in detector chirp mass, mass ratio and component deformabilities, with bandwidth .30 and the documented PE-prior convention. It is posterior recycling, not a new strain likelihood. Saved dense checks compare 32x96 and 64x192 quadrature with log-integral difference <=.02. The GW score has an arbitrary common additive constant. Different groups have different MR targets, so total scores cannot be ranked across groups as if they were the same objective.

## Packaging changes

The five objective modules are sourced from the saved Loss package. Toolkit defaults are changed to repository-relative paths; `GQHD_BPS` optionally specifies the exact external BPS table. Dependency modules are copied from the available workspace; their source hashes are recorded, but they are not claimed to be independently archived at the original optimization date. Stored numerical results and parameter values are not modified. Local path/host strings in JSON metadata are redacted. `Provenance.json` records original and packaged hashes.
