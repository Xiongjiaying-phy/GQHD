# Data provenance and external inputs

## Not redistributed

1. BPS table: place the exact historical table at `data/external/bps.dat`, or set `GQHD_BPS`. Its checksum is in `data/external/Required.json`. The original table's distribution provenance/permission remains unresolved. Do not download an arbitrary BPS-labelled table and claim exact reproduction. Matching is sensitive to this input.
2. GW170817 sample table: public release entry https://dcc.ligo.org/LIGO-P1800061/public . The historical file is `low_spin_PhenomPNRT_posterior_samples.dat.gz`; its destination, byte count and hash are recorded in `Required.json`. The loader skips one header row and uses zero-based columns 2,3 for detector masses and 4,5 for Lambda1,Lambda2. Preserve the original format. No automatic download is included because the exact archive/member and redistribution terms must be checked.

`scripts/check_external.py` checks both files before a refinement. Parameter/curve checks, redrawing and saturation calculations do not require these files.

## Included numerical material

- Model results, computed curves and loss diagnostics originate from the saved paper snapshot.
- HIC vertices in `flow_band.json`/`audit/flow.json` are a manual digitization of Danielewicz, Lacey and Lynch (2002), Fig. 3, not an author-provided numeric table: https://arxiv.org/abs/nucl-th/0208016 . Preserve this attribution and review redistribution permissions before public release.
- Original M–R posterior samples and Kaon/EFT polygon data are not included. Saved figures contain these overlays; their numerical reconstruction and permissions need the original source material. The new plot command redraws model curves only and does not fabricate observational regions.

The saved figures/tables include fixed reference models. They are retained as existing paper outputs, not additional optimization targets. The nine-candidate CSV is a historical seed-selection input; it is not a nine-model release.

## References already encoded in the optimizer

M–R input attribution is in `OBSERVATIONS` in `scripts/run_gqhd_mr_constraint_fit.py`: Salmi et al. (2024), Choudhury et al. (2024), Mauviard et al. (2025), Doroshenko et al. (2022). These are marginal summaries, not imported joint posterior likelihoods. Verify the bibliography against the final manuscript before submission.
