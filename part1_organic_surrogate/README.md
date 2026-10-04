# Part 1 — Binary water–organic surrogates

Reproduces the tables and figures of *Neural-network surrogates for AIOMFAC – Part 1: Binary water–organic
activity coefficients, molecular representation and data scale*.

## Setup

```bash
pip install -r ../requirements.txt
python fetch_mcm_species.py        # writes mcm_v331_species.tsv (needed by notebooks 01–05)
```

Run each notebook with this folder as the working directory. `bimog_unified.json` is included.

## Notebooks

| Notebook | Content | Run time* |
|---|---|---|
| `01_unified_reruns_and_checkpoint.ipynb` | Representation comparison on a common label set (R_count, R_count′, R_desc, R_emb; hash split and CHO cross-validation); R_desc MLP and GNN on MCM subsets; **training of the release checkpoint** | ~55 min |
| `02_representation_emb_reruns.ipynb` | All R_emb analyses: training pools, Model A_emb, coverage cross-validation, MCM learning curves and transfer, BAT on BIMOG, literature compounds (M1–M4), viscosity emulator | ~40 min |
| `03_mcm_learning_curve_desc_and_bat.ipynb` | MCM pool construction and filter statistics; R_desc learning curves; BAT on MCM | ~55 min |
| `04_sensitivity_viscosity_analyses.ipynb` | Descriptor-space Sobol analysis with CIs and OOD check; real-molecule variance decomposition; Sobol analysis of AIOMFAC interaction parameters; viscosity controls and Tg sensitivity | ~40 min |
| `05_cost_benchmark.ipynb` | All timings of Sect. 3.8, Table 6 and Appendix A (Run 1) | ~1 min |
| `06_coverage_unmatched_atoms.ipynb` | Unmatched-atom statistics, halogen/H-substituted twin test, feature–label consistency variants (R_desc) | ~10 min |
| `07_modelA_desc_multiseed_and_bat.ipynb` | Seeded R_desc training-pool runs; BAT on BIMOG | ~5 min |
| `08_gnn_hybrids_multiseed.ipynb` | Table 3: Models A_emb, A_desc and GNN/hybrids B–G on one label set, full scale, 5 seeds (synthetic pretraining pool included) | ~25 min |
| `make_figures.py` | Regenerates `figures/` from `results/` | seconds |

\* Apple-silicon laptop, CPU only.

`supplementary/` holds two earlier analyses that are referred to in the text: the first seeded representation
pass (`feature_ablation_first_pass.ipynb`) and the R_desc literature-compound evaluation
(`malonic_acid_desc_surrogate.ipynb`). `legacy_v1/` holds the v1 notebook (see its README).

## Where each result comes from

| Paper item | Source |
|---|---|
| Table 1 | `01` §1–§3 (BIMOG and CHO columns; R_desc and GNN on MCM); R_emb on MCM from `02` §4 |
| Table 2, Fig. 1 | `02` §1 |
| Sect. 3.3 | `02` §3 (R_emb cross-validation and regression); `06` (unmatched atoms, twin test, R_desc variants) |
| Table 3, Fig. 2 | `08` (all trained models); BAT: `07` §4 and `02` §5 |
| Table 4, Fig. 3 | R_emb: `02` §4; R_desc and GNN: `01` §3; BAT on MCM: `03` §4 |
| Table 5, Fig. 4 | R_emb emulator: `02` §7; T_g-only, T_g + O:C, R_desc, T_g sensitivity, coefficient model: `04` §C1–C3 |
| Sect. 3.7, Fig. 5 | `04` §A1–A4 |
| Sect. 3.8, Table 6, Appendix A Run 1 | `05`; Appendix A Run 2: `legacy_v1/timing_results_gpu_run.json` |
| Table 7, Fig. 6 | `02` §6; R_desc value for malonic acid: `supplementary/malonic_acid_desc_surrogate.ipynb` |
| Release checkpoint | `01` §4 → `checkpoints/` |

**Synthetic pretraining pool (Models C, E, F).** The pool in `08` follows the published template counts
(chain lengths 1–10 × 3 backbones × 13 monofunctional templates + 6 difunctional templates; 361 molecules after
removing BIMOG compounds). The exact template list of the original pool was not archived; the templates used
here are listed in the notebook.

## Release checkpoint

`checkpoints/` contains three R_emb networks (seeds 0–2) trained on all 3705 BIMOG + MCM molecules, the subgroup
vocabulary and normalization statistics (`remb_bimog_mcm_meta.json`), and `predict_remb.py`:

```python
from predict_remb import predict
mean, sd, n_unmatched = predict("CC(=O)C1CC(CC(=O)O)C1(C)C", [0.2, 0.5, 0.8], 298.15)
# mean[:, 0] = ln γ_water, mean[:, 1] = ln γ_organic (ensemble mean); sd = ensemble spread
```

Valid ranges: water mass fraction 0.02–0.98 and 273–313 K. The molecule's AIOMFAC subgroups must be in the stored
vocabulary. Molecules with atoms that S2AS cannot match are flagged, because AIOMFAC (and therefore the surrogate)
ignores those atoms.

## Results files

`results/` holds the per-run outputs behind the tables and figures. Files prefixed `C_` come from `02`, `U_` from
`01`; `param_gsa_meanST.csv` is the compact summary of the parameter Sobol analysis (the 373 MB raw file written by
`04` is not distributed).
