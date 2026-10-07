# Excess-Gibbs-energy surrogates (v3)

Reproducibility material for the revised papers of the series, in which every surrogate predicts an **excess Gibbs
energy** and all activity coefficients follow from it by differentiation, so that the Gibbs–Duhem equation holds
exactly and the surrogates can be used inside Gibbs-energy-minimization (phase-equilibrium) solvers:

- *Part 1: Thermodynamically consistent binary water–organic activity coefficients, molecular representation and data scale*
- *Part 2: Thermodynamically consistent activity coefficients of aqueous electrolyte mixtures and organic–electrolyte coupling*

The earlier versions (activity coefficients predicted directly; Part 2 on nominal molalities) remain in
[`../part1_organic_surrogate/`](../part1_organic_surrogate/) and [`../part2_inorganic_surrogate/`](../part2_inorganic_surrogate/)
(release v2.0.1); several scripts here read data from those folders.

All scripts are plain Python. Requirements: `aiomfac_py` v1.1.0 or later (with the `carbonate`, `smiles` and `sle`
extras), PyTorch, NumPy, SciPy, pandas, RDKit, SALib, matplotlib, pyarrow. Run every script **from its own folder**
(`code/`, `part1_reruns/` or `part2_figures/`); results go to `results/` (or `part1_reruns/results/`).

## Layout

| Folder | Content |
|---|---|
| `code/` | label generation, training and analysis scripts (Part 1: `s1_*`; Part 2: `s2_*`, `speciation.py`, `s3_coupling.py`) |
| `results/` | labels (`s1_labels.npz`, `s2v2_labels.npz`), run tables, logs and all trained networks (`s1_models/`, `s2v2_models/`, `s3_models/`) |
| `part1_reruns/` | Part 1 reruns of the representation, pool, GNN, learning-curve, literature, Sobol and cost analyses with excess-Gibbs-energy output, their results and the figures |
| `part2_figures/` | figure script and figures of Part 2 |

## Part 1 (binary water–organic)

| Paper item | Script → result |
|---|---|
| labels (Part 1 molecules, compositions, temperatures; derivative labels) | `code/s1_data.py` → `results/s1_labels.npz`, `s1_pool.json` (needs `../part1_organic_surrogate/mcm_v331_species.tsv` from `fetch_mcm_species.py`) |
| Table 1 (representations), Table 2 / Fig. 1 (training pools), coverage | `part1_reruns/ge_t1.py {hash,cv,pools,coverage}` → `GE_table1_*`, `GE_table2_pools_runs.csv`, `GE_coverage_oof.csv` |
| Table 3 / Fig. 2 (GNN and hybrids), MCM pool | `part1_reruns/ge_gnn.py {t3,mcm}` → `GE_table3_runs.csv`, `GE_table4_gnn_runs.csv` |
| Table 4 (output forms), Table 5 (LLPS), Fig. 3 | `code/s1_train.py` → `results/s1_runs.csv`; `code/s1_llps.py` → `s1_llps.json`; `part1_reruns/make_fig7_ge.py` |
| phase equilibrium of five α-pinene products (55 points) | `code/s1_release.py` (release ensemble) → `s1_models/ge_silu_sob_release_seed*.pt`; `code/s1_a2.py` → `s1_a2.json` |
| Table 6 / Fig. 4 (learning curves) | `part1_reruns/ge_t4.py` → `GE_learning_curve_*` |
| Fig. 6 (sensitivity) | `part1_reruns/ge_sobol.py` → `GE_sobol_descriptor_indices.csv` |
| Table 8 (cost) | `part1_reruns/ge_cost.py` → `GE_cost.json` |
| Table 9 / Fig. 7 (literature compounds) | `part1_reruns/ge_lit.py` → `GE_literature_*` |
| Figures 1, 2, 4–7 | `part1_reruns/make_figs_ge.py` (Fig. 6c needs `param_gsa_raw.npz` from `../part1_organic_surrogate/04_sensitivity_viscosity_analyses.ipynb`, not archived because of its size; Fig. 4 (viscosity) is unchanged from v2) |

**Release ensemble (Part 1).** `results/s1_models/ge_silu_sob_release_seed{0,1,2}.pt`: excess-Gibbs-energy networks with
a 16-dimensional subgroup embedding, trained on all 3705 BIMOG and MCM molecules (`s1_release_meta.json`; vocabulary in
`s1_labels.npz["vocab"]`; class `Net(..., "ge", "silu")` in `code/s1_train.py`). Valid for w_water 0.02–0.98 and
273–313 K.

## Part 2 (electrolytes and organic–electrolyte coupling)

| Paper item | Script → result |
|---|---|
| free-species labels (80 000 compositions; 3602 extreme states removed at training) | `code/s2_data.py 80000 v2` → `results/s2v2_labels.npz` |
| Table 1 (output forms), Fig. 1a | `S2_TAG=s2v2 python s2_train.py {direct,ge,gex}` → `s2v2_runs_*.csv`, `s2v2_models/`; `s2_part2.py t1` → `P2GE_table1_runs.csv`, `P2GE_per_species.csv` |
| release ensemble | `S2_TAG=s2v2 S2_RELEASE=1 python s2_train.py gex` → `s2v2_models/gex_release_seed*.pt` |
| speciation (bisulfate, carbonate, CaSO4) | `code/speciation.py` |
| Table 2 (end to end), Fig. 1b | `s2_part2.py e2e` → `P2GE_e2e_*` |
| Table 3 (Fortran c005, c006) | `s2_part2.py fort` → `P2GE_fortran.csv` |
| bisulfate check | `s2_part2.py bis` → `P2GE_bisulfate*.{csv,json}` |
| cost | `s2_part2.py cost` → `P2GE_cost.json` |
| Fig. 3 (Sobol) | `s2_part2.py sobol` → `P2GE_sobol.csv` |
| Fig. 2 (phase equilibrium of inorganic particles) | `s2_pe_gex.py gex direct` → `s2v2_pe_gex.json` |
| Tables 4–6, Figs. 4–6 (Option A/C, worked examples, Lopes et al.) | `s3_coupling.py` → `s3_*.csv`, `s3_guard.json`, `s3_models/` |
| all figures | `part2_figures/make_figs_p2ge.py` |

**Electrolyte release ensemble.** `results/s2v2_models/gex_release_seed{0..4}.pt` (class `GEXNet` in
`code/s2_train.py`): excess Gibbs energy of water + the free species of 14 ions (Na+, K+, NH4+, Ca2+, H+, Cl-, NO3-,
SO4--, HSO4-, HCO3-, CO3--, IO3-, I-, OH-) at 263–313 K. Use the mean of the five G functions; for nominal compositions,
solve the dissociation equilibria with `speciation.speciate`. `s2_pe_gex.SurrogatePE` shows how to use it inside
`aiomfac_py.phase_equilibrium.PhaseEquilibrium`.

**Cross term.** `results/s3_models/cross_seed{0,1,2}.pt` (class `Cross` in `code/s3_coupling.py`) for water + one
organic + one salt, used with the two release ensembles above.
