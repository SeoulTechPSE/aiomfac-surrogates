# Changelog

## v3.0.0

This release accompanies the excess-Gibbs-energy revision of both papers. All new material is in
[`excess_gibbs/`](excess_gibbs/); the v2 folders are unchanged and reproduce the earlier versions.

- **Part 1:** every surrogate now predicts the molar excess Gibbs energy of the binary water–organic mixture, from
  which both activity coefficients follow by automatic differentiation (Gibbs–Duhem exact). All tables and figures
  were regenerated with this output (`excess_gibbs/part1_reruns/`). New analyses: comparison of output forms
  (unconstrained ReLU/SiLU, excess Gibbs energy with and without a derivative loss), binary liquid–liquid phase
  separation of the held-out molecules, and phase equilibrium of α-pinene oxidation products with the surrogate inside
  `aiomfac_py`'s Gibbs-energy-minimization solver. New release ensemble trained on 3705 BIMOG + MCM molecules.
- **Part 2:** the electrolyte surrogate is now an excess-Gibbs-energy model of the free species of 14 ions
  (Debye–Hückel term plus a mole-fraction Margules expansion with network coefficients), trained on 80 000 free-species
  compositions; the bisulfate and carbonate equilibria are solved explicitly (`speciation.py`). The organic–electrolyte
  coupling (Option C) is a learned cross term of the excess Gibbs energy, trained with two added dilute salt levels.
  New analysis: phase equilibrium of inorganic particles with the surrogate inside the solver.
- Requires `aiomfac_py` v1.1.0 (phase-equilibrium solver).

## v2.0.1

Corrections after internal review of the revised manuscripts.

- **Part 1, Table 2:** pool C (BIMOG + synthetic N) was evaluated on its own hash split, which also placed some
  synthetic molecules in the test set. All pools are now evaluated on the same held-out BIMOG molecules
  (`02_representation_emb_reruns.ipynb` §1); the pooled error of pool C changes from 0.144 to 0.148, the
  per-category errors are unchanged.
- **Part 1, viscosity emulator (§7 of `02`):** errors are also reported for dry, near-glassy points
  (w_water ≤ 0.1, T ≤ Tg + 50 K) and on the 123-molecule subset used for the coefficient model. Fig. 4b shows the
  first of these.
- **Part 2:** `02_split_composition_check.ipynb` added; it explains why the error for unseen ion combinations is
  lower than for new compositions of seen ones (the held-out set contains only single-salt, mostly dilute
  solutions).

## v2.0.0

This release accompanies the revised manuscripts of Parts 1 and 2. Results of v1 (v1.0.0, v1.1.0) are superseded;
the v1.1.0 notebooks are kept under `legacy_v1/` folders for traceability, and the complete v1 state remains
available under the git tag `v1.1.0`.

### Part 1

- **New main model: learned subgroup embedding (R_emb).** Seeded and cross-validated comparisons show that R_emb
  generalizes best on every pool. The apparent advantage of hand-crafted descriptors over raw subgroup counts in v1
  was an artefact of input standardization (control R_count′ added).
- **One label set and protocol for all main tables.** Tables 1, 2, 4 and 7 now come from training runs on a common
  set of AIOMFAC labels (`01_unified_reruns_and_checkpoint.ipynb`, `02_representation_emb_reruns.ipynb`).
- **Explicit seeding** of every training run (v1 drew from the global random state). Results are reported as
  mean ± sd over 3–5 seeds.
- **MCM v3.3.1 pool** (3337 atmospheric oxidation products) added. It is used for learning curves, transfer tests
  and the comparison with the GNN at larger data scale.
- **BAT baseline** (Gorkowski et al., 2019) added.
- **Coverage analysis:** the halogen "ceiling" of v1 is replaced by an analysis of atoms that S2AS cannot match.
  These occur in all halogenated and 93 % of the halogen-free nitrogen-containing BIMOG compounds and limit AIOMFAC's accuracy
  against reality, not surrogate fidelity.
- **Sensitivity analyses** extended: a real-molecule variance decomposition and a Sobol analysis of AIOMFAC's
  238 interaction parameters were added, and the descriptor-space analysis now reports confidence intervals and an
  out-of-distribution check.
- **Viscosity:** Tg-only and Tg + O:C controls, Tg-uncertainty propagation, a like-for-like comparison with the
  coefficient model, and an R_emb emulator.
- **Cost benchmark** redone end to end in one session (`05_cost_benchmark.ipynb`), including SMILES decomposition
  and multicomponent scaling.
- **GNN and hybrids (Table 3) re-run at full scale over 5 seeds** on the common label set
  (`08_gnn_hybrids_multiseed.ipynb`). The v1 GNN runs were undertrained (at most 25 epochs). With full training,
  the GNN error halves, synthetic pretraining shows no consistent benefit, the end-to-end hybrids no longer
  overfit, and a frozen pretrained encoder remains worse than a frozen random one.
- **Release checkpoint:** an R_emb ensemble (3 seeds) trained on all 3705 BIMOG + MCM molecules, with
  `predict_remb.py`. It replaces both v1 descriptor-MLP checkpoints (`trained_model_delivered/` and the
  dense-grid `trained_model_dense/` of v1.1.0).
- **Literature-compound check** (v1.1.0, Part 1 Sect. 8) redone with R_emb surrogates over three seeds (Table 7,
  `02_representation_emb_reruns.ipynb` §6); the Part 2 worked examples now use the R_emb release checkpoint.
- The 373 MB raw parameter-GSA file is not distributed; `results/param_gsa_meanST.csv` holds what Fig. 5c needs.

### Part 2

- **Corrected training protocol** for the electrolyte surrogate: mini-batches, per-column output standardization,
  and a validation split separate from the interpolation test set (v1 used the test set for early stopping).
  Results are now reported over 5 seeds, together with a replication of the v1 protocol.
- **Comparison with real Fortran reference output** (cases c005, c006). In v1, the "Fortran cross-check" was a
  comparison with `aiomfac_py` at a single composition.
- **End-to-end cost** including construction of the input vector.
- **Sobol analysis through the surrogate checked against AIOMFAC directly** (same samples).
- **Organic–electrolyte coupling rebuilt on the Part 1 R_emb surrogate.** The residual network uses the R_emb
  embedding. The OOD guard is calibrated and tested on separate organics, and is found to be unnecessary with R_emb.
- **Comparison with experimental γ± of KCl in water–ethanol** (Lopes et al., 1999) added.
- Self-contained folder: the v1 bundles (`*_bundle.zip`, which included a copy of `aiomfac-python`) are replaced by
  the required helper modules, data and reference files; `aiomfac_py` is installed from its repository.
