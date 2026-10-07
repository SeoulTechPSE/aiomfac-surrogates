# aiomfac-surrogates

Reproducibility material for a two-part paper series on thermodynamically consistent neural-network surrogates for
[AIOMFAC](https://github.com/andizuend/AIOMFAC) activity-coefficient calculations, built on
[`aiomfac_py`](https://github.com/SeoulTechPSE/aiomfac-python), a pure-Python port of AIOMFAC validated against the
Fortran reference model.

| | Paper |
|---|---|
| **Part 1** | *Neural-network surrogates for AIOMFAC – Part 1: Thermodynamically consistent binary water–organic activity coefficients, molecular representation and data scale* |
| **Part 2** | *Neural-network surrogates for AIOMFAC – Part 2: Thermodynamically consistent activity coefficients of aqueous electrolyte mixtures and organic–electrolyte coupling* |

Every surrogate predicts an **excess Gibbs energy**, and all activity coefficients follow from it by differentiation.
The Gibbs–Duhem equation therefore holds exactly, and the surrogates can be used inside Gibbs-energy-minimization
(phase-equilibrium) solvers such as `aiomfac_py.phase_equilibrium.PhaseEquilibrium`.

All material for both papers is in [`excess_gibbs/`](excess_gibbs/). Its README maps every table and figure to the
script and result file that produce it; all result files and trained networks are included, so results can be
inspected without re-running.

## What the repository contains

- **Part 1 (binary water–organic):** label generation for BIMOG and MCM molecules, the comparison of molecular
  representations (subgroup counts, hand-crafted descriptors, learned subgroup embedding, graph neural networks and
  hybrids), training pools and coverage, learning curves on the Master Chemical Mechanism (MCM) pool, the comparison of
  output forms (unconstrained versus excess Gibbs energy), binary liquid–liquid phase separation, phase equilibrium of
  α-pinene oxidation products, sensitivity analyses, cost and literature compounds. **Release ensemble:** three
  excess-Gibbs-energy networks trained on 3705 BIMOG + MCM molecules.
- **Part 2 (electrolytes and organic–electrolyte coupling):** free-species training data (water and 14 ions), the
  comparison of output forms, the speciation code for the bisulfate and carbonate equilibria, end-to-end validation
  (including Fortran reference output), bisulfate speciation, cost, sensitivity analysis through the surrogate versus
  AIOMFAC, phase equilibrium of inorganic particles, and the organic–electrolyte coupling through a learned cross term
  of the excess Gibbs energy, with worked examples and the comparison with experimental mean activity coefficients.
  **Release ensembles:** five electrolyte networks and three cross-term networks.

## Quick start

```bash
git clone https://github.com/SeoulTechPSE/aiomfac-surrogates.git
cd aiomfac-surrogates
pip install -r requirements.txt
cd part1_organic_surrogate && python fetch_mcm_species.py   # MCM species list, needed to regenerate the Part 1 labels
```

All scripts are run from their own folder, for example

```bash
cd excess_gibbs/code
S2_TAG=s2v2 python s2_part2.py e2e fort      # Part 2: end-to-end and Fortran comparisons
python s3_coupling.py                         # Part 2: organic–electrolyte coupling
```

See [`excess_gibbs/README.md`](excess_gibbs/README.md) for all scripts and for how to use the release ensembles.

## Data sources

- **BIMOG** (`part1_organic_surrogate/bimog_unified.json`): 368 organic compounds from the Bielefeld Molecular
  Organic Glasses database (Armeli Iapichino et al., 2023, https://doi.org/10.5281/zenodo.7319485). Please cite
  the database when using these data.
- **MCM v3.3.1** species list: downloaded from https://mcm.york.ac.uk/MCM by `fetch_mcm_species.py`; it is not
  redistributed here. Please cite the MCM website and Jenkin et al. (1997) and Saunders et al. (2003).
- **AIOMFAC labels** (`excess_gibbs/results/s1_labels.npz`, `s2v2_labels.npz`): generated with `aiomfac_py` by the
  scripts in `excess_gibbs/code/`.
- **Nominal-composition test data** (`part2_inorganic_surrogate/training_data.parquet`) and **Fortran reference
  output** (`part2_inorganic_surrogate/fortran_reference/`, cases c005 and c006 from the instrumented Fortran AIOMFAC
  build used to validate `aiomfac_py`).
- **Lopes et al. (1999)**: only the published Pitzer parameters and Debye–Hückel coefficients (their Tables II
  and III) are used, written directly into `excess_gibbs/code/s3_coupling.py`.

The folders `part1_organic_surrogate/` and `part2_inorganic_surrogate/` hold these input data and the material of the
earlier versions of the papers (release v2.0.1), whose surrogates predicted the activity coefficients directly.

## Environment and run times

The results were produced on an Apple-silicon laptop (CPU only, PyTorch 2.14, `aiomfac_py` 1.1.0). Training one
network takes from seconds (Part 1, BIMOG split) to about 20 minutes (Part 2 electrolyte network on one CPU thread);
the analyses take from seconds to about 15 minutes each. Results are seeded, but bit-identical reproduction across
platforms and library versions is not guaranteed.

## Changes

See [CHANGELOG.md](CHANGELOG.md).

## License

GPL-3.0-or-later (see `LICENSE`), matching `aiomfac_py`, which this repository depends on.

## Citation

Please cite the relevant paper (Part 1 and/or Part 2), this repository (Zenodo; v3.0.1:
https://doi.org/10.5281/zenodo.23203148, all versions: https://doi.org/10.5281/zenodo.23122610), and `aiomfac_py`
(v1.1.0: https://doi.org/10.5281/zenodo.23202730) together with the AIOMFAC papers (Zuend et al., 2008, 2011).
