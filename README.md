# aiomfac-surrogates

Reproducibility material for a two-part paper series on neural-network surrogates for
[AIOMFAC](https://github.com/andizuend/AIOMFAC) activity-coefficient calculations, built on
[`aiomfac_py`](https://github.com/SeoulTechPSE/aiomfac-python), a pure-Python port of AIOMFAC validated against the
Fortran reference model.

| | Paper | Folder |
|---|---|---|
| **Part 1** | *Neural-network surrogates for AIOMFAC – Part 1: Binary water–organic activity coefficients, molecular representation and data scale* | [`part1_organic_surrogate/`](part1_organic_surrogate/) |
| **Part 2** | *Neural-network surrogates for AIOMFAC – Part 2: Aqueous electrolyte mixtures and organic–electrolyte coupling* | [`part2_inorganic_surrogate/`](part2_inorganic_surrogate/) |

Each folder has its own README, which maps every table and figure of the paper to the notebook section that
produces it. All notebooks are stored **with their outputs**, so results can be inspected without re-running.

## What the repository contains

- **Part 1:** notebooks for the representation comparison (raw subgroup counts, hand-crafted descriptors, learned
  subgroup embedding R_emb, graph neural network), the training-pool and coverage analyses, learning curves on the
  Master Chemical Mechanism (MCM) pool, the BAT baseline, the viscosity emulator, the sensitivity analyses
  (descriptor space, real molecules, AIOMFAC interaction parameters), the cost benchmark and the
  literature-compound check. The folder also contains the **release checkpoint**: an R_emb ensemble trained on
  3705 BIMOG + MCM molecules, with a ready-to-use prediction script.
- **Part 2:** one notebook for the electrolyte surrogate and its validation (including Fortran reference output),
  the bisulfate closed-form check, cost, sensitivity analysis through the surrogate versus AIOMFAC, the
  organic–electrolyte coupling (Option A/C) and the comparison with experimental mean activity coefficients. It
  also includes the five trained electrolyte-surrogate networks.

## Quick start

```bash
git clone https://github.com/SeoulTechPSE/aiomfac-surrogates.git
cd aiomfac-surrogates
pip install -r requirements.txt
cd part1_organic_surrogate && python fetch_mcm_species.py   # MCM species list, needed by most Part 1 notebooks
```

Then open any notebook in Jupyter and run all cells, with the notebook's own folder as the working directory.

To use the release surrogate directly:

```bash
cd part1_organic_surrogate/checkpoints
python predict_remb.py "OC(=O)CC(=O)O" 0.5 298.15     # SMILES, water mass fraction, temperature (K)
```

## Data sources

- **BIMOG** (`part1_organic_surrogate/bimog_unified.json`): 368 organic compounds from the Bielefeld Molecular
  Organic Glasses database (Armeli Iapichino et al., 2023, https://doi.org/10.5281/zenodo.7319485). Please cite
  the database when using these data.
- **MCM v3.3.1** species list: downloaded from https://mcm.york.ac.uk/MCM by `fetch_mcm_species.py`; it is not
  redistributed here. Please cite the MCM website and Jenkin et al. (1997) and Saunders et al. (2003).
- **Electrolyte training data** (`part2_inorganic_surrogate/training_data.parquet`): 12 000 compositions
  generated with `aiomfac_py` by `generate_training_data.py`.
- **Fortran reference output** (`part2_inorganic_surrogate/fortran_reference/`): cases c005 and c006 from the
  instrumented Fortran AIOMFAC build used to validate `aiomfac_py`.
- **Lopes et al. (1999)**: only the published Pitzer parameters and Debye–Hückel coefficients (their Tables II
  and III) are used, written directly into the Part 2 notebook.

## Environment and run times

The stored outputs were produced on an Apple-silicon laptop (CPU only, PyTorch 2.14, `aiomfac_py` 0.0.24). Run
times on that machine are listed in each folder's README; they range from about a minute to about an hour per
notebook. Results are seeded, but bit-identical reproduction across platforms and library versions is not
guaranteed.

## Changes since v1.1.0

See [CHANGELOG.md](CHANGELOG.md).

## License

GPL-3.0-or-later (see `LICENSE`), matching `aiomfac_py`, which this repository depends on.

## Citation

Please cite the relevant paper (Part 1 and/or Part 2), this repository (Zenodo; v2.0.0: https://doi.org/10.5281/zenodo.23137531, all versions:
https://doi.org/10.5281/zenodo.23122610), and `aiomfac_py` (v1.0.1: https://doi.org/10.5281/zenodo.23137735) together with
the AIOMFAC papers (Zuend et al., 2008, 2011).
