# Part 2 — Inorganic surrogate and organic-inorganic combination

Two notebooks, run in order:

1. **`01_inorganic_surrogate_notebook.ipynb`** — Sections 2-3 of the paper: the canonical-form
   component-basis construction, the 14-ion fixed-vector MLP surrogate, the closed-form bisulfate
   dissociation pipeline (validated against both a self-consistent oracle and real Fortran reference data),
   the speed/accuracy benchmark, and the Sobol global sensitivity analysis. Self-contained via
   `inorganic_surrogate_bundle.zip` (library source + training data + helper scripts), which the notebook
   finds/unzips automatically.
2. **`organic_inorganic_combination/02_combination_notebook.ipynb`** — Section 4: loads the inorganic
   surrogate trained in notebook 1 *and* the organic surrogate trained in
   [Part 1](../../part1_organic_surrogate/), and evaluates Option A (independent sum), Option C (residual
   correction), and the distance-based out-of-distribution guard for the organic correction. Self-contained
   via `combo_bundle.zip` (both trained surrogates + the inorganic component-basis code).

## Contents

- `01_inorganic_surrogate_notebook.ipynb`, `inorganic_surrogate_bundle.zip`
- `organic_inorganic_combination/02_combination_notebook.ipynb`, `organic_inorganic_combination/combo_bundle.zip`
- `figs/fig1_inorganic_parity.png` — Figure 1 (inorganic-only surrogate accuracy).
- `organic_inorganic_combination/figs/` — Figures 2-3 (Option A vs. guarded Option C; OOD guard calibration).

## Running

Both notebooks are self-contained: open either in Jupyter or Colab and run all cells. Each looks for its
bundle zip (or an already-unzipped copy) in its own working directory and unzips/installs automatically if
needed — no manual setup required. Both have been executed end-to-end with zero error cells; outputs are
saved in the `.ipynb` files.
