# aiomfac-surrogates

Reproducibility material for a two-part paper series on fast neural-network surrogates for
[AIOMFAC](https://github.com/andizuend/AIOMFAC) activity-coefficient calculations, built on
[`aiomfac_py`](https://github.com/SeoulTechPSE/aiomfac-python) (K.-Y. Yoo's pure-Python port of AIOMFAC).

| | Paper | Notebook(s) |
|---|---|---|
| **Part 1** | *A Fast Neural-Network Surrogate for AIOMFAC Activity Coefficients: Feature Design, Functional-Group Coverage Limits, and Downstream Applications to Viscosity Prediction and Global Sensitivity Analysis* | [`part1_organic_surrogate/01_organic_surrogate_notebook.ipynb`](part1_organic_surrogate/01_organic_surrogate_notebook.ipynb) |
| **Part 2** | *Inorganic Aerosol Surrogate Model: A Fixed-Vector Neural Surrogate for Electrolyte Activity Coefficients via the Canonical-Form Component Basis* | [`part2_inorganic_surrogate/01_inorganic_surrogate_notebook.ipynb`](part2_inorganic_surrogate/01_inorganic_surrogate_notebook.ipynb) (inorganic surrogate, Sections 2-3), then [`part2_inorganic_surrogate/organic_inorganic_combination/02_combination_notebook.ipynb`](part2_inorganic_surrogate/organic_inorganic_combination/02_combination_notebook.ipynb) (combines it with the Part 1 organic surrogate, Section 4) |

Part 2 builds on Part 1: it reuses the organic surrogate trained in Part 1's notebook (shipped as a checkpoint
in `part1_organic_surrogate/trained_model_delivered/`) and composes it with its own inorganic surrogate.

## Dependency

All three notebooks require `aiomfac_py` (the validated Python port of AIOMFAC itself, a separate repository
with its own release/citation — see [SeoulTechPSE/aiomfac-python](https://github.com/SeoulTechPSE/aiomfac-python)).
Each notebook is self-contained: on first run it looks for a local `aiomfac-python/` checkout or the
accompanying `*_bundle.zip` next to it, and falls back to `git clone` + `pip install` (Colab-friendly) if
neither is present. No manual setup is required beyond opening the notebook and running all cells.

## Status

Every notebook in this repository has been executed end-to-end with zero error cells (outputs are saved
in the `.ipynb` files themselves, so results are visible without re-running). The Part 1 notebook's
CPU-only speed-benchmark figures were additionally cross-checked on a Google Colab GPU runtime (NVIDIA T4);
see the paper for those numbers.

## License

Code in this repository is released under GPL-3.0-or-later (see `LICENSE`), matching `aiomfac_py`,
which it depends on but does not redistribute.

## Citation

If you use this code, please cite the relevant paper (Part 1 and/or Part 2; see each paper's own
"Code and data availability" section) and `aiomfac_py` (see that repository's README for the AIOMFAC
citation chain).
