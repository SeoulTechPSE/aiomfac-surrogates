# Part 2 — Aqueous electrolyte mixtures and organic–electrolyte coupling

Reproduces the tables and figures of *Neural-network surrogates for AIOMFAC – Part 2: Aqueous electrolyte
mixtures and organic–electrolyte coupling*.

## Setup

```bash
pip install -r ../requirements.txt
```

Run `01_part2_revision.ipynb` with this folder as the working directory (about 2 minutes on an Apple-silicon
laptop, CPU only). The coupling sections load the Part 1 release checkpoint from
`../part1_organic_surrogate/checkpoints/`; set the environment variable `CKPT_DIR` to use another location.

## Contents

| Item | Description |
|---|---|
| `01_part2_revision.ipynb` | Full analysis (executed, outputs saved) |
| `training_data.parquet` | 12 000 electrolyte compositions with AIOMFAC labels |
| `generate_training_data.py` | Generator of `training_data.parquet` (four sampling axes) |
| `inorganic_component_basis.py` | The 14-ion basis and input/output specification |
| `closed_form_pipeline.py` | Closed-form bisulfate equilibrium (Eqs. 1–2) and its checks |
| `train_mlp_benchmark.py` | v1 training script (imported for shared definitions; the v1 protocol is replicated in the notebook) |
| `organics.json` | The 243 BIMOG CHO organics used for the coupling |
| `fortran_reference/` | Fortran AIOMFAC reference inputs and dumps for cases c005 and c006, with `reference_utils.py` |
| `models/` | The five electrolyte-surrogate networks (corrected protocol, seeds 0–4) with input normalization |
| `results/`, `figures/` | Outputs written by the notebook |
| `legacy_v1/` | v1 notebooks (superseded; see its README) |

## Where each result comes from

| Paper item | Notebook section |
|---|---|
| Table 1, Fig. 1 | I.1–I.2 (corrected and replicated v1 protocol, 5 seeds) |
| Table 2 | I.3 (Fortran reference comparison) |
| Sect. 3.3 | I.4 (closed-form bisulfate check) |
| Sect. 3.4 | I.5 (end-to-end cost) |
| Sect. 3.5, Fig. 2 | I.6 (Sobol analysis, surrogate vs. AIOMFAC) |
| Table 3 | II.2 (Option A error decomposition) |
| Table 4, Fig. 3 | II.3 and III (Option C, OOD guard, per-organic effects) |
| Table 5, Fig. 4 | II.4 (worked examples) |
| Sect. 3.9, Fig. 5 | IV (comparison with Lopes et al., 1999) |

## Using the electrolyte surrogate

Each file in `models/` stores the network weights (`state_dict` of a 15→128→128→128→15 MLP whose output is
de-standardized by the stored `ym`/`ys` buffers), the input normalization (`x_mean`, `x_std`), and the column names.
The inputs are log(1 + m) of the 14 ion molalities (mol per kg water), in the order of `in_cols`, followed by
temperature in K. The outputs are ln γ of water and of the 14 ions, in the order of `out_cols`. See
`Surrogate`, `MLP` and `predict` in section I.2 of the notebook.
