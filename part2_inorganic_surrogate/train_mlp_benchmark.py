"""
Step 3 of the inorganic-aerosol-surrogate roadmap: MLP benchmark model.

Trains "Model A" from the design doc (§모델 아키텍처 제안): a plain 3-hidden-layer
MLP (128 units each, ReLU, Adam, early stopping) mapping the fixed 15-dim input
vector (14 ion molalities + T) to the 15-dim output vector (ln_gamma of H2O +
the 14 component ions), using the Step-2 dataset (tools/generate_training_data.py).

Evaluates (§평가 지표):
  - interpolation MAE   : held-out points, but the same ion-combination seen in training
  - generalization MAE  : ion combinations held out ENTIRELY from training
  - Fortran cross-validation proxy: aiomfac_py's own ActivityModel IS the Fortran-
    validated ground truth (see README), so this script re-evaluates the trained
    MLP against it on the multi-salt / CaSO4(s)-precipitation reference points used
    in Step 1, which is the same role tests/reference/carb_sulf/ plays for aiomfac_py
    itself (no Fortran binary is available in this environment to call directly).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).parent
AIOMFAC_SRC = Path(__file__).resolve().parent  # helper modules live next to this file
sys.path.insert(0, str(AIOMFAC_SRC / "tools"))
from inorganic_component_basis import CATIONS, ANIONS  # noqa: E402

CATION_NAMES = [n for n, _, _ in CATIONS]
ANION_NAMES = [n for n, _, _ in ANIONS]
ALL_IONS = CATION_NAMES + ANION_NAMES
IN_COLS = [f"m_{n}" for n in ALL_IONS] + ["T_K"]
OUT_COLS = ["ln_gamma_H2O"] + [f"ln_gamma_{n}" for n in ALL_IONS]

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Data loading + split
# ---------------------------------------------------------------------------

HELD_OUT_COMBOS = [frozenset({"Ca2+", "CO3--"}), frozenset({"NH4+", "HSO4-"}), frozenset({"K+", "IO3-"})]


def active_ions(row) -> frozenset:
    return frozenset(n for n in ALL_IONS if row[f"m_{n}"] > 0)


def load_and_split(path: Path):
    df = pd.read_parquet(path)
    df = df[df[OUT_COLS[1:]].notna().any(axis=1)].reset_index(drop=True)  # need >=1 valid ion target
    combos = df.apply(active_ions, axis=1)
    is_generalization = combos.isin(HELD_OUT_COMBOS)
    df_gen = df[is_generalization].reset_index(drop=True)
    df_rest = df[~is_generalization].reset_index(drop=True)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df_rest))
    n_val = int(0.1 * len(df_rest))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_interp = df_rest.iloc[val_idx].reset_index(drop=True)
    return df_train, df_interp, df_gen


def featurize(df: pd.DataFrame, mu=None, sd=None):
    X = np.log1p(df[IN_COLS[:-1]].to_numpy(dtype=np.float32))  # log1p(molality)
    T = df["T_K"].to_numpy(dtype=np.float32)[:, None]
    X = np.concatenate([X, T], axis=1)
    if mu is None:
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xs = (X - mu) / sd
    Y = df[OUT_COLS].to_numpy(dtype=np.float32)
    mask = np.isfinite(Y).astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.0)
    return Xs, Y, mask, mu, sd


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom


def masked_mae_per_col(pred, target, mask):
    diff = np.abs(pred - target) * mask
    denom = mask.sum(axis=0)
    denom[denom == 0] = np.nan
    return diff.sum(axis=0) / denom


def train(df_train, df_interp, df_gen, epochs=400, patience=25):
    Xtr, Ytr, Mtr, mu, sd = featurize(df_train)
    Xva, Yva, Mva, _, _ = featurize(df_interp, mu, sd)
    Xge, Yge, Mge, _, _ = featurize(df_gen, mu, sd) if len(df_gen) else (None, None, None, mu, sd)

    model = MLP(Xtr.shape[1], Ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t, Ytr_t, Mtr_t = map(torch.from_numpy, (Xtr, Ytr, Mtr))
    Xva_t, Yva_t, Mva_t = map(torch.from_numpy, (Xva, Yva, Mva))

    best_val, best_state, bad = math.inf, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = masked_mse(pred, Ytr_t, Mtr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = masked_mse(model(Xva_t), Yva_t, Mva_t).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                break
        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d}  train_loss={loss.item():.5f}  val_loss={val_loss:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_interp = model(Xva_t).numpy()
        pred_train = model(Xtr_t).numpy()
        pred_gen = model(torch.from_numpy(Xge)).numpy() if Xge is not None else None

    mae_interp = masked_mae_per_col(pred_interp, Yva, Mva)
    mae_train = masked_mae_per_col(pred_train, Ytr, Mtr)
    mae_gen = masked_mae_per_col(pred_gen, Yge, Mge) if Xge is not None else None
    return model, (mu, sd), dict(train=mae_train, interp=mae_interp, gen=mae_gen), best_val


def fortran_crossval(model, scaler):
    """Compare the trained MLP against aiomfac_py (Fortran-validated) directly, on the
    multi-salt + CaSO4(s)-precipitation case from Step 1 (same role as tests/reference/carb_sulf/)."""
    from aiomfac_py import ActivityModel
    from aiomfac_py.io import Component

    water = Component(1, "Water", ((16, 1),))
    ca_io3 = Component(2, "Ca(IO3)2", ((221, 1), (246, 2)))
    na_hso4 = Component(3, "NaHSO4", ((202, 1), (248, 1)))
    na_hco3 = Component(4, "NaHCO3", ((202, 1), (250, 1)))
    model_aio = ActivityModel([water, ca_io3, na_hso4, na_hco3])
    res = model_aio.evaluate([0.85, 0.03, 0.06, 0.06], 298.15, basis="mass")

    sr = model_aio.mixture.sr
    nn_, nc = sr.n_neutral, sr.n_cation
    truth = {"ln_gamma_H2O": float(res.ln_gamma[0])}
    ion_sub = {n: sub for n, sub, _ in CATIONS + ANIONS}
    for n in ALL_IONS:
        sub = ion_sub[n]
        if sub in sr.cations:
            truth[f"ln_gamma_{n}"] = float(res.ln_gamma[nn_ + sr.cations.index(sub)])
        elif sub in sr.anions:
            truth[f"ln_gamma_{n}"] = float(res.ln_gamma[nn_ + nc + sr.anions.index(sub)])

    # Build this point's *input* molalities (mirrors generate_training_data.py's bookkeeping)
    from generate_training_data import _build_system, ION_MW  # noqa: F401
    totals = {n: 0.0 for n in ALL_IONS}
    totals["Ca2+"] += 1 * 0.03 / (0.03 * (40.078e-3 + 2 * 174.903e-3))  # placeholder, overwritten below

    # simpler: recompute molalities directly from the known stoichiometry & mass basis
    import numpy as _np
    sg_water_frac = 0.85
    comp_fracs = {"Ca(IO3)2": 0.03, "NaHSO4": 0.06, "NaHCO3": 0.06}
    mw = {"Ca(IO3)2": 40.078e-3 + 2 * 174.903e-3, "NaHSO4": 22.99e-3 + 97.071e-3, "NaHCO3": 22.99e-3 + 61.016e-3}
    totals = {n: 0.0 for n in ALL_IONS}
    for name, frac in comp_fracs.items():
        molal = (frac / sg_water_frac) / mw[name]   # mol per kg water
        if name == "Ca(IO3)2":
            totals["Ca2+"] += molal; totals["IO3-"] += 2 * molal
        elif name == "NaHSO4":
            totals["Na+"] += molal; totals["HSO4-"] += molal
        elif name == "NaHCO3":
            totals["Na+"] += molal; totals["HCO3-"] += molal

    row = pd.DataFrame([{**{f"m_{n}": totals[n] for n in ALL_IONS}, "T_K": 298.15}])
    mu, sd = scaler
    X = np.log1p(row[IN_COLS[:-1]].to_numpy(dtype=np.float32))
    X = np.concatenate([X, row[["T_K"]].to_numpy(dtype=np.float32)], axis=1)
    Xs = (X - mu) / sd
    with torch.no_grad():
        pred = model(torch.from_numpy(Xs)).numpy()[0]

    print("\n=== Fortran cross-validation proxy (aiomfac_py multi-salt + CaSO4(s) precip case) ===")
    print(f"{'species':10s} {'aiomfac_py (truth)':>20s} {'MLP pred':>12s} {'abs err':>10s}")
    errs = []
    for i, col in enumerate(OUT_COLS):
        if col in truth:
            t = truth[col]
            p = float(pred[i])
            if t == -9999.9:
                # Ca2+ fully consumed by CaSO4(s) precipitation in this case -- not a regression
                # target (same convention as tools/generate_training_data.py); report the prediction
                # but exclude it from the error average.
                print(f"{col:10s} {'(precipitated, excl.)':>20s} {p:12.4f} {'--':>10s}")
                continue
            e = abs(t - p)
            errs.append(e)
            print(f"{col:10s} {t:20.4f} {p:12.4f} {e:10.4f}")
    print(f"mean abs error over present (non-precipitated) species: {np.mean(errs):.4f}")
    return errs


def main():
    data_path = HERE / "training_data.parquet"
    df_train, df_interp, df_gen = load_and_split(data_path)
    print(f"train={len(df_train)}  interp-test={len(df_interp)}  generalization-test={len(df_gen)}")
    print(f"held-out combos: {[set(c) for c in HELD_OUT_COMBOS]}")

    model, scaler, maes, best_val = train(df_train, df_interp, df_gen)
    print(f"\nbest val (masked MSE) = {best_val:.5f}")

    rows = []
    for col in OUT_COLS:
        i = OUT_COLS.index(col)
        rows.append({
            "species": col,
            "train_MAE": maes["train"][i],
            "interp_MAE": maes["interp"][i],
            "generalization_MAE": maes["gen"][i] if maes["gen"] is not None else np.nan,
        })
    table = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:8.4f}")
    print("\n=== Table: per-species MAE (log10... no, natural-log gamma units) ===")
    print(table.to_string(index=False))

    overall = table[["train_MAE", "interp_MAE", "generalization_MAE"]].mean()
    print("\n=== Overall (mean across species) ===")
    print(overall.to_string())

    errs = fortran_crossval(model, scaler)

    out = {
        "n_train": len(df_train), "n_interp": len(df_interp), "n_generalization": len(df_gen),
        "held_out_combos": [sorted(c) for c in HELD_OUT_COMBOS],
        "per_species_mae": table.to_dict(orient="records"),
        "overall_mae": overall.to_dict(),
        "fortran_crossval_mean_abs_err": float(np.mean(errs)),
    }
    with open(HERE / "mlp_benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {HERE / 'mlp_benchmark_results.json'}")


if __name__ == "__main__":
    main()
