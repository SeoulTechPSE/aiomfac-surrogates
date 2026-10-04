"""Retrains the Part 1 organic surrogate with a DENSE, FIXED water-mass-fraction grid for
composition sampling, replacing the main notebook's `build_dataset()` 20-randomly-jittered-
bins-per-molecule scheme with a 200-point grid shared identically across every molecule (no
per-molecule jitter).

Why: the delivered `trained_model_delivered/` checkpoint shows small non-monotonic "wiggles"
in its predicted ln(gamma_organic) between training points, most visibly for 1,2,4-butanetriol
(see Sec. 8 / paper Sec. 3.11 in the main notebook -- on a fine 400-point evaluation grid, the
delivered checkpoint's curve has 7 slope-sign-changes versus 2 for the true AIOMFAC curve). The
cause is composition-axis training-data sparsity: only 20 samples per molecule gives an
unconstrained feedforward MLP too little composition-axis supervision to rule out non-monotonic
interpolants between samples, especially for molecules with high curvature.

This is "denser/fixed sampling" rather than an explicit curvature penalty on the training loss,
chosen because it fixes the actual information deficiency without imposing a global smoothness
prior that could suppress genuine curvature elsewhere in the 368-molecule pool.

Otherwise identical architecture/features/training procedure to the main notebook's own Part I
cells (same MLPHead, same rich_features, same SHA-256-hash molecule split): the only change is
build_dataset()'s composition sampling (n_comp 20->200, fixed shared grid instead of per-molecule
random jitter within bins) and batch size (256->1024, for the ~10x larger dataset). n_temp stays
at 4 -- temperature was not implicated in the wiggling.

Output checkpoint (surrogate_weights.pt / surrogate_norm.npz / surrogate_meta.json) is delivered
alongside this script as `trained_model_dense/`, used only in the main notebook's Sec. 8 /
paper Sec. 3.11 worked-example check -- every other result in the paper uses
`trained_model_delivered/` (this script's un-modified sibling training procedure).

Usage: run from this directory (`part1_organic_surrogate/`) -- it expects `bimog_unified.json`
next to it, same as the main notebook.
"""
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

from aiomfac_py.s2as import smiles_to_components
from aiomfac_py import ActivityModel, Component
from aiomfac_py.params import load_subgroup_params

HERE = Path(__file__).parent
OUT_DIR = HERE / "trained_model_dense"

torch.manual_seed(0)
np.random.seed(0)
sg = load_subgroup_params()


def is_test_molecule(smiles):
    h = int(hashlib.sha256(smiles.encode()).hexdigest(), 16)
    return (h % 100) < 15


def bimog_category(d):
    has_n, has_hal = d["#N"] != 0, d["#Hal"] != 0
    if has_n and has_hal:
        return "bimog_n_and_hal"
    if has_n:
        return "bimog_n_only"
    if has_hal:
        return "bimog_hal_only"
    return "cho"


def decompose_pool(candidates):
    kept = []
    for d in candidates:
        smi = d["smiles"]
        r = smiles_to_components([smi], water_as_component1=True)
        if r.removed:
            continue
        organic = r.components[1]
        try:
            m = ActivityModel(r.components)
            res = m.evaluate([0.9, 0.1], 298.15, basis="mass")
            if not (res.gamma_neutral > 0).all():
                raise ValueError
        except Exception:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol_h = Chem.AddHs(mol)
        n_c = sum(1 for a in mol_h.GetAtoms() if a.GetSymbol() == "C")
        n_o = sum(1 for a in mol_h.GetAtoms() if a.GetSymbol() == "O")
        n_n = sum(1 for a in mol_h.GetAtoms() if a.GetSymbol() == "N")
        mw = sum(a.GetMass() for a in mol_h.GetAtoms())
        kept.append({"name": d["name"], "smiles": smi, "category": d["category"],
                     "subgroups": organic.subgroups, "molar_mass_g_mol": mw,
                     "o_to_c": n_o / max(n_c, 1), "n_to_c": n_n / max(n_c, 1)})
    return kept


def build_dataset(kept, seed=0, n_comp=200, n_temp=4):
    """Dense, FIXED composition grid shared by every molecule (no per-molecule random jitter),
    crossed with n_temp random temperatures per composition point."""
    rng = np.random.RandomState(seed)
    wtf_water_grid = np.linspace(0.02, 0.98, n_comp)
    rows = []
    for oi, o in enumerate(kept):
        water = Component(1, "Water", ((16, 1),))
        organic = Component(2, o["name"], tuple((s, q) for s, q in o["subgroups"]))
        try:
            m = ActivityModel([water, organic])
        except Exception:
            continue
        temps = rng.uniform(273.15, 313.15, n_temp)
        for wtf_w in wtf_water_grid:
            for T in temps:
                try:
                    res = m.evaluate([wtf_w, 1.0 - wtf_w], T, basis="mass")
                    lw, lo = float(res.ln_gamma[0]), float(res.ln_gamma[1])
                    if np.isfinite(lw) and np.isfinite(lo):
                        rows.append({"organic_idx": oi, "wtf_water": float(wtf_w), "T_K": float(T),
                                     "ln_gamma_water": lw, "ln_gamma_organic": lo})
                except Exception:
                    continue
    return rows


def rich_features(kept, main_groups):
    mg_index = {g: i for i, g in enumerate(main_groups)}
    feat = []
    for o in kept:
        mg_counts = np.zeros(len(main_groups))
        total_q = 0
        for s, q in o["subgroups"]:
            g = sg.main_group(int(s))
            if g in mg_index:
                mg_counts[mg_index[g]] += q
            total_q += q
        mg_frac = mg_counts / max(total_q, 1)
        feat.append(np.concatenate([mg_frac, [o["o_to_c"], o["n_to_c"], np.log(o["molar_mass_g_mol"]),
                                               np.log1p(total_q)]]))
    return np.array(feat, dtype=np.float32)


class MLPHead(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def to_t(a):
    return torch.tensor(a, dtype=torch.float32)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("loading BIMOG pool and decomposing (full 368, CHO + N + halogen)...")
    t0 = time.time()
    with open(HERE / "bimog_unified.json") as f:
        bimog = json.load(f)
    cands = [{"smiles": d["smiles"], "name": d["name"], "category": bimog_category(d)} for d in bimog]
    kept = decompose_pool(cands)
    print(f"  kept {len(kept)} / {len(bimog)} molecules ({time.time()-t0:.0f}s)")

    vocab = sorted({int(s) for o in kept for s, q in o["subgroups"]})
    main_groups = sorted({sg.main_group(s) for s in vocab})
    print(f"  {len(main_groups)} main groups in vocabulary: {main_groups}")

    print("generating labeled (molecule, composition, T) -> ln(gamma) dataset (dense grid)...")
    t0 = time.time()
    rows = build_dataset(kept)
    feat = rich_features(kept, main_groups)
    organic_idx = np.array([r["organic_idx"] for r in rows])
    X_comp_T = np.array([[r["wtf_water"], (r["T_K"] - 293.15) / 20.0] for r in rows], dtype=np.float32)
    y = np.array([[r["ln_gamma_water"], r["ln_gamma_organic"]] for r in rows], dtype=np.float32)
    X = np.concatenate([feat[organic_idx], X_comp_T], axis=1)
    print(f"  {len(rows)} points ({time.time()-t0:.0f}s)")

    test_mols = {oi for oi, o in enumerate(kept) if is_test_molecule(o["smiles"])}
    train_mask = ~np.isin(organic_idx, list(test_mols))
    gen_mask = np.isin(organic_idx, list(test_mols))
    train_idx_all = np.where(train_mask)[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(train_idx_all))
    n_interp = int(0.2 * len(train_idx_all))
    interp_idx = train_idx_all[perm[:n_interp]]
    fit_idx = train_idx_all[perm[n_interp:]]
    gen_idx = np.where(gen_mask)[0]
    n_val = int(0.1 * len(fit_idx))
    val_sel, fit_sel = np.arange(n_val), np.arange(n_val, len(fit_idx))

    x_mean, x_std = X[fit_idx].mean(0), X[fit_idx].std(0) + 1e-6
    y_mean, y_std = y[fit_idx].mean(0), y[fit_idx].std(0) + 1e-6
    Xn, Yn = (X - x_mean) / x_std, (y - y_mean) / y_std
    X_fit_t, Y_fit_t = to_t(Xn[fit_idx]), to_t(Yn[fit_idx])

    model = MLPHead(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    print("training surrogate MLP (dense grid, batch size 1024)...")
    t0 = time.time()
    epoch = 0
    for epoch in range(500):
        model.train()
        perm_e = np.random.permutation(fit_sel)
        for s in range(0, len(perm_e), 1024):
            idx = perm_e[s:s + 1024]
            opt.zero_grad()
            loss = loss_fn(model(X_fit_t[idx]), Y_fit_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(X_fit_t[val_sel]), Y_fit_t[val_sel]).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 30:
                break
    model.load_state_dict(best_state)
    print(f"  done ({time.time()-t0:.0f}s, {epoch+1} epochs)")

    def mae(idx_set):
        if len(idx_set) == 0:
            return None
        model.eval()
        with torch.no_grad():
            pred = model(to_t(Xn[idx_set])).numpy() * y_std + y_mean
        ytrue = y[idx_set]
        return float(np.mean(np.abs(pred[:, 0] - ytrue[:, 0]))), float(np.mean(np.abs(pred[:, 1] - ytrue[:, 1])))

    mi, mg_ = mae(interp_idx), mae(gen_idx)
    print(f"sanity check -- interpolation MAE water={mi[0]:.4f} organic={mi[1]:.4f}")
    print(f"sanity check -- generalization MAE water={mg_[0]:.4f} organic={mg_[1]:.4f}")

    torch.save(model.state_dict(), OUT_DIR / "surrogate_weights.pt")
    np.savez(OUT_DIR / "surrogate_norm.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    with open(OUT_DIR / "surrogate_meta.json", "w") as f:
        json.dump({"main_groups": main_groups, "n_features": int(X.shape[1]),
                   "feature_names": [f"mg_frac_{g}" for g in main_groups] +
                                     ["o_to_c", "n_to_c", "log_molar_mass", "log1p_total_q",
                                      "wtf_water", "T_norm"],
                   "sanity_interp_mae": {"water": mi[0], "organic": mi[1]},
                   "sanity_gen_mae": {"water": mg_[0], "organic": mg_[1]},
                   "n_organics": len(kept), "n_points": len(rows)}, f, indent=1)
    print(f"\nwrote surrogate_weights.pt, surrogate_norm.npz, surrogate_meta.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
