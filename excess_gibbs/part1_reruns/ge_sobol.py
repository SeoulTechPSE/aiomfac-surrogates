"""Sect. 3.7 (i) with a G^E head: descriptor-space Sobol analysis through an R_desc surrogate (notebook 04 cells 6, 7,
9).  The surrogate is trained as in notebook 04 (BIMOG, main groups of BIMOG, seed 0, Table-1 split) but with the
G^E head.  For Saltelli samples, which are descriptor vectors rather than molecules, the molar mass needed to convert
w_water to x_org is taken from the sampled ln M."""
import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from ge_common import *
from ge_common import _sg

isB = SRC == "bimog"
bids = np.where(isB)[0]
MGB = sorted({_sg.main_group(int(s)) for i in bids for s, q in POOL[i]["subgroups"]})
featB = np.array([desc(POOL[i], MGB) for i in bids], np.float32)
loc = -np.ones(len(POOL), int); loc[bids] = np.arange(len(bids))
FEATS["R_descB"] = np.zeros((len(POOL), featB.shape[1]), np.float32); FEATS["R_descB"][bids] = featB
testB = TEST[bids]
tr_all = pts_of(isB & ~TEST)
pred, ep = fit_ge("R_descB", tr_all, 0, interp_split=True)
g = pts_of(isB & TEST); err = np.abs(pred(PM[g], EXTRA[g]) - Y[g]).mean(0)
print(f"surrogate trained ({ep} epochs); held-out MAE water={err[0]:.3f} organic={err[1]:.3f}", flush=True)
emb, head, fm, fs = pred.modules

FUNC_GROUPS = {3: "frac_aromatic_ACH", 69: "frac_hydroxyl_OH", 52: "frac_polyol_CH", 20: "frac_carboxylic_COOH",
               13: "frac_ether_CH2O", 9: "frac_ketone_CH2CO", 11: "frac_ester_CCOO", 15: "frac_amine_CNH"}
nmg = len(MGB)
raw_caps = {}
for gr in FUNC_GROUPS:
    col = featB[:, MGB.index(gr)]; nz = col[col > 0]
    raw_caps[gr] = min(float(np.percentile(nz, 95)), 0.5) if len(nz) else 0.1
caps = {gr: v * 0.9 / sum(raw_caps.values()) for gr, v in raw_caps.items()}
names = list(FUNC_GROUPS.values()) + ["o_to_c", "n_to_c", "log_molar_mass", "log1p_total_q", "wtf_water", "T_norm"]
pct = lambda j, q: float(np.percentile(featB[:, nmg + j], q))
bounds = ([[0.0, caps[gr]] for gr in FUNC_GROUPS] + [[0.0, pct(0, 95)], [0.0, pct(1, 95)], [pct(2, 5), pct(2, 95)],
          [pct(3, 5), pct(3, 95)], [0.02, 0.98], [(263.15 - 293.15) / 20, (313.15 - 293.15) / 20]])
problem = {"num_vars": len(names), "names": names, "bounds": bounds}


def surrogate(X_sa):
    Fm = np.zeros((len(X_sa), featB.shape[1]), np.float32)
    for k, gr in enumerate(FUNC_GROUPS):
        Fm[:, MGB.index(gr)] = X_sa[:, k]
    Fm[:, MGB.index(1)] = 1 - X_sa[:, :len(FUNC_GROUPS)].sum(1)
    Fm[:, nmg:nmg + 4] = X_sa[:, len(FUNC_GROUPS):len(FUNC_GROUPS) + 4]
    w, tn = X_sa[:, -2], X_sa[:, -1]
    M = np.exp(X_sa[:, len(FUNC_GROUPS) + 2]) * 1e-3
    out = []
    for s in range(0, len(X_sa), 50000):
        sl = slice(s, s + 50000)
        f = torch.tensor(((Fm[sl] - fm) / fs).astype(np.float32))
        x = torch.tensor(x_of_w(w[sl], M[sl]).astype(np.float32))
        out.append(head(f, x, torch.tensor(M[sl].astype(np.float32)), torch.tensor(tn[sl].astype(np.float32)),
                        create_graph=False).detach().numpy())
    return np.concatenate(out)


rows = {}
for N in (1024, 4096):
    X_sa = sobol_sample.sample(problem, N, calc_second_order=False, seed=0)
    P = surrogate(X_sa)
    for k, out in enumerate(["ln_gamma_water", "ln_gamma_organic"]):
        Si = sobol_analyze.analyze(problem, P[:, k].astype(float), calc_second_order=False, seed=0)
        for i, nm in enumerate(names):
            rows[(out, nm, N)] = (Si["S1"][i], Si["S1_conf"][i], Si["ST"][i], Si["ST_conf"][i])
    if N == 1024:
        X1024 = X_sa
gsa = pd.DataFrame(rows, index=["S1", "S1_conf", "ST", "ST_conf"]).T
gsa.index.names = ["output", "factor", "N"]
gsa.to_csv("results/GE_sobol_descriptor_indices.csv")
tab = gsa.xs(4096, level="N")
for out in ["ln_gamma_water", "ln_gamma_organic"]:
    t = tab.loc[out].sort_values("ST", ascending=False)
    t["ST_N1024"] = gsa.xs((out, 1024), level=("output", "N")).loc[t.index, "ST"]
    print(f"\n{out} (N=4096):\n" + t.round(3).to_string(), flush=True)
mol_cols = slice(0, nmg + 4)
F_train = featB[~testB][:, mol_cols]
mu, sd = F_train.mean(0), F_train.std(0); sd = np.where(sd < 1e-6, 1, sd)


def nn_dist(F):
    Z, Zt = (F - mu) / sd, (F_train - mu) / sd
    return np.array([np.min(np.linalg.norm(Zt - z, axis=1)) for z in Z])


def to_feat(X_sa):
    Fm = np.zeros((len(X_sa), featB.shape[1]), np.float32)
    for k, gr in enumerate(FUNC_GROUPS):
        Fm[:, MGB.index(gr)] = X_sa[:, k]
    Fm[:, MGB.index(1)] = 1 - X_sa[:, :len(FUNC_GROUPS)].sum(1)
    Fm[:, nmg:nmg + 4] = X_sa[:, len(FUNC_GROUPS):len(FUNC_GROUPS) + 4]
    return Fm


d_real = nn_dist(featB[testB][:, mol_cols]); d_sob = nn_dist(to_feat(X1024[:4000])[:, mol_cols]); thr = np.percentile(d_real, 95)
print(f"NN distance: real held-out median {np.median(d_real):.2f} (95th pct {thr:.2f}); Sobol samples median "
      f"{np.median(d_sob):.2f}; beyond 95th pct: {100*(d_sob > thr).mean():.0f}%")
print("DONE sobol")
