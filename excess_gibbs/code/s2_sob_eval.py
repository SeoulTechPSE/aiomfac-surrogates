"""Effect of the Sobolev-type derivative loss on the electrolyte G^E surrogate (GEXNet, "gex" vs "gex_sob").

Same split and protocol as s2_train.py (Table 1 of Part 2); for each seed available in both variants:
  * value errors (ln gamma of the ions present, ln a_w) and the Gibbs-Duhem residual (s2_train.evaluate);
  * errors of the scaled derivatives d/d ln m_j of ln a_w and ln gamma_i along fixed random directions among the species
    present, against AIOMFAC (s2_dlabels.py), held-out compositions and unseen ion pairs;
  * phase equilibrium of the five inorganic particles of Sect. 3.5 (60 states, single network, JAX Gibbs function)
    against AIOMFAC (from jax_pe_bench.json): phase state and particle water.
Writes (updates) ../results/s2v2_sob_eval.json.
Usage: python s2_sob_eval.py [seeds, e.g. 0,1] [variants, e.g. gex,gex_sob,gex_sob_l003]"""
import json
import os
import sys

os.environ.setdefault("S2_TAG", "s2v2")
import numpy as np
import torch

import s2_train as S  # noqa: E402

from aiomfac_py.phase_equilibrium import PhaseEquilibrium  # noqa: E402

import jax_surrogates as JS  # noqa: E402
import s2_pe_gex as P2  # noqa: E402


def split():
    HOLD = [("Ca2+", "CO3--"), ("NH4+", "HSO4-"), ("K+", "IO3-")]
    held = np.zeros(len(S.M), bool)
    for a_, b_ in HOLD:
        held |= (S.M[:, S.NAMES.index(a_)] > 0) & (S.M[:, S.NAMES.index(b_)] > 0)
    rest = np.flatnonzero(~held)
    perm = np.random.RandomState(123).permutation(rest)
    nte = int(0.15 * len(rest))
    return perm[:nte], np.flatnonzero(held)


def load(kind, seed):
    net = S.GEXNet().float()                    # float32 as trained (s2_pe_gex sets the torch default to float64)
    net.load_state_dict(torch.load(f"../results/s2v2_models/{kind}_seed{seed}.pt"))
    return net.float().eval()


def deriv_mae(net, idx, n_max=4000, neutral=False):
    """MAE of the directional derivatives (water, ions) along one fixed random direction per composition (float64,
    central difference 1e-4 in ln m), against the AIOMFAC derivative labels."""
    rng = np.random.RandomState(1)
    idx = idx[np.isfinite(S.DL[idx]).any((1, 2))][:n_max]
    netd = S.GEXNet()
    netd.load_state_dict(net.state_dict())
    netd = netd.double().eval()
    m = torch.tensor(S.M[idx]); tn = torch.tensor((S.T[idx] - 293.15) / 20)
    A = torch.tensor(S.AT[idx]); b = torch.tensor(S.BT[idx]); pr = torch.tensor(S.PRES[idx])
    v = torch.tensor(rng.normal(size=S.M[idx].shape)) * pr
    if neutral:
        v = S.neutral_directions(v, m)
    dw, di = S.directional(netd, m, tn, A, b, v, h=1e-4, create_graph=False)
    yw, yi, okw, oki = S.directional_labels(torch.tensor(S.DL[idx], dtype=torch.float64), v)
    oki = oki & pr
    dw, di = dw.detach().numpy(), di.detach().numpy()
    return {"mae_dlnaw": float(np.abs(dw - yw.numpy())[okw.numpy()].mean()),
            "mae_dlng_ion": float(np.abs(di - yi.numpy())[oki.numpy()].mean()),
            "median_rel_dlng_ion": float(np.median(np.abs(di - yi.numpy())[oki.numpy()]
                                                   / (np.abs(yi.numpy())[oki.numpy()] + 1e-3)))}


def pe(net, ref):
    gf = JS.part2_gibbs([net.double()])
    out = {"same_state": 0, "n": 0, "water_rel_err": [], "converged_or_dry": 0, "water_rel_err_max_by_case": {}}
    for name, (ions, feed) in P2.CASES.items():
        for rh, a in zip(P2.RH, ref[name]["aiomfac"]):
            r = PhaseEquilibrium([], ions, 298.15, liquid_model=JS.part2_liquid(ions, None, gf)).solve(feed, rh)
            out["n"] += 1
            out["converged_or_dry"] += r.status in ("converged", "dry")
            out["same_state"] += (r.n_liquids, sorted(r.solids)) == (a["n_liquids"], a["solids"])
            w = float(sum(L.amounts[0] for L in r.liquids))
            if a["water"] > 0 and w > 0:
                out["water_rel_err"].append(abs(w / a["water"] - 1))
                c = out["water_rel_err_max_by_case"]
                c[name] = max(c.get(name, 0.0), abs(w / a["water"] - 1))
    e = out.pop("water_rel_err")
    out["water_rel_err_median"], out["water_rel_err_max"] = float(np.median(e)), float(np.max(e))
    return out


if __name__ == "__main__":
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")]
    te, tu = split()
    ref = json.load(open("../results/jax_pe_bench.json"))["part2"]
    kinds = (sys.argv[2] if len(sys.argv) > 2 else "gex,gex_sob").split(",")
    out_path = "../results/s2v2_sob_eval.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for kind in kinds:
        for seed in seeds:
            path = f"../results/s2v2_models/{kind}_seed{seed}.pt"
            if not os.path.exists(path):
                continue
            net = load(kind, seed)
            r = {**S.evaluate(net, te, kind), **{k + "_unseen": v for k, v in S.evaluate(net, tu, kind).items()},
                 **deriv_mae(net, te), **{k + "_unseen": v for k, v in deriv_mae(net, tu).items()},
                 **{k + "_neutral": v for k, v in deriv_mae(net, te, neutral=True).items()},
                 **{k + "_neutral_unseen": v for k, v in deriv_mae(net, tu, neutral=True).items()}, **pe(load(kind, seed), ref)}
            res[f"{kind}_seed{seed}"] = r
            print(kind, seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
