"""Phase equilibrium of particles with the new ions (Li+, Mg2+, Br-) through the extended electrolyte surrogate
(s4_extend.py, frozen mode, release ensemble of five networks) against AIOMFAC, with aiomfac_py's PhaseEquilibrium as in
Part 2 Sect. 3.5 (298.15 K, 12 relative humidities, solids of the solver's database).  The surrogate enters as a
JAX Gibbs function (jax_surrogates.part2x_gibbs); a self-check compares it with the torch network first.
Writes ../results/s4_pe.json."""
import json
import os
import time

os.environ.setdefault("S2_TAG", "s2v2")
import numpy as np
import torch

import s4_extend as X
import jax_surrogates as JS
from aiomfac_py.phase_equilibrium import PhaseEquilibrium
from aiomfac_py.gibbs_model import GibbsLiquidModel

CASES = {"MgCl2": (["Mg++", "Cl-"], {"Mg++": 1.0, "Cl-": 2.0}),
         "MgSO4": (["Mg++", "SO4--"], {"Mg++": 1.0, "SO4--": 1.0}),
         "Mg(NO3)2": (["Mg++", "NO3-"], {"Mg++": 1.0, "NO3-": 2.0}),
         "NaCl + MgCl2": (["Na+", "Mg++", "Cl-"], {"Na+": 1.0, "Mg++": 0.5, "Cl-": 2.0}),
         "NaBr": (["Na+", "Br-"], {"Na+": 1.0, "Br-": 1.0}),
         "LiCl": (["Li+", "Cl-"], {"Li+": 1.0, "Cl-": 1.0})}
RH = [0.95, 0.9, 0.85, 0.82, 0.8, 0.78, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3]
T = 298.15


def load_release(k=5):
    nets = []
    for s in range(k):
        net = X.make("frozen", s, release=True)
        net.load_state_dict(torch.load(f"../results/s4_models/frozen_nall_release_seed{s}.pt"))
        nets.append(net.double().eval())
    return nets


def self_check(nets, gf):
    """ln a of water and ions: JAX Gibbs function vs the torch ensemble (mean of G -> mean of derivatives)"""
    idx = np.concatenate([X.NEW_TE[:20], X.OLD_TE[:10]])
    m, tn, A, b = X.tens(idx, torch.float64)[:4]
    pa = np.mean([n(m, tn, A, b, create_graph=False)[0].detach().numpy() for n in nets], 0)
    pi = np.mean([n(m, tn, A, b, create_graph=False)[1].detach().numpy() for n in nets], 0)
    dev = 0.0
    for r, k in enumerate(idx):
        kk = gf.constants(float(X.TA[k]))
        mm = X.MA[k]
        dw, di = gf.dG(1.0 / X.MW, mm, kk)
        lnxw = -np.log1p(X.MW * mm.sum())
        jw = lnxw + float(dw); ji = lnxw + np.asarray(di) - np.asarray(kk["ref"])
        p = mm > 0
        dev = max(dev, abs(jw - pa[r]), float(np.max(np.abs(ji[p] - pi[r][p]))))
    return dev


if __name__ == "__main__":
    nets = load_release()
    gf = JS.part2x_gibbs(nets, X.NAMES17, X.Z17)
    dev = self_check(nets, gf)
    print("JAX vs torch max |d ln a| =", dev, flush=True)
    out = {"jax_vs_torch_max": dev, "cases": {}}
    for name, (ions, feed) in CASES.items():
        rows = []
        lmx = GibbsLiquidModel([], ions, gf)
        for rh in RH:
            t0 = time.time(); a = PhaseEquilibrium([], ions, T).solve(feed, rh); ta = time.time() - t0
            t0 = time.time(); s = PhaseEquilibrium([], ions, T, liquid_model=lmx).solve(feed, rh); ts = time.time() - t0
            wa = float(sum(L.amounts[0] for L in a.liquids)); ws = float(sum(L.amounts[0] for L in s.liquids))
            rows.append({"rh": rh, "aiomfac": {"status": a.status, "n_liquids": a.n_liquids, "solids": sorted(a.solids),
                                              "water": wa, "s": ta},
                         "surrogate": {"status": s.status, "n_liquids": s.n_liquids, "solids": sorted(s.solids),
                                       "water": ws, "s": ts},
                         "same_state": (a.n_liquids, sorted(a.solids)) == (s.n_liquids, sorted(s.solids))})
            print(name, rh, rows[-1]["same_state"], f"{wa:.4g} {ws:.4g}", sorted(a.solids), sorted(s.solids), flush=True)
        out["cases"][name] = rows
        json.dump(out, open("../results/s4_pe.json", "w"), indent=1)
    allr = [r for v in out["cases"].values() for r in v]
    e = [abs(r["surrogate"]["water"] / r["aiomfac"]["water"] - 1) for r in allr
         if r["aiomfac"]["water"] > 0 and r["surrogate"]["water"] > 0]
    out["summary"] = {"n": len(allr), "same_state": sum(r["same_state"] for r in allr),
                      "water_rel_err_median": float(np.median(e)), "water_rel_err_max": float(np.max(e)),
                      "time_aiomfac_s": sum(r["aiomfac"]["s"] for r in allr),
                      "time_surrogate_s": sum(r["surrogate"]["s"] for r in allr)}
    json.dump(out, open("../results/s4_pe.json", "w"), indent=1)
    print(out["summary"])
