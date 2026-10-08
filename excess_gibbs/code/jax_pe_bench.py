"""Phase-equilibrium runs of the papers with the excess-Gibbs-energy surrogates: torch autograd at every call
(surrogate_pe.GEPhaseEquilibrium, s2_pe_gex.SurrogatePE, as used for the papers) versus the jit-compiled JAX
transcription (jax_surrogates.py), and AIOMFAC, on the same machine and one CPU thread.

  part1: five alpha-pinene products, water + organic at 289 K, RH scan 0.95-0.9999 (55 states; s1_a2.py)
  part2: five inorganic particles at 298.15 K, RH 0.95-0.30 (60 states; s2_pe_gex.py)

Writes ../results/jax_pe_bench.json.  Usage: python jax_pe_bench.py [part1] [part2]"""
import json
import os
import sys
import time

os.environ.setdefault("S2_TAG", "s2v2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

from aiomfac_py.phase_equilibrium import ExplicitLiquidModel, PhaseEquilibrium  # noqa: E402
from aiomfac_py.s2as import smiles_to_components  # noqa: E402

import jax_surrogates as JS  # noqa: E402
import s1_train as S1  # noqa: E402
import s2_pe_gex as P2  # noqa: E402
from surrogate_pe import GEPhaseEquilibrium  # noqa: E402

from jax_check import CAND  # noqa: E402

SCAN = [0.95, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.9985, 0.999, 0.9995, 0.9999]


def run(make, feed, rh, solids, org=False):
    pe = make()
    t0 = time.perf_counter()
    r = pe.solve(feed, rh, solids=solids)
    dt = time.perf_counter() - t0
    return {"rh": rh, "status": r.status, "n_liquids": r.n_liquids, "solids": sorted(r.solids),
            "water": float(sum(L.amounts[0] for L in r.liquids)),
            "x_org": [float(L.mole_fractions[1]) for L in r.liquids] if org else None,
            "n_eval": int(pe.lm.n_eval), "n_jac": int(getattr(pe.lm, "n_jac", 0)), "time_s": dt}


def part1(out):
    T = 289.0
    vocab = [int(v) for v in np.load("../results/s1_labels.npz")["vocab"]]
    nets = []
    for seed in range(3):
        net = S1.Net(len(vocab), "ge", "silu")
        net.load_state_dict(torch.load(f"../results/s1_models/ge_silu_sob_release_seed{seed}.pt")); net.eval()
        nets.append(net.double())
    res = {}
    for name, smi in CAND.items():
        org = smiles_to_components([smi], names=[name]).components[1]
        cnt = np.zeros(len(vocab))
        for s, q in org.subgroups:
            cnt[vocab.index(int(s))] = q
        M = float(ExplicitLiquidModel([org], [])._mm[1])
        gf = JS.part1_gibbs(nets, cnt, name)
        makers = {"aiomfac": lambda: PhaseEquilibrium([org], [], T_K=T),
                  "torch": lambda: GEPhaseEquilibrium(org, T, nets, cnt, M),
                  "jax": lambda: PhaseEquilibrium([org], [], T_K=T, liquid_model=JS.GibbsLiquidModel([org], [], gf))}
        warm = run(makers["jax"], {name: 1.0}, 0.95, "none")              # first call: jit compilation
        res[name] = {tag: [run(mk, {name: 1.0}, rh, "none", org=True) for rh in SCAN] for tag, mk in makers.items()}
        res[name]["jax_first_solve_s"] = warm["time_s"]
        print(name, {t: round(sum(p["time_s"] for p in v), 2) for t, v in res[name].items() if isinstance(v, list)},
              "first jax solve", round(res[name]["jax_first_solve_s"], 2), flush=True)
    out["part1"] = res


def part2(out):
    T = 298.15
    nets = P2.load("gex", release=True)
    gf = JS.part2_gibbs(nets)
    res = {}
    for name, (ions, feed) in P2.CASES.items():
        makers = {"aiomfac": lambda: PhaseEquilibrium([], ions, T),
                  "torch": lambda: P2.SurrogatePE(ions, T, nets, "gex"),
                  "jax": lambda: PhaseEquilibrium([], ions, T, liquid_model=JS.part2_liquid(ions, nets, gf))}
        warm = run(makers["jax"], feed, 0.9, "all")                       # first call: jit compilation
        res[name] = {tag: [run(mk, feed, rh, "all") for rh in P2.RH] for tag, mk in makers.items()}
        res[name]["jax_first_solve_s"] = warm["time_s"]
        print(name, {t: round(sum(p["time_s"] for p in v), 2) for t, v in res[name].items() if isinstance(v, list)},
              "first jax solve", round(res[name]["jax_first_solve_s"], 2), flush=True)
    out["part2"] = res


if __name__ == "__main__":
    parts = sys.argv[1:] or ["part1", "part2"]
    path = "../results/jax_pe_bench.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    for p in parts:
        {"part1": part1, "part2": part2}[p](out)
        json.dump(out, open(path, "w"), indent=1)
