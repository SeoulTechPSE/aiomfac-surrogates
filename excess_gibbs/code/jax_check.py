"""JAX transcription (jax_surrogates.py) versus the torch implementation (surrogate_pe.py, s2_pe_gex.py):
(1) identical activities and Hessians, (2) cost per call of ln a, the Hessian and one Hessian-vector product.

Per-call times are medians over repeated single-point calls after warm-up (one CPU thread), the setting of the
phase-equilibrium solver.  Writes ../results/jax_check.json."""
import json
import os
import time

os.environ.setdefault("S2_TAG", "s2v2")                 # Part 2 label set of the release ensemble
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

from aiomfac_py.phase_equilibrium import ExplicitLiquidModel  # noqa: E402
from aiomfac_py.s2as import smiles_to_components  # noqa: E402

import jax_surrogates as JS  # noqa: E402
import s1_train as S1  # noqa: E402
import s2_pe_gex as P2  # noqa: E402
from surrogate_pe import GEBinaryLiquidModel  # noqa: E402

T = 289.0
CAND = {"pinonaldehyde": "CC(=O)C1CC(C=O)C1(C)C", "pinonic_acid": "CC(=O)C1CC(C1(C)C)CC(=O)O",
        "pinic_acid": "CC1(C(CC1C(=O)O)CC(=O)O)C", "terpenylic_acid": "OC(=O)C1(C)CCC(=O)C1CC(=O)O",
        "MBTCA": "OC(=O)CC(C(=O)O)C(C)(C(=O)O)C"}
SYSTEMS = {"(NH4)2SO4": ["NH4+", "SO4--"], "NaCl": ["Na+", "Cl-"], "NH4NO3": ["NH4+", "NO3-"],
           "NH4HSO4": ["NH4+", "H+", "SO4--"], "NaCl + (NH4)2SO4": ["Na+", "NH4+", "Cl-", "SO4--"]}


def timeit(f, reps=300):
    for _ in range(5):
        f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6                                   # microseconds


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = {"part1": {}, "part2": {}}
    vocab = [int(v) for v in np.load("../results/s1_labels.npz")["vocab"]]
    nets1 = []
    for seed in range(3):
        net = S1.Net(len(vocab), "ge", "silu")
        net.load_state_dict(torch.load(f"../results/s1_models/ge_silu_sob_release_seed{seed}.pt")); net.eval()
        nets1.append(net.double())
    for name, smi in CAND.items():
        org = smiles_to_components([smi], names=[name]).components[1]
        cnt = np.zeros(len(vocab))
        for s, q in org.subgroups:
            cnt[vocab.index(int(s))] = q
        M = float(ExplicitLiquidModel([org], [])._mm[1])
        lt = GEBinaryLiquidModel(org, nets1, cnt, M)
        lj = JS.part1_liquid(org, nets1, cnt)
        da = dh = 0.0
        for _ in range(50):
            n = rng.uniform(0.01, 1.0, 2)
            da = max(da, rel(lj.ln_a(n, T), lt.ln_a(n, T)))
            dh = max(dh, rel(lj.hessian_ad(n, T), lt.hessian_ad(n, T)))
        n = np.array([0.6, 0.4]); v = np.array([0.3, -0.2])
        out["part1"][name] = {"max_rel_diff_ln_a": da, "max_rel_diff_hessian": dh,
                              "us_ln_a_torch": timeit(lambda: lt.ln_a(n, T)), "us_ln_a_jax": timeit(lambda: lj.ln_a(n, T)),
                              "us_hess_torch": timeit(lambda: lt.hessian_ad(n, T), 100),
                              "us_hess_jax": timeit(lambda: lj.hessian_ad(n, T)), "us_hvp_jax": timeit(lambda: lj.hvp(n, T, v))}
        print("P1", name, {k: f"{v:.3g}" for k, v in out["part1"][name].items()}, flush=True)
    nets2 = P2.load("gex", release=True)
    gf = JS.part2_gibbs(nets2)
    for name, ions in SYSTEMS.items():
        lt = P2.SurrogateElectrolyteLiquid(ions, nets2, "gex")
        lj = JS.part2_liquid(ions, nets2, gf)
        lt.set_conditions(298.15, np.log(0.7)); lj.set_conditions(298.15, np.log(0.7))
        z = lj.z
        da = dh = 0.0
        for _ in range(30):
            n = np.concatenate([[rng.uniform(5, 50)], rng.uniform(0.1, 2.0, lj.N - 1)])
            pos, neg = z > 0, z < 0                                             # electroneutral composition
            n[neg] *= float(np.dot(z[pos], n[pos]) / -np.dot(z[neg], n[neg]))
            da = max(da, rel(lj.ln_a(n, 298.15), lt.ln_a(n, 298.15)))
            dh = max(dh, rel(lj.hessian_ad(n, 298.15), lt.hessian_ad(n, 298.15)))
        v = rng.normal(size=lj.N)
        out["part2"][name] = {"N": int(lj.N), "max_rel_diff_ln_a": da, "max_rel_diff_hessian": dh,
                              "us_ln_a_torch": timeit(lambda: lt.ln_a(n, 298.15), 100),
                              "us_ln_a_jax": timeit(lambda: lj.ln_a(n, 298.15)),
                              "us_hess_torch": timeit(lambda: lt.hessian_ad(n, 298.15), 30),
                              "us_hess_jax": timeit(lambda: lj.hessian_ad(n, 298.15)),
                              "us_hvp_jax": timeit(lambda: lj.hvp(n, 298.15, v))}
        print("P2", name, {k: f"{v:.3g}" for k, v in out["part2"][name].items()}, flush=True)
    json.dump(out, open("../results/jax_check.json", "w"), indent=1)
