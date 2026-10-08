"""Timings of the papers that are not produced by ge_cost.py (Part 1) or s2_part2.py cost (Part 2), measured in the
same session on the same machine (one CPU thread):

  machine      platform, CPU, cores, memory, software versions                       -> cloud_machine.json
  jax_one      the Part 1 release network (seed 0, as in ge_cost.py) evaluated one point at a time with the
               derivatives compiled once (JAX, aiomfac_py.gibbs_model): both activity coefficients per call
  multicomp    one AIOMFAC evaluation of water + 1 ... 40 MCM organics (Part 1 Sect. 3.9)
  lle          short-range AIOMFAC evaluations and time of one LLE calculation, 1-butanol + water (Sect. 3.9)
  anova        the composition-temperature grid of the real-molecule variance decomposition (10 w x 5 T per molecule)
               over the MCM molecules of the release pool (Part 1 Sect. 3.8)

The MCM molecules come from results/s1_pool.json (3337 MCM molecules of the release pool, with their AIOMFAC
subgroups); the MCM species list itself (fetch_mcm_species.py) is not needed.  Writes ../results/cloud_timings.json."""
import json
import os
import platform
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import torch

torch.set_num_threads(1)

from aiomfac_py import ActivityModel, Component, lle  # noqa: E402
import aiomfac_py.model as _model_mod  # noqa: E402
from aiomfac_py.s2as import smiles_to_components  # noqa: E402

import s1_train as S  # noqa: E402
import jax_surrogates as JS  # noqa: E402

WATER = Component(1, "Water", ((16, 1),))
OUT = {}


def med(fn, n, warm=5):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(np.median(t))


def machine():
    import jax
    import scipy
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    mem = ""
    try:
        mem = f"{int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemTotal'))) / 2**20:.1f} GiB"
    except (OSError, StopIteration):
        pass
    return {"platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(), "memory": mem,
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "jax": jax.__version__, "threads_used": 1}


if __name__ == "__main__":
    OUT["machine"] = machine()
    print(OUT["machine"], flush=True)
    # ---- JAX, one point at a time (same network as ge_cost.py: release seed 0)
    net = S.Net(S.CNT.shape[1], "ge", "silu")
    net.load_state_dict(torch.load("../results/s1_models/ge_silu_sob_release_seed0.pt")); net.eval()
    pool = json.load(open("../results/s1_pool.json"))["pool"]
    vocab = [int(v) for v in np.load("../results/s1_labels.npz")["vocab"]]
    rng = np.random.RandomState(0)
    mols = [pool[i] for i in rng.choice(len(pool), 200, replace=False)]
    t_jax = []
    for o in mols[:20]:
        cnt = np.zeros(len(vocab))
        for s, q in o["subgroups"]:
            cnt[vocab.index(int(s))] = q
        org = Component(2, "org", tuple(map(tuple, o["subgroups"])))
        lm = JS.part1_liquid(org, [net.double()], cnt)
        pts = rng.uniform(0.01, 0.99, 200)
        lm.ln_a(np.array([0.5, 0.5]), 298.15)                              # compile (first molecule only)
        t_jax.append(med(lambda: [lm.ln_a(np.array([1 - x, x]), 298.15) for x in pts[:1]], 300))
    OUT["jax_one_us"] = 1e6 * float(np.median(t_jax))
    print("JAX one at a time:", round(OUT["jax_one_us"], 1), "us", flush=True)
    # ---- AIOMFAC cost vs number of organics (MCM molecules of the release pool)
    mcm = [o["subgroups"] for o in pool if o["src"] == "mcm"]
    rng = np.random.RandomState(1)
    first = mcm[:400]
    OUT["multicomp_us"] = {}
    for n_org in (1, 2, 5, 10, 20, 40):
        picks = [first[i] for i in rng.choice(len(first), n_org, replace=False)]
        m = ActivityModel([WATER] + [Component(k + 2, f"o{k}", tuple(map(tuple, p))) for k, p in enumerate(picks)])
        x = np.full(n_org + 1, 1.0 / (n_org + 1))
        OUT["multicomp_us"][n_org] = 1e6 * med(lambda: m.evaluate(x, 298.15, basis="mass"), 200)
    print("multicomponent:", {k: round(v, 1) for k, v in OUT["multicomp_us"].items()}, flush=True)
    # ---- LLE call count, 1-butanol + water
    calls = {"n": 0}
    orig = _model_mod.sr_terms

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    _model_mod.sr_terms = counting
    try:
        comps = smiles_to_components(["CCCCO"]).components
        t0 = time.perf_counter(); lle.solve_pep(comps, np.array([0.5, 0.5]), 298.15)
        OUT["lle"] = {"sr_evaluations": calls["n"], "ms": 1e3 * (time.perf_counter() - t0)}
    finally:
        _model_mod.sr_terms = orig
    print("LLE:", OUT["lle"], flush=True)
    # ---- real-molecule variance decomposition grid (Sect. 3.8): 10 w x 5 T per molecule
    W_GRID, T_GRID = np.linspace(0.05, 0.95, 10), np.linspace(273.15, 313.15, 5)
    mcm_all = [o for o in pool if o["src"] == "mcm"]
    n_ok = 0
    t0 = time.time()
    for o in mcm_all:
        m = ActivityModel([WATER, Component(2, "o", tuple(map(tuple, o["subgroups"])))])
        for w in W_GRID:
            for T in T_GRID:
                try:
                    m.evaluate([w, 1 - w], T, basis="mass"); n_ok += 1
                except Exception:                                           # noqa: BLE001
                    pass
    dt = time.time() - t0
    OUT["anova"] = {"molecules": len(mcm_all), "evaluations": n_ok, "s": dt,
                    "s_scaled_to_3409": dt * 3409 / len(mcm_all)}
    print("ANOVA grid:", OUT["anova"], flush=True)
    json.dump(OUT, open("../results/cloud_timings.json", "w"), indent=1)
