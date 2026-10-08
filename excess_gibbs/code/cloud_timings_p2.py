"""Part 2 Sect. 3.6 with the derivatives compiled once (JAX): the same release network (seed 0), molality vectors and
300 ion combinations as s2_part2.py cost, one evaluation and the end-to-end cost with speciation (speciation.speciate),
and agreement with the PyTorch model_fn.  Adds "part2_jax" to ../results/cloud_timings.json."""
import json
import os

os.environ.setdefault("S2_TAG", "s2v2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import torch

torch.set_num_threads(1)
import s2_part2 as P  # noqa: E402
import jax_surrogates as JS  # noqa: E402


def med(fn, n, warm=5):
    import time
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(np.median(t))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    combos = [P.gtd.sample_combo(rng) for _ in range(300)]                 # as in s2_part2.task_cost
    vec = lambda sm_T: np.array([P.gtd._build_system(*sm_T)[2][n] for n in P.NAMES])
    net = P.load("gex", 0, True)
    f_t = P.model_fn([net])
    f_j = JS.part2_activity_fn(JS.part2_gibbs([net]))
    dev = 0.0
    for s in combos[:50]:
        a_t, g_t = f_t(vec(s), s[1]); a_j, g_j = f_j(vec(s), s[1])
        dev = max(dev, abs(a_t - a_j), float(np.max(np.abs(g_t - g_j))))
    m1 = vec(combos[0])
    t_one = med(lambda: f_j(m1, 298.15), 500)
    t_e2e = float(np.median([med(lambda s=s: P.SPC.speciate(f_j, vec(s), s[1]), 1, warm=1) for s in combos]))
    d = json.load(open("../results/cloud_timings.json"))
    d["part2_jax"] = {"one_s": t_one, "e2e_vary_s": t_e2e, "max_abs_diff_vs_torch": dev}
    json.dump(d, open("../results/cloud_timings.json", "w"), indent=1)
    c = json.load(open("../results/P2GE_cost.json"))
    print({k: (1e6 * v if k.endswith("_s") else v) for k, v in d["part2_jax"].items()})
    print(f"varying combination: one at a time {c['vary'] / (c['vec'] + t_one):.1f}x, with speciation {c['vary'] / t_e2e:.1f}x;"
          f" fixed system one at a time {c['fixed'] / (c['vec'] + t_one):.1f}x")
