"""When does a Hessian-vector product (HVP) pay off against the full Hessian in a Newton iteration on a G^E surrogate?

Per call, one CPU thread, jit-compiled (after warm-up): gradient, full Hessian forward-over-reverse (jacfwd of grad),
full Hessian reverse-over-reverse (jacrev of grad), one HVP (jvp of grad), and the Newton direction H d = -g by
(a) the full Hessian + dense solve, (b) conjugate gradients with HVPs (k iterations).  Systems:
  * the Part 2 release ensemble (GEXNet, 5 members) for N = 3, 5 and 11 species;
  * a synthetic Gibbs function of the GEXNet form (one SiLU MLP 3 x 128 giving a symmetric Margules matrix) for
    N = 2 ... 40 species, as a stand-in for multicomponent surrogates.
Writes ../results/hvp_bench.json."""
import json
import os
import time

os.environ.setdefault("S2_TAG", "s2v2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax
import jax.numpy as jnp
import numpy as np

from aiomfac_py.gibbs_model import ideal_mixing

import jax_surrogates as JS
import s2_pe_gex as P2


def tm(f, reps=200):
    for _ in range(5):
        jax.block_until_ready(f())
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(f()); t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e6


def measure(g, n, k, cg_iters=(5, 10, 20)):
    grad = jax.jit(jax.grad(g))
    hf = jax.jit(jax.jacfwd(jax.grad(g)))
    hr = jax.jit(jax.jacrev(jax.grad(g)))
    gr_ = jax.grad(g)
    hvp = jax.jit(lambda n, k, v: jax.jvp(lambda m: gr_(m, k), (n,), (v,))[1])
    N = n.shape[0]
    v = jnp.ones(N) / N
    dense = jax.jit(lambda n, k: jnp.linalg.solve(hf(n, k) + 1e-9 * jnp.eye(N), -grad(n, k)))

    def cg(iters):
        @jax.jit
        def run(n, k):
            b = -grad(n, k)
            x = jnp.zeros(N); r = b; p = r; rr = r @ r
            def body(_, s):
                x, r, p, rr = s
                Ap = jax.jvp(lambda m: gr_(m, k), (n,), (p,))[1]
                a = rr / (p @ Ap); x = x + a * p; r = r - a * Ap
                rn = r @ r
                return x, r, r + rn / rr * p, rn
            return jax.lax.fori_loop(0, iters, body, (x, r, p, rr))[0]
        return run
    out = {"N": int(N), "grad_us": tm(lambda: grad(n, k)), "hess_fwd_rev_us": tm(lambda: hf(n, k)),
           "hess_rev_rev_us": tm(lambda: hr(n, k)), "hvp_us": tm(lambda: hvp(n, k, v)),
           "newton_dense_us": tm(lambda: dense(n, k))}
    for it in cg_iters:
        f = cg(it)
        out[f"newton_cg{it}_us"] = tm(lambda: f(n, k))
    return out


def synthetic(N, seed=0, hidden=128):
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(N)
    sizes = [N + 1, hidden, hidden, hidden, len(iu[0])]
    layers = [(jnp.asarray(rng.normal(size=(a, b)) / np.sqrt(a)), jnp.asarray(rng.normal(size=b) * 0.1))
              for a, b in zip(sizes[:-1], sizes[1:])]

    def g(n, k):
        x = n / jnp.sum(n)
        h = jnp.concatenate([jnp.log1p(n), k["tn"][None]])
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = h * jax.nn.sigmoid(h)
        H = jnp.zeros((N, N)).at[iu].set(0.1 * h)
        H = 0.5 * (H + H.T)
        return ideal_mixing(n) + jnp.sum(n) * x @ H @ x
    return g


if __name__ == "__main__":
    res = {"part2": [], "synthetic": []}
    nets = P2.load("gex", release=True)
    gf = JS.part2_gibbs(nets)
    for ions in (["Na+", "Cl-"], ["Na+", "NH4+", "Cl-", "SO4--"],
                 ["Na+", "K+", "NH4+", "Ca++", "H+", "Cl-", "NO3-", "SO4--", "I-"]):
        lm = JS.part2_liquid(ions, None, gf)
        idx = jnp.asarray(lm._idx)
        g = lambda n, k, idx=idx: gf.g(jnp.zeros(len(gf.names)).at[idx].set(n), k)
        n = jnp.asarray(np.concatenate([[30.0], np.full(lm.N - 1, 0.5)]))
        r = measure(g, n, gf.constants(298.15))
        res["part2"].append(r)
        print("part2", {k: round(v, 1) for k, v in r.items()}, flush=True)
    for N in (2, 5, 10, 20, 40):
        g = synthetic(N)
        n = jnp.asarray(np.linspace(1.0, 2.0, N))
        r = measure(g, n, {"tn": jnp.asarray(0.25)})
        res["synthetic"].append(r)
        print("synthetic", {k: round(v, 1) for k, v in r.items()}, flush=True)
    json.dump(res, open("../results/hvp_bench.json", "w"), indent=1)
