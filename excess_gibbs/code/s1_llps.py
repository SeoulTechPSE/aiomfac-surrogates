"""S1, phase behaviour of the test molecules: binary water-organic LLPS predicted by AIOMFAC and by the S1 surrogates
(BIMOG-trained, 5 seeds each) for the BIMOG hash-split test molecules, at 298.15 K and 273.15 K.

For each molecule and model: whether a miscibility gap exists, and if so the binodal (x_org of both liquids) and the
LLPS onset water activity a_w*.  Consistent models (AIOMFAC, ge_*) by the lower convex hull of
g(x) = (1-x) ln a_w + x ln a_o, refined by equal activities; the R_emb models (no Gibbs function) by equal activities
from the spinodal of a_w.  Writes ../results/s1_llps.json and prints the confusion counts and a_w* errors.
"""
import json
import math
import sys

import numpy as np
import torch
from scipy.optimize import fsolve

from aiomfac_py.phase_equilibrium import ExplicitLiquidModel
from aiomfac_py.io import Component

import s1_train as S

XS = np.unique(np.concatenate([np.logspace(-6, -1, 400), np.linspace(0.1, 0.999, 900)]))
TEMPS = (298.15, 273.15)
MODELS = sys.argv[1:] or ["remb_relu", "remb_silu", "ge_silu", "ge_silu_sob"]


def lower_hull(x, g):
    h = []
    for i in range(len(x)):
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (x[b] - x[a]) * (g[i] - g[a]) - (g[b] - g[a]) * (x[i] - x[a]) <= 0:
                h.pop()
            else:
                break
        h.append(i)
    return h


def binodal(lna_fn, consistent):
    """lna_fn(x array) -> (N, 2) ln a.  Returns (x1, x2, aw*) or None."""
    L = lna_fn(XS)
    def refine(a, b):
        def eqs(v):
            p, q = math.exp(v[0]), v[1]
            if not (0 < p < 1 and 0 < q < 1):
                return [1e3, 1e3]
            A = lna_fn(np.array([p, q]))
            return [A[0, 0] - A[1, 0], A[0, 1] - A[1, 1]]
        v, info, ier, msg = fsolve(eqs, [math.log(a), b], full_output=True, xtol=1e-12)
        if ier == 1 and math.exp(v[0]) < v[1] - 1e-4 and max(abs(np.array(eqs(v)))) < 1e-8:
            return math.exp(v[0]), float(v[1])
        return None
    sol = None
    if consistent:
        g = (1 - XS) * L[:, 0] + XS * L[:, 1]
        h = lower_hull(XS, g)
        gaps = [(XS[a], XS[b]) for a, b in zip(h[:-1], h[1:]) if b - a > 3 and XS[b] / XS[a] > 1.5]
        if gaps:
            a, b = max(gaps, key=lambda t: t[1] - t[0])
            sol = refine(a, b) or (float(a), float(b))
    else:
        rw = np.flatnonzero(np.gradient(L[:, 0], XS) > 0)
        if len(rw):
            s0, s1 = XS[rw[0]], XS[rw[-1]]
            for g1 in (s0 / 10, s0 / 3, s0 / 30, s0 / 100):
                for g2 in (min(0.95, s1 * 2), min(0.95, s1 * 1.3), 0.5, 0.8, 0.95):
                    sol = refine(g1, g2)
                    if sol:
                        break
                if sol:
                    break
            if sol is None:
                return "spinodal_only"
    if sol is None:
        return None
    return {"x1": sol[0], "x2": sol[1], "aw": float(math.exp(lna_fn(np.array([sol[0]]))[0, 0]))}


def aiomfac_fn(o, T):
    lm = ExplicitLiquidModel([Component(2, o["name"], tuple(map(tuple, o["subgroups"])))], [])
    def f(xs):
        return np.array([lm.ln_a(np.array([1 - x, x]), T)[:2] for x in xs])
    return f


def surr_fn(net, o, T):
    c = torch.tensor(S.CNT[S.POOL.index(o)])
    def f(xs):
        n = len(xs)
        x = torch.tensor(xs, dtype=torch.float32).requires_grad_(True)
        lg, _ = net(c.expand(n, -1), x, torch.full((n,), (T - 293.15) / 20), torch.full((n,), o["M_kg"]))
        lg = lg.detach().numpy().astype(float)
        return lg + np.log(np.column_stack([1 - xs, xs]))
    return f


if __name__ == "__main__":
    test_mols = [o for o in S.POOL if o["src"] == "bimog" and o["hash100"] < 15]
    nets = {}
    for m in MODELS:
        kind, act = ("remb", "relu") if m == "remb_relu" else ("remb", "silu") if m == "remb_silu" else ("ge", "silu")
        nets[m] = []
        for seed in range(5):
            net = S.Net(S.CNT.shape[1], kind, act)
            net.load_state_dict(torch.load(f"../results/s1_models/{m}_bimog_seed{seed}.pt")); net.eval()
            nets[m].append(net)
    OUT = []
    for o in test_mols:
        for T in TEMPS:
            r = {"name": o["name"], "T": T, "aiomfac": binodal(aiomfac_fn(o, T), True)}
            for m in MODELS:
                r[m] = [binodal(surr_fn(net, o, T), not m.startswith("remb")) for net in nets[m]]
            OUT.append(r)
    json.dump(OUT, open("../results/s1_llps.json", "w"), indent=1)


    def has(b):
        return isinstance(b, dict) or b == "spinodal_only"


    n_ref = sum(has(r["aiomfac"]) for r in OUT)
    print(f"{len(test_mols)} test molecules x {len(TEMPS)} temperatures; AIOMFAC LLPS in {n_ref} cases")
    for m in MODELS:
        tp = fp = fn = tn = 0
        daw = []
        for r in OUT:
            a = has(r["aiomfac"])
            for b in r[m]:
                s = has(b)
                tp += a and s; fp += (not a) and s; fn += a and (not s); tn += (not a) and (not s)
                if isinstance(r["aiomfac"], dict) and isinstance(b, dict):
                    daw.append(b["aw"] - r["aiomfac"]["aw"])
        daw = np.array(daw)
        print(f"{m:12s} TP {tp} FP {fp} FN {fn} TN {tn} | a_w* error: median |d| {np.median(np.abs(daw)) if len(daw) else float('nan'):.4f}"
              f", max |d| {np.max(np.abs(daw)) if len(daw) else float('nan'):.4f} (n={len(daw)})")
