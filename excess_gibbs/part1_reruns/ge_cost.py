"""Sect. 3.8, Table 6 and Appendix A (Run 1) for the G^E release network (research/paper_5 S1 release, seed 0).
The G^E surrogate needs dg/dx_org in addition to g; the NumPy implementation propagates value and derivative
together (forward mode through the SiLU layers), so one pass gives both activity coefficients.
Run on an otherwise idle machine, one CPU thread."""
import json
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch

torch.set_num_threads(1)
from aiomfac_py import ActivityModel
from aiomfac_py.s2as import smiles_to_components
import aiomfac_py.s2as._mapping as _m

_m.print = lambda *a, **k: None
import sys
_HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "code"))
os.chdir(_HERE.parent / "code")
import s1_train as S

net = S.Net(S.CNT.shape[1], "ge", "silu")
net.load_state_dict(torch.load("../results/s1_models/ge_silu_sob_release_seed0.pt")); net.eval()
n_par = sum(p.numel() for p in net.parameters())
Wemb = net.emb.weight.detach().numpy().astype(np.float64)
lin = [m for m in net.head if isinstance(m, torch.nn.Linear)]
Ws = [(m.weight.detach().numpy().astype(np.float32), m.bias.detach().numpy().astype(np.float32)) for m in lin]
em, es = net.em.numpy(), net.es.numpy(); hs = float(net.ys[1])


def timeit(fn, n):
    fn(); t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


def np_ge(e, x, tn):
    """e: (N, 16) molecular embedding; x, tn: (N,) -> (N, 2) ln gamma (value and dx through the network)."""
    a = np.concatenate([e, ((x - em[0]) / es[0])[:, None], ((tn - em[1]) / es[1])[:, None]], 1).astype(np.float32)
    da = np.zeros_like(a); da[:, -2] = 1.0 / es[0]
    for W, b in Ws[:-1]:
        u = a @ W.T + b; du = da @ W.T
        s = 1.0 / (1.0 + np.exp(-u))
        a = u * s; da = (s * (1.0 + u * (1.0 - s))) * du
    W, b = Ws[-1]
    h = (a @ W.T + b)[:, 0] * hs; dh = (da @ W.T)[:, 0] * hs
    g = x * (1 - x) * h; gp = (1 - 2 * x) * h + x * (1 - x) * dh
    return np.stack([g - x * gp, g + (1 - x) * gp], 1)


# check against autograd
rng = np.random.RandomState(0)
cnt = S.CNT[rng.randint(0, len(S.CNT), 200)].astype(np.float32)
x = rng.uniform(0.01, 0.99, 200).astype(np.float32); tn = rng.uniform(-1, 1, 200).astype(np.float32)
e = cnt @ Wemb.T
ref, _ = net(torch.tensor(cnt), torch.tensor(x).requires_grad_(True), torch.tensor(tn), torch.ones(200) * 0.15)
diff = float(np.max(np.abs(np_ge(e.astype(np.float32), x, tn) - ref.detach().numpy())))
print(f"NumPy forward-mode vs autograd: max |d ln gamma| = {diff:.1e}")

bench = ["CC(=O)C1CC(CC(=O)O)C1(C)C", "OC(=O)CC1CC(C(=O)O)C1(C)C", "OCC(O)CO", "CC(=O)O", "OCC(O)C(O)C(O)C(O)CO",
         "OC(=O)c1ccccc1"]
rows = []
for smi in bench:
    comps = smiles_to_components([smi]).components
    m = ActivityModel(comps)
    rows.append({"s2as_ms": 1e3 * timeit(lambda: smiles_to_components([smi]), 20),
                 "build_ms": 1e3 * timeit(lambda: ActivityModel(comps), 100),
                 "aiomfac_us": 1e6 * timeit(lambda: m.evaluate([0.5, 0.5], 298.15, basis="mass"), 500)})
med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
N = 12000
cntb = S.CNT[rng.randint(0, len(S.CNT), N)].astype(np.float32)
xb = rng.uniform(0.01, 0.99, N).astype(np.float32); tb = rng.uniform(-1, 1, N).astype(np.float32)
eb = (cntb @ Wemb.T).astype(np.float32)
t_np_batch = timeit(lambda: np_ge(eb, xb, tb), 20) / N
t_np_one = timeit(lambda: np_ge(eb[:1], xb[:1], tb[:1]), 2000)
ct, xt, tt, Mt = torch.tensor(cntb), torch.tensor(xb), torch.tensor(tb), torch.ones(N) * 0.15


def torch_eval(sl):
    xx = xt[sl].clone().requires_grad_(True)
    return net(ct[sl], xx, tt[sl], Mt[sl])[0].detach()


t_pt_batch = timeit(lambda: torch_eval(slice(0, N)), 10) / N
t_pt_one = timeit(lambda: torch_eval(slice(0, 1)), 500)
P = np.array([1, 10, 100, 1000, 10000])
t_aio = (med["s2as_ms"] + med["build_ms"]) * 1e-3 + P * med["aiomfac_us"] * 1e-6
t_sur = med["s2as_ms"] * 1e-3 + P * t_np_batch
out = {"n_parameters": int(n_par), "numpy_vs_autograd_max_abs": diff, "aiomfac_median": med,
       "per_point_us": {"numpy_batched": 1e6 * t_np_batch, "numpy_one": 1e6 * t_np_one, "torch_cpu_batched": 1e6 * t_pt_batch,
                        "torch_cpu_one": 1e6 * t_pt_one},
       "table6": [{"points": int(p), "aiomfac_ms": 1e3 * a, "surrogate_ms": 1e3 * s, "speedup": a / s}
                  for p, a, s in zip(P, t_aio, t_sur)]}
if torch.backends.mps.is_available():
    dev = torch.device("mps"); netg = S.Net(S.CNT.shape[1], "ge", "silu").to(dev)
    netg.load_state_dict(torch.load("../results/s1_models/ge_silu_sob_release_seed0.pt")); netg.eval()
    cg, xg, tg, Mg = ct.to(dev), xt.to(dev), tt.to(dev), Mt.to(dev)

    def gpu_eval(sl):
        xx = xg[sl].clone().requires_grad_(True)
        r = netg(cg[sl], xx, tg[sl], Mg[sl])[0].detach().cpu()
        return r
    out["per_point_us"]["mps_batched"] = 1e6 * timeit(lambda: gpu_eval(slice(0, N)), 10) / N
    out["per_point_us"]["mps_one"] = 1e6 * timeit(lambda: gpu_eval(slice(0, 1)), 100)
json.dump(out, open(str(_HERE / "results" / "GE_cost.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
