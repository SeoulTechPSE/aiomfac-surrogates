"""S1: Gibbs-Duhem-consistent (excess Gibbs energy) surrogate vs the Part 1 R_emb surrogate.

Same molecules, labels, hash split and training protocol as Part 1 Table 1 (BIMOG pool, test = hash100 < 15,
fixed 20 % interpolation split, 10 % validation, Adam 1e-3, weight decay 1e-5, batch 256, patience 30, <= 500 epochs,
5 seeds).  Models (all with the 16-dimensional subgroup embedding and three hidden layers of 128):

  remb_relu    the Part 1 network: (emb, w, T) -> (ln gamma_w, ln gamma_o), ReLU            [baseline, reproduces Table 1]
  remb_silu    the same with SiLU activations                                                 [effect of the activation]
  ge_silu      g^E/RT = x_w x_o h(emb, x_o, T), h an SiLU network; ln gamma_w = g - x_o g',
               ln gamma_o = g + x_w g' (autograd): Gibbs-Duhem and the pure-component references hold exactly
  ge_silu_sob  ge_silu with an added derivative loss on x_o(1-x_o) d ln gamma_i/dx_o (AIOMFAC central differences)

Metrics on the test molecules: MAE of ln gamma_w and ln gamma_o (Table 1 "gen"), MAE of the scaled derivatives
x_o(1-x_o) d ln gamma_i/dx_o, and the Gibbs-Duhem residual x_w d ln g_w/dx + x_o d ln g_o/dx relative to
|x_w d ln g_w/dx| + |x_o d ln g_o/dx| (median).  Writes ../results/s1_runs.csv and the trained networks to
../results/s1_models/.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MW = 0.01801528
D = np.load("../results/s1_labels.npz")
POOL = json.load(open("../results/s1_pool.json"))["pool"]
PM, X, TN, Y, DY, CNT = D["pm"], D["x"], D["tn"], D["y"], D["dy"], D["cnt"]
MK = np.array([o["M_kg"] for o in POOL])[PM]
SRC = np.array([o["src"] for o in POOL])
TEST_MOL = np.array([o["hash100"] < 15 for o in POOL])
SDY = (X * (1 - X))[:, None] * DY                      # scaled derivative labels
os.makedirs("../results/s1_models", exist_ok=True)
T0 = time.time()


class Net(nn.Module):
    def __init__(self, n_vocab, kind, act, emb=16):
        super().__init__()
        self.kind = kind
        self.emb = nn.Linear(n_vocab, emb, bias=False)
        A = {"relu": nn.ReLU, "silu": nn.SiLU}[act]
        n_out = 2 if kind == "remb" else 1
        self.head = nn.Sequential(nn.Linear(emb + 2, 128), A(), nn.Linear(128, 128), A(), nn.Linear(128, 128), A(),
                                  nn.Linear(128, n_out))
        self.register_buffer("em", torch.zeros(2)); self.register_buffer("es", torch.ones(2))
        self.register_buffer("ym", torch.zeros(2)); self.register_buffer("ys", torch.ones(2))

    def forward(self, c, x, tn, M, derivs=False, create_graph=False):
        """ln gamma (N, 2); with derivs also d ln gamma/dx (N, 2).  x must have requires_grad when derivs or kind=='ge'."""
        e = self.emb(c)
        if self.kind == "remb":
            w = (1 - x) * MW / ((1 - x) * MW + x * M)
            z = torch.stack([(w - self.em[0]) / self.es[0], (tn - self.em[1]) / self.es[1]], 1)
            lg = self.head(torch.cat([e, z], 1)) * self.ys + self.ym
            if not derivs:
                return lg, None
            d = torch.stack([torch.autograd.grad(lg[:, k].sum(), x, create_graph=create_graph, retain_graph=True)[0]
                             for k in range(2)], 1)
            return lg, d
        z = torch.stack([(x - self.em[0]) / self.es[0], (tn - self.em[1]) / self.es[1]], 1)
        h = self.head(torch.cat([e, z], 1))[:, 0] * self.ys[1]
        g = x * (1 - x) * h
        gp = torch.autograd.grad(g.sum(), x, create_graph=True)[0]
        lg = torch.stack([g - x * gp, g + (1 - x) * gp], 1)
        if not derivs:
            return lg, None
        gpp = torch.autograd.grad(gp.sum(), x, create_graph=create_graph)[0]
        d = torch.stack([-x * gpp, (1 - x) * gpp], 1)
        return lg, d


def tens(idx):
    return (torch.tensor(CNT[PM[idx]]), torch.tensor(X[idx], dtype=torch.float32), torch.tensor(TN[idx], dtype=torch.float32),
            torch.tensor(MK[idx], dtype=torch.float32), torch.tensor(Y[idx], dtype=torch.float32),
            torch.tensor(SDY[idx], dtype=torch.float32))


def train(model_name, train_pts, seed, lam=0.1, max_epochs=500, patience=30, batch=256, holdout=0.2):
    kind, act = {"remb_relu": ("remb", "relu"), "remb_silu": ("remb", "silu"), "ge_silu": ("ge", "silu"),
                 "ge_silu_sob": ("ge", "silu")}[model_name]
    sob = model_name.endswith("_sob")
    tr = np.array(train_pts)
    p = np.random.RandomState(0).permutation(len(tr)); fit = tr[p[int(holdout * len(tr)):]]
    nv = int(0.1 * len(fit)); val, fit = fit[:nv], fit[nv:]
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net = Net(CNT.shape[1], kind, act)
    if kind == "remb":
        wfit = (1 - X[fit]) * MW / ((1 - X[fit]) * MW + X[fit] * MK[fit])
        net.em[:] = torch.tensor([wfit.mean(), TN[fit].mean()]); net.es[:] = torch.tensor([wfit.std(), TN[fit].std()])
    else:
        net.em[:] = torch.tensor([X[fit].mean(), TN[fit].mean()]); net.es[:] = torch.tensor([X[fit].std(), TN[fit].std()])
    net.ym[:] = torch.tensor(Y[fit].mean(0)); net.ys[:] = torch.tensor(Y[fit].std(0) + 1e-6)
    ys = net.ys.clone(); sdys = torch.tensor(SDY[fit].std(0) + 1e-6, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Tf, Tv = tens(fit), tens(val)
    best, st, bad = 1e9, None, 0
    for ep in range(max_epochs):
        net.train()
        perm = rng.permutation(len(fit))
        for s in range(0, len(perm), batch):
            i = torch.from_numpy(perm[s:s + batch])
            c, x, tn, M, y, sdy = (t[i] for t in Tf)
            x = x.clone().requires_grad_(True)
            opt.zero_grad()
            lg, d = net(c, x, tn, M, derivs=sob, create_graph=True)
            loss = (((lg - y) / ys) ** 2).mean()
            if sob:
                loss = loss + lam * ((((x * (1 - x))[:, None] * d - sdy) / sdys) ** 2).mean()
            loss.backward(); opt.step()
        net.eval()
        c, x, tn, M, y, sdy = Tv
        x = x.clone().requires_grad_(True)
        lg, _ = net(c, x, tn, M)
        v = (((lg - y) / ys) ** 2).mean().item()
        if v < best - 1e-5:
            best, st, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(st); net.eval()
    return net, ep + 1, best


def evaluate(net, idx):
    out = []
    for s in range(0, len(idx), 20000):
        c, x, tn, M, y, sdy = tens(idx[s:s + 20000])
        x = x.clone().requires_grad_(True)
        lg, d = net(c, x, tn, M, derivs=True)
        out.append((lg.detach().numpy(), d.detach().numpy()))
    lg = np.concatenate([o[0] for o in out]); d = np.concatenate([o[1] for o in out])
    x = X[idx]
    e = np.abs(lg - Y[idx])
    sd = (x * (1 - x))[:, None] * d
    gd = (1 - x) * d[:, 0] + x * d[:, 1]
    rel = np.abs(gd) / (np.abs((1 - x) * d[:, 0]) + np.abs(x * d[:, 1]) + 1e-12)
    return {"mae_w": e[:, 0].mean(), "mae_o": e[:, 1].mean(), "mae_sdw": np.abs(sd[:, 0] - SDY[idx, 0]).mean(),
            "mae_sdo": np.abs(sd[:, 1] - SDY[idx, 1]).mean(), "gd_rel_median": float(np.median(rel))}


if __name__ == "__main__":
    models = sys.argv[1:] or ["remb_relu", "remb_silu", "ge_silu", "ge_silu_sob"]
    isB = SRC == "bimog"
    train_pts = np.where((isB & ~TEST_MOL)[PM])[0]
    test_pts = np.where((isB & TEST_MOL)[PM])[0]
    rows = []
    out_csv = "../results/s1_runs.csv"
    if os.path.exists(out_csv):
        rows = pd.read_csv(out_csv).to_dict("records")
        rows = [r for r in rows if r["model"] not in models]
    for mname in models:
        for seed in range(5):
            t1 = time.time()
            net, ep, best = train(mname, train_pts, seed)
            r = {"pool": "BIMOG", "model": mname, "seed": seed, "epochs": ep, "val": best, **evaluate(net, test_pts),
                 "train_s": time.time() - t1}
            rows.append(r)
            torch.save(net.state_dict(), f"../results/s1_models/{mname}_bimog_seed{seed}.pt")
            print(f"{mname} seed {seed}: ep {ep} mae_w {r['mae_w']:.4f} mae_o {r['mae_o']:.4f} "
                  f"sd {r['mae_sdw']:.4f}/{r['mae_sdo']:.4f} GD {r['gd_rel_median']:.2e} ({r['train_s']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    fmt = lambda s: f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"
    print(df.groupby("model", sort=False)[["mae_w", "mae_o", "mae_sdw", "mae_sdo", "gd_rel_median"]].agg(fmt).to_string())
    print(f"S1 DONE ({time.time()-T0:.0f}s)")
