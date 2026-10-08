"""S2: Gibbs-Duhem-consistent (excess Gibbs energy) electrolyte surrogate vs an unconstrained MLP, free-species basis.

Per kg of water, g(m, T) = G^E/(RT kg_w) is the excess Gibbs energy on the molal scale.  Then
    ln gamma_i (molal) = dg/dm_i,
    ln a_w = -M_w sum_i m_i + M_w (g - sum_i m_i dg/dm_i),
so the water and ion activities are consistent by construction (n_w d mu_w + sum_i n_i d mu_i = 0).  Model:
    g = g_DH(I, T) + sum_ij m_i m_j H_ij(m, T),
with g_DH the integral of AIOMFAC's long-range term (A, b of aiomfac_py.lr), so that the dilute limit is the
Debye-Hueckel law, and H a symmetric matrix output by an MLP of (log(1 + m), T) (SiLU).  The training targets are
AIOMFAC's ln gamma_i of the ions present and its ln a_w (converted from the mole-fraction-scale ln gamma_w).

Baseline: an MLP (ReLU, 3 x 128) mapping (log(1 + m), T) directly to (ln gamma_w, ln gamma_1..14), as in Part 2.
Split: 15 % of the samples held out (seeded).  5 seeds each.  Metrics: MAE of ln gamma of the ions present and of
ln a_w; Gibbs-Duhem residual (1/M_w) d ln a_w + sum m_i d ln a_i along random composition directions, relative to
the sum of the absolute terms.  Writes ../results/s2_runs.csv and models to ../results/s2_models/."""
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from aiomfac_py.lr import debye_huckel_parameters

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
MW = 0.01801528
TAG = os.environ.get("S2_TAG", "s2")   # "s2v2" for the concentrated-solution label set
D = np.load(f"../results/{TAG}_labels.npz", allow_pickle=True)
M, T, LGW, LGI, Z = D["m"], D["T"], D["lgw"], D["lgi"], D["z"]
KIND = D["kind"]
if TAG != "s2":
    # concentrated label set: drop AIOMFAC states far outside the physical range (a_w < 0.02 or |ln gamma| > 25;
    # 3602 of 80 000, mostly Ca2+/SO4-- and I- above I = 20)
    _lnaw = np.log(1.0 / (1.0 + 0.01801528 * M.sum(1))) + LGW
    _ok = (_lnaw >= np.log(0.02)) & (np.nanmax(np.abs(LGI), 1) <= 25)
    M, T, LGW, LGI, KIND = M[_ok], T[_ok], LGW[_ok], LGI[_ok], KIND[_ok]
# derivative labels (s2_dlabels.py) for the Sobolev-type variant "gex_sob": d[ln a_w, ln gamma_i]/d ln m_j
DL = None
if TAG != "s2" and os.path.exists(f"../results/{TAG}_dlabels.npz"):
    DL = np.load(f"../results/{TAG}_dlabels.npz")["D"][_ok]
NAMES = list(D["names"])
NI = M.shape[1]
# water target: ln a_w (AIOMFAC: ln x_w + ln gamma_w, x on the dissociated basis with water as the only neutral)
LNAW = np.log(1.0 / (1.0 + MW * M.sum(1))) + LGW
PRES = ~np.isnan(LGI)
YI = np.nan_to_num(LGI)
os.makedirs(f"../results/{TAG}_models", exist_ok=True)


def dh_params(Tk):
    A, b = zip(*[debye_huckel_parameters(float(t)) for t in Tk])
    return np.array(A), np.array(b)


AT, BT = dh_params(T)


class GENet(nn.Module):
    def __init__(self, n=NI, hidden=128):
        super().__init__()
        self.n = n
        self.iu = torch.triu_indices(n, n)
        self.f = nn.Sequential(nn.Linear(n + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.iu.shape[1]))
        self.register_buffer("z2", torch.tensor(Z ** 2, dtype=torch.float32))

    def g(self, m, tn, A, b):
        I = 0.5 * (m * self.z2).sum(1)
        s = torch.sqrt(I + 1e-30); bs = b * s
        gdh = -(4 * A / b ** 3) * ((1 + bs) ** 2 / 2 - 2 * (1 + bs) + torch.log1p(bs) + 1.5)
        hv = self.f(torch.cat([torch.log1p(m), tn[:, None]], 1))
        H = torch.zeros(len(m), self.n, self.n, dtype=m.dtype)
        H[:, self.iu[0], self.iu[1]] = hv
        H = 0.5 * (H + H.transpose(1, 2))
        return gdh + torch.einsum("bi,bij,bj->b", m, H, m)

    def forward(self, m, tn, A, b, create_graph=True):
        m = m.clone().requires_grad_(True)
        g = self.g(m, tn, A, b)
        dg = torch.autograd.grad(g.sum(), m, create_graph=create_graph)[0]
        lnaw = -MW * m.sum(1) + MW * (g - (m * dg).sum(1))
        return lnaw, dg


class GEXNet(nn.Module):
    """Mole-fraction excess Gibbs energy (per kg water):
        G/RT = n_w M_w g_DH(m) + n_tot sum_ij x_i x_j H_ij(x, T),  species = water + ions, H_ww = 0,
    with ln a_w = ln x_w + dG/dn_w and ln gamma_i(molal) = ln x_w + dG/dn_i - [dG/dn_i at infinite dilution].
    All outputs are derivatives of one scalar, so Gibbs-Duhem holds; the short-range part stays bounded as x_w -> 0."""
    def __init__(self, n=NI, hidden=128):
        super().__init__()
        self.n = n
        self.iu = torch.triu_indices(n + 1, n + 1)[:, 1:]          # drop the (water, water) entry
        self.f = nn.Sequential(nn.Linear(n + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.iu.shape[1]))
        self.register_buffer("z2", torch.tensor(Z ** 2, dtype=torch.float32))

    def G(self, nw, ni, tn, A, b):
        m = ni / (nw * MW)[:, None]
        I = 0.5 * (m * self.z2).sum(1)
        s = torch.sqrt(I + 1e-30); bs = b * s
        gdh = -(4 * A / b ** 3) * ((1 + bs) ** 2 / 2 - 2 * (1 + bs) + torch.log1p(bs) + 1.5)
        ntot = nw + ni.sum(1)
        x = torch.cat([nw[:, None], ni], 1) / ntot[:, None]
        hv = self.f(torch.cat([torch.log1p(m), x[:, :1], tn[:, None]], 1))
        H = torch.zeros(len(m), self.n + 1, self.n + 1, dtype=m.dtype)
        H[:, self.iu[0], self.iu[1]] = hv
        H = 0.5 * (H + H.transpose(1, 2))
        return nw * MW * gdh + ntot * torch.einsum("bi,bij,bj->b", x, H, x)

    def forward(self, m, tn, A, b, create_graph=True):
        nw = torch.full((len(m),), 1.0 / MW, dtype=m.dtype, requires_grad=True)
        ni = m.clone().requires_grad_(True)
        dw, di = torch.autograd.grad(self.G(nw, ni, tn, A, b).sum(), (nw, ni), create_graph=create_graph)
        n0 = torch.zeros_like(m).requires_grad_(True)
        ref = torch.autograd.grad(self.G(nw.detach(), n0, tn, A, b).sum(), n0, create_graph=create_graph)[0]
        lnxw = -torch.log1p(MW * m.sum(1))
        return lnxw + dw, lnxw[:, None] + di - ref


class DirectNet(nn.Module):
    def __init__(self, n=NI, hidden=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n + 1))

    def forward(self, m, tn, A, b, create_graph=True):
        y = self.f(torch.cat([torch.log1p(m), tn[:, None]], 1))
        return y[:, 0], y[:, 1:]


def tens(idx):
    out = (torch.tensor(M[idx], dtype=torch.float32), torch.tensor((T[idx] - 293.15) / 20, dtype=torch.float32),
           torch.tensor(AT[idx], dtype=torch.float32), torch.tensor(BT[idx], dtype=torch.float32),
           torch.tensor(LNAW[idx], dtype=torch.float32), torch.tensor(YI[idx], dtype=torch.float32),
           torch.tensor(PRES[idx]))
    if DL is not None:
        out = out + (torch.tensor(DL[idx], dtype=torch.float32),)
    return out


# derivative loss: weight, form ("mse" or "huber", on standardized residuals) and Huber threshold
SOB_LAMBDA, SOB_H = float(os.environ.get("SOB_LAMBDA", "0.1")), 1.0e-2
SOB_LOSS, SOB_DELTA = os.environ.get("SOB_LOSS", "mse"), float(os.environ.get("SOB_DELTA", "1.0"))
RUN_NAME = os.environ.get("S2_NAME")                 # file name of a variant (default: the kind)


def _sob_term(r, ok):
    """mean over valid entries of the standardized derivative residual r: squared, or Huber (quadratic below
    SOB_DELTA, linear above), which limits the weight of the heavy tail of the AIOMFAC derivative labels"""
    if SOB_LOSS == "huber":
        a = r.abs()
        v = torch.where(a <= SOB_DELTA, 0.5 * r ** 2, SOB_DELTA * (a - 0.5 * SOB_DELTA)) * 2.0
    else:
        v = r ** 2
    return (v * ok).sum() / ok.sum().clamp(min=1)


def directional(net, m, tn, A, b, v, h=SOB_H, create_graph=True):
    """Directional derivative of (ln a_w, ln gamma_i) along v in ln m (central difference, step h)."""
    ap, ip = net(m * torch.exp(h * v), tn, A, b, create_graph=create_graph)
    am, im = net(m * torch.exp(-h * v), tn, A, b, create_graph=create_graph)
    return (ap - am) / (2 * h), (ip - im) / (2 * h)


def directional_labels(dl, v):
    """AIOMFAC directional derivatives D v (water, ions) and their validity (finite labels of present rows)."""
    dz = torch.nan_to_num(dl)
    y = torch.einsum("brj,bj->br", dz, v)
    ok = torch.isfinite(dl).any(2)                       # rows with at least one finite derivative (present)
    return y[:, 0], y[:, 1:], ok[:, 0], ok[:, 1:]


def train(kind, seed, tr, max_epochs=400, patience=30, batch=256):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    tr = tr.copy(); rng.shuffle(tr); nv = int(0.1 * len(tr)); val, fit = tr[:nv], tr[nv:]
    net = {"ge": GENet, "gex": GEXNet, "gex_sob": GEXNet, "direct": DirectNet}[kind]()
    sob = kind.endswith("_sob")
    if sob and DL is None:
        raise RuntimeError("gex_sob needs the derivative labels of s2_dlabels.py")
    sw = float(LNAW[fit].std()); si = float(LGI[fit][PRES[fit]].std())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Tf, Tv = tens(fit), tens(val)
    if sob:                                              # scale of the directional-derivative labels
        g0 = torch.Generator().manual_seed(seed)
        v0 = torch.randn(Tf[0].shape, generator=g0) * Tf[6]
        yw0, yi0, okw0, oki0 = directional_labels(Tf[7], v0)
        sdw, sdi = float(yw0[okw0].std()), float(yi0[oki0].std())

    def loss(B, cg=True, with_sob=False):
        m, tn, A, b, aw, yi, pr = B[:7]
        pa, pi = net(m, tn, A, b, create_graph=cg)
        li = (((pi - yi) / si) ** 2 * pr).sum() / pr.sum()
        out = li + (((pa - aw) / sw) ** 2).mean()
        if with_sob:
            v = torch.randn(m.shape) * pr                   # random direction among the species present
            yw, yi_, okw, oki = directional_labels(B[7], v)
            dw, di = directional(net, m, tn, A, b, v)
            oki = oki & pr
            lw = _sob_term((dw - yw) / sdw, okw)
            ld = _sob_term((di - yi_) / sdi, oki)
            out = out + SOB_LAMBDA * (lw + ld)
        return out
    best, st, bad = 1e9, None, 0
    for ep in range(max_epochs):
        net.train(); perm = rng.permutation(len(fit))
        for s in range(0, len(perm), batch):
            i = torch.from_numpy(perm[s:s + batch]); opt.zero_grad()
            loss([t[i] for t in Tf], with_sob=sob).backward(); opt.step()
        net.eval()
        v = loss(Tv, cg=False).item()
        if v < best - 1e-5:
            best, st, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(st); net.eval()
    return net, ep + 1


def evaluate(net, te, kind):
    m, tn, A, b, aw, yi, pr = tens(te)[:7]
    pa, pi = net(m, tn, A, b, create_graph=False)
    pa, pi = pa.detach().numpy(), pi.detach().numpy()
    ei = np.abs(pi - yi.numpy())[pr.numpy()]
    out = {"mae_lng_ion": float(ei.mean()), "mae_lnaw": float(np.abs(pa - aw.numpy()).mean())}
    # Gibbs-Duhem along random directions among the species present (finite differences, step 1e-4 relative; float64)
    import copy
    net = copy.deepcopy(net).double()
    if hasattr(net, "z2"):
        net.z2 = net.z2.double()
    rng = np.random.RandomState(0); res = []
    for k in rng.choice(len(te), min(2000, len(te)), replace=False):
        mm = M[te[k]]; p = mm > 0
        dvec = np.zeros(NI); dvec[p] = rng.normal(size=p.sum()) * mm[p] * 1e-4
        q = float(Z @ dvec)
        idx = np.flatnonzero(p & (Z * q < 0))
        if q != 0 and len(idx):
            dvec[idx[0]] -= q / Z[idx[0]]
        mp, mn_ = mm + dvec, mm - dvec
        if np.any(mp < 0) or np.any(mn_ < 0):
            continue
        X = torch.tensor(np.stack([mp, mn_]), dtype=torch.float64)
        tt = torch.full((2,), float((T[te[k]] - 293.15) / 20), dtype=torch.float64)
        Aa = torch.full((2,), float(AT[te[k]]), dtype=torch.float64); Bb = torch.full((2,), float(BT[te[k]]), dtype=torch.float64)
        a_, g_ = net(X, tt, Aa, Bb, create_graph=False)
        a_, g_ = a_.detach().numpy().astype(float), g_.detach().numpy().astype(float)
        lnai_p = g_[0][p] + np.log(mp[p]); lnai_m = g_[1][p] + np.log(mn_[p])
        terms = np.concatenate([[(a_[0] - a_[1]) / MW], mm[p] * (lnai_p - lnai_m)])
        res.append(abs(terms.sum()) / (np.abs(terms).sum() + 1e-30))
    out["gd_rel_median"] = float(np.median(res))
    return out


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["direct", "ge"]
    # unseen ion combinations (as in Part 2): every sample in which both ions of a held-out pair are present
    HOLD = [("Ca2+", "CO3--"), ("NH4+", "HSO4-"), ("K+", "IO3-")]
    held = np.zeros(len(M), bool)
    for a_, b_ in HOLD:
        held |= (M[:, NAMES.index(a_)] > 0) & (M[:, NAMES.index(b_)] > 0)
    rest = np.flatnonzero(~held) if TAG != "s2" else np.arange(len(M))
    perm = np.random.RandomState(123).permutation(rest); nte = int(0.15 * len(rest))
    te, tr = perm[:nte], perm[nte:]
    tu = np.flatnonzero(held)
    REL = os.environ.get("S2_RELEASE") == "1"      # release ensemble: train on every sample (10 % validation only)
    if REL:
        tr, tu = np.arange(len(M)), np.array([], int)
        te = np.random.RandomState(7).choice(len(M), 2000, replace=False)   # in-sample check only
    SEEDS = [int(v) for v in os.environ.get("S2_SEEDS", "0,1,2,3,4").split(",")]
    sfx = "" if len(SEEDS) == 5 else "_s" + "".join(map(str, SEEDS))
    rows = []
    for kind in kinds:
        for seed in SEEDS:
            t0 = time.time(); net, ep = train(kind, seed, tr)
            r = {"model": kind, "seed": seed, "epochs": ep, **evaluate(net, te, kind), "train_s": time.time() - t0}
            if len(tu) and TAG != "s2":
                r.update({k + "_unseen": v for k, v in evaluate(net, tu, kind).items()})
            name = RUN_NAME or kind
            r["model"] = name
            rows.append(r); torch.save(net.state_dict(), f"../results/{TAG}_models/{name}{'_release' if REL else ''}_seed{seed}.pt")
            print(f"{kind} seed {seed}: ep {ep} ion {r['mae_lng_ion']:.4f} lnaw {r['mae_lnaw']:.5f} GD {r['gd_rel_median']:.2e} "
                  f"({r['train_s']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(f"../results/{TAG}_runs_{'release_' if REL else ''}{RUN_NAME or '_'.join(kinds)}{sfx}.csv",
                                      index=False)
    df = pd.DataFrame(rows)
    f = lambda s: f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"
    print(df.groupby("model")[["mae_lng_ion", "mae_lnaw", "gd_rel_median"]].agg(f).to_string())
