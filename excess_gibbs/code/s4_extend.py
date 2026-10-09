"""Extend the electrolyte G^E surrogate of Part 2 (GEXNet, water + 14 ions) to Li+, Mg2+ and Br- (Part 2, Sect. 4.1).

The new ions enter the trained network in two places only:
  * three new input columns, ln(1 + m_i) of Li+, Mg2+, Br-, added to the pre-activation of the first hidden layer;
  * a new output head for the Margules coefficients H_ij of every pair that involves a new ion (51 pairs);
the Debye-Hueckel term needs only the charges.  The new weights start at zero.

Modes
  frozen   only the new parameters are trained (on the new-ion data); every system without a new ion is predicted
           exactly as before, because its new inputs are zero and the new coefficients multiply zero mole fractions
  full     all weights are trained, on the new-ion data plus the original training data (rehearsal)
  scratch  a network of the same 17-ion form trained from random weights on the original plus the new-ion data
AIOMFAC labels (s4_data.py) stand in for measurements.  Held-out ion pair: Mg2+/Br- (never in training).

usage: S2_TAG=s2v2 python s4_extend.py <mode> <seed> [n_new_train | all] [release]
Writes ../results/s4_models/<mode>_n<N>_seed<seed>.pt and ../results/s4_runs.jsonl (one line per run)."""
import copy
import json
import os
import sys
import time

os.environ.setdefault("S2_TAG", "s2v2")
import numpy as np
import torch
import torch.nn as nn

import s2_train as S

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
MW = S.MW
NEW = ["Li+", "Mg2+", "Br-"]

# ---------------------------------------------------------------- data on the 17-ion basis
D4 = np.load("../results/s4_labels.npz", allow_pickle=True)
NAMES17 = list(D4["names"])
assert NAMES17[:14] == S.NAMES, "the 14 Part 2 ions must come first"
Z17 = D4["z"]


def _filter(m, lgw, lgi):
    lnaw = np.log(1.0 / (1.0 + MW * m.sum(1))) + lgw
    return (lnaw >= np.log(0.02)) & (np.nanmax(np.abs(lgi), 1) <= 25)


ok4 = _filter(D4["m"], D4["lgw"], D4["lgi"])
M4, T4, LGW4, LGI4, HELD4 = D4["m"][ok4], D4["T"][ok4], D4["lgw"][ok4], D4["lgi"][ok4], D4["heldout"][ok4]
# original data (already filtered in s2_train), padded with zeros for the new ions
M0 = np.concatenate([S.M, np.zeros((len(S.M), 3))], 1)
LGI0 = np.concatenate([S.LGI, np.full((len(S.M), 3), np.nan)], 1)
MA = np.concatenate([M0, M4]); TA = np.concatenate([S.T, T4]); LGWA = np.concatenate([S.LGW, LGW4])
LGIA = np.concatenate([LGI0, LGI4])
LNAWA = np.log(1.0 / (1.0 + MW * MA.sum(1))) + LGWA
PRESA = ~np.isnan(LGIA); YIA = np.nan_to_num(LGIA)
AA, BA = S.dh_params(TA)
N0 = len(M0)

# splits: original (as s2_train) and new (held-out pair + 15 % test, fixed permutation)
_HOLD = [("Ca2+", "CO3--"), ("NH4+", "HSO4-"), ("K+", "IO3-")]
_h = np.zeros(N0, bool)
for a_, b_ in _HOLD:
    _h |= (S.M[:, S.NAMES.index(a_)] > 0) & (S.M[:, S.NAMES.index(b_)] > 0)
_rest = np.flatnonzero(~_h); _p = np.random.RandomState(123).permutation(_rest); _nte = int(0.15 * len(_rest))
OLD_TE, OLD_TR = _p[:_nte], _p[_nte:]
_rest4 = np.flatnonzero(~HELD4); _p4 = np.random.RandomState(321).permutation(_rest4); _nte4 = int(0.15 * len(_rest4))
NEW_TE, NEW_TR = N0 + _p4[:_nte4], N0 + _p4[_nte4:]
NEW_HELD = N0 + np.flatnonzero(HELD4)


def tens(idx, dtype=torch.float32):
    return (torch.tensor(MA[idx], dtype=dtype), torch.tensor((TA[idx] - 293.15) / 20, dtype=dtype),
            torch.tensor(AA[idx], dtype=dtype), torch.tensor(BA[idx], dtype=dtype),
            torch.tensor(LNAWA[idx], dtype=dtype), torch.tensor(YIA[idx], dtype=dtype), torch.tensor(PRESA[idx]))


# ---------------------------------------------------------------- extended network
class GEXNetX(nn.Module):
    """GEXNet on water + 14 ions, extended by n_new ions (appended after the 14)."""
    def __init__(self, base, n_new=3, hidden=128):
        super().__init__()
        self.base = base
        self.n0, self.n = base.n, base.n + n_new
        self.new_in = nn.Linear(n_new, hidden, bias=False)
        iu = torch.triu_indices(self.n + 1, self.n + 1)[:, 1:]
        old = {(int(i), int(j)): k for k, (i, j) in enumerate(base.iu.T.tolist())}
        is_old = torch.tensor([(int(i), int(j)) in old for i, j in iu.T.tolist()])
        self.register_buffer("iu", iu)
        self.register_buffer("old_pos", torch.tensor([k for k, (i, j) in enumerate(iu.T.tolist()) if (i, j) in old]))
        self.register_buffer("old_src", torch.tensor([old[(i, j)] for (i, j) in iu.T.tolist() if (i, j) in old]))
        self.register_buffer("new_pos", torch.nonzero(~is_old).flatten())
        self.new_out = nn.Linear(hidden, int((~is_old).sum()))
        nn.init.zeros_(self.new_in.weight); nn.init.zeros_(self.new_out.weight); nn.init.zeros_(self.new_out.bias)
        self.register_buffer("z2", torch.tensor(Z17 ** 2, dtype=torch.float32))

    def hidden(self, m, xw, tn):
        f = self.base.f
        u = f[0](torch.cat([torch.log1p(m[:, :self.n0]), xw[:, None], tn[:, None]], 1)) \
            + self.new_in(torch.log1p(m[:, self.n0:]))
        h = f[1](u); h = f[3](f[2](h)); h = f[5](f[4](h))
        return h

    def G(self, nw, ni, tn, A, b):
        m = ni / (nw * MW)[:, None]
        I = 0.5 * (m * self.z2.to(m.dtype)).sum(1)
        s = torch.sqrt(I + 1e-30); bs = b * s
        gdh = -(4 * A / b ** 3) * ((1 + bs) ** 2 / 2 - 2 * (1 + bs) + torch.log1p(bs) + 1.5)
        ntot = nw + ni.sum(1)
        x = torch.cat([nw[:, None], ni], 1) / ntot[:, None]
        h = self.hidden(m, x[:, 0], tn)
        hv = torch.zeros(len(m), self.iu.shape[1], dtype=m.dtype)
        hv[:, self.old_pos] = self.base.f[6](h)[:, self.old_src]
        hv[:, self.new_pos] = self.new_out(h)
        H = torch.zeros(len(m), self.n + 1, self.n + 1, dtype=m.dtype)
        H[:, self.iu[0], self.iu[1]] = hv
        H = 0.5 * (H + H.transpose(1, 2))
        return nw * MW * gdh + ntot * torch.einsum("bi,bij,bj->b", x, H, x)

    forward = S.GEXNet.forward


def load_base(seed, release=False):
    net = S.GEXNet()
    net.load_state_dict(torch.load(f"../results/s2v2_models/gex{'_release' if release else ''}_seed{seed}.pt"))
    return net.float()


def make(mode, seed, release=False):
    if mode == "scratch":
        torch.manual_seed(1000 + seed)
        net = GEXNetX(S.GEXNet())
        nn.init.normal_(net.new_in.weight, std=0.05); nn.init.normal_(net.new_out.weight, std=0.05)
    else:
        net = GEXNetX(load_base(seed, release))
    if mode == "frozen":
        for p in net.base.parameters():
            p.requires_grad_(False)
    return net


def train(net, tr, seed, max_epochs=400, patience=30, batch=256, lr=1e-3):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    tr = tr.copy(); rng.shuffle(tr); nv = max(1, int(0.1 * len(tr))); val, fit = tr[:nv], tr[nv:]
    sw = float(LNAWA[fit].std()); si = float(YIA[fit][PRESA[fit]].std())
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    Tf, Tv = tens(fit), tens(val)

    def loss(B, cg=True):
        m, tn, A, b, aw, yi, pr = B
        pa, pi = net(m, tn, A, b, create_graph=cg)
        return (((pi - yi) / si) ** 2 * pr).sum() / pr.sum() + (((pa - aw) / sw) ** 2).mean()
    best, st, bad = 1e9, None, 0
    for ep in range(max_epochs):
        net.train(); perm = rng.permutation(len(fit))
        for s in range(0, len(perm), batch):
            i = torch.from_numpy(perm[s:s + batch]); opt.zero_grad()
            loss([t[i] for t in Tf]).backward(); opt.step()
        net.eval()
        v = loss(Tv, cg=False).item()
        if v < best - 1e-5:
            best, st, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(st); net.eval()
    return ep + 1


NEW_IDX = [NAMES17.index(n) for n in NEW]


def evaluate(net, idx, ref=None):
    """MAE of ln a_w and of ln gamma of all ions present / of the new ions present; optional max |change| vs ref."""
    out = {}
    if len(idx) == 0:
        return out
    m, tn, A, b, aw, yi, pr = tens(idx)
    pa, pi = net(m, tn, A, b, create_graph=False)
    pa, pi = pa.detach().numpy(), pi.detach().numpy()
    e = np.abs(pi - yi.numpy()); prn = pr.numpy()
    out["mae_lnaw"] = float(np.abs(pa - aw.numpy()).mean())
    out["mae_lng_ion"] = float(e[prn].mean())
    newmask = np.zeros_like(prn); newmask[:, NEW_IDX] = True
    if (prn & newmask).any():
        out["mae_lng_new"] = float(e[prn & newmask].mean())
        out["mae_lng_old_ions"] = float(e[prn & ~newmask].mean())
    if ref is not None:
        ra, ri = ref
        out["max_change_lnaw"] = float(np.abs(pa - ra).max())
        out["max_change_lng"] = float(np.abs((pi - ri)[prn]).max())
    return out


def base_predictions(seed, idx):
    """predictions of the unmodified 14-ion network on the original test set"""
    net = load_base(seed)
    m, tn, A, b = [t for t in tens(idx)[:4]]
    pa, pi = net(m[:, :14], tn, A, b, create_graph=False)
    pi = np.concatenate([pi.detach().numpy(), np.zeros((len(idx), 3))], 1)
    return pa.detach().numpy(), pi


if __name__ == "__main__":
    mode, seed = sys.argv[1], int(sys.argv[2])
    ntr = sys.argv[3] if len(sys.argv) > 3 else "all"
    release = len(sys.argv) > 4 and sys.argv[4] == "release"
    new_tr = NEW_TR if not release else np.concatenate([NEW_TR, NEW_TE, NEW_HELD])
    if ntr != "all":
        new_tr = np.random.RandomState(seed).permutation(new_tr)[:int(ntr)]
    tr = new_tr if mode == "frozen" else np.concatenate([OLD_TR if not release else np.arange(N0), new_tr])
    net = make(mode, seed, release)
    # the frozen mode trains ~7000 parameters only: larger step and more epochs (it reached the 400-epoch cap at 1e-3)
    kw = {"lr": 3e-3, "max_epochs": 1500, "patience": 50} if mode == "frozen" else {}
    t0 = time.time(); ep = train(net, tr, seed, **kw)
    os.makedirs("../results/s4_models", exist_ok=True)
    name = f"{mode}_n{ntr}{'_release' if release else ''}_seed{seed}"
    torch.save(net.state_dict(), f"../results/s4_models/{name}.pt")
    r = {"name": name, "mode": mode, "seed": seed, "n_new_train": len(new_tr), "n_train": len(tr), "epochs": ep,
         "train_s": time.time() - t0,
         "trainable_params": int(sum(p.numel() for p in net.parameters() if p.requires_grad))}
    if not release:
        ref = base_predictions(seed, OLD_TE) if mode != "scratch" else None
        r.update({"new_test": evaluate(net, NEW_TE), "new_heldout_MgBr": evaluate(net, NEW_HELD),
                  "old_test": evaluate(net, OLD_TE, ref)})
    with open("../results/s4_runs.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r), flush=True)
