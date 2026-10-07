"""S3: organic-electrolyte coupling in excess-Gibbs-energy form (revised Part 2, Sects. 2.9-2.10 and 3.6-3.9).

For water (w) + one organic (o) + one salt (cation c, anion a) the surrogate excess Gibbs energy is

    G/RT = G_org(n_w, n_o) + G_el(n_w, n_c, n_a) + G_cross(n_w, n_o, n_c, n_a),

  G_org    the Part 1 binary surrogate, (n_w + n_o) g(x'_o, T) on the salt-free basis (S1 release ensemble);
  G_el     the electrolyte surrogate (mole-fraction G^E, S2 release ensemble);
  G_cross  = n_tot x_o (x_c + x_a) h(organic embedding, x'_o, ln(1 + m), T, ion identities) * SCALE,

and every activity coefficient is a derivative of G (ions: asymmetric convention on the mole-fraction scale,
ln gamma*_i = dG/dn_i minus its value at infinite dilution in water).  Option A is G without the cross term; it is
consistent and corresponds to adding the water activity coefficients of the two binary surrogates.  Option C adds
G_cross, trained on the residual of all four activity coefficients of AIOMFAC.  G_cross vanishes without organic or
without salt, so the binary limits and the reference states are those of the binary surrogates.

Errors are reported in ln gamma on AIOMFAC's scales; at a given composition they equal the errors of the mole-fraction
scale values used here (the conversion terms are exact and common).  Organics, salts, compositions and splits are
those of the earlier Part 2 analysis (aiomfac-surrogates v2.0.1, part2_inorganic_surrogate/01_part2_revision.ipynb).

Writes ../results/s3_*.csv, s3_data.npz (labels), s3_models/cross_seed*.pt."""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ.setdefault("S2_TAG", "s2v2")
P2 = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "part2_inorganic_surrogate")
sys.path.insert(0, P2)
import aiomfac_py.s2as._mapping as _mp                                   # noqa: E402
_mp.print = lambda *a, **k: None
from aiomfac_py import Component                                          # noqa: E402
from aiomfac_py.phase_equilibrium import ExplicitLiquidModel              # noqa: E402
from aiomfac_py.s2as import smiles_to_components                          # noqa: E402
import generate_training_data as gtd                                       # noqa: E402
import s1_train as S1                                                      # noqa: E402
import s2_train as S2                                                      # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
MW = 0.01801528
SCALE = 30.0
VOCAB = json.load(open("../results/s1_pool.json"))["vocab"]
CATS = ["Na+", "K+", "NH4+", "Ca2+", "H+"]
ANS = ["Cl-", "NO3-", "SO4--", "HSO4-", "HCO3-", "CO3--", "IO3-", "I-", "OH-"]
ION_Z = {n: z for n, _, z in gtd.CATIONS + gtd.ANIONS}
SP = ["H2O", "organic", "cation", "anion"]
OUT = os.environ.get("S3_OUT", "../results/")
os.makedirs(OUT + "s3_models", exist_ok=True)
T0 = time.time()

# ---------------------------------------------------------------- binary surrogates (release ensembles)
ORG = []
for s in range(3):
    n_ = S1.Net(len(VOCAB), "ge", "silu"); n_.load_state_dict(torch.load(f"../results/s1_models/ge_silu_sob_release_seed{s}.pt"))
    ORG.append(n_.double().eval())
EL = []
for s in [int(v) for v in os.environ.get("S2_REL_SEEDS", "0,1,2,3,4").split(",")]:
    n_ = S2.GEXNet(); n_.load_state_dict(torch.load(f"../results/s2v2_models/gex_release_seed{s}.pt"))
    EL.append(n_.double().eval())
W_EMB = ORG[0].emb.weight.detach().numpy()                 # (16, vocab): embedding of the first release member


def counts(smiles):
    comps = smiles_to_components([smiles]).components
    sub = comps[1].subgroups
    c = np.zeros(len(VOCAB))
    for s, q in sub:
        if int(s) not in VOCAB:
            return None, sub
        c[VOCAB.index(int(s))] += q
    return c, sub


def org_lng(cnt, xo, T):
    """binary surrogate: ln gamma_w, ln gamma_o on the salt-free mole-fraction basis (ensemble mean)."""
    xo = np.atleast_1d(np.asarray(xo, float))
    out = 0
    for net in ORG:
        x = torch.tensor(xo, requires_grad=True)
        lg, _ = net(torch.tensor(np.tile(cnt, (len(xo), 1))), x, torch.full((len(xo),), (T - 293.15) / 20),
                    torch.zeros(len(xo)))
        out = out + lg.detach().numpy()
    return out / len(ORG)


def el_lng(m14, T):
    """electrolyte surrogate: ln gamma_w (mole fraction, water + ions), ln gamma*_i (mole fraction) (ensemble mean)."""
    m = torch.tensor(np.atleast_2d(m14), dtype=torch.float64)
    A, b = S2.dh_params(np.full(len(m), T))
    args = (m, torch.full((len(m),), (T - 293.15) / 20), torch.tensor(A), torch.tensor(b))
    aw = gi = 0
    for net in EL:
        a_, g_ = net(*args, create_graph=False)
        aw, gi = aw + a_.detach().numpy(), gi + g_.detach().numpy()
    aw, gi = aw / len(EL), gi / len(EL)
    lnxw = -np.log1p(MW * m.numpy().sum(1))
    return aw - lnxw, gi - lnxw[:, None]


def salt_stoich(c, a):
    g = math.gcd(abs(ION_Z[c]), abs(ION_Z[a])); return abs(ION_Z[a]) // g, abs(ION_Z[c]) // g


_LM = {}


def lm_for(key, sub, ions):
    if key not in _LM:
        orgs = [Component(2, "org", tuple(map(tuple, sub)))] if sub is not None else []
        _LM[key] = ExplicitLiquidModel(orgs, [{"Ca2+": "Ca++"}.get(i, i) for i in ions])
    return _LM[key]


def lng_x(lna, n, ion_mask):
    x = n / n.sum()
    return np.where(ion_mask, lna + math.log(MW) - np.log(x), lna - np.log(x))


def point(name, smiles, sub, cnt, c, a, wo, ws, T):
    """AIOMFAC labels, Option A and binary-sum values (all ln gamma on the mole-fraction scales) for one composition."""
    qc, qa = salt_stoich(c, a)
    lm = lm_for((name, c, a), sub, [c, a])
    Mo = float(lm._mm[1])
    ww = 1 - wo - ws
    ns = ws / (qc * gtd.ION_MW[c] + qa * gtd.ION_MW[a])
    n = np.array([ww / MW, wo / Mo, qc * ns, qa * ns])
    t = lng_x(lm.ln_a(n, T), n, np.array([0, 0, 1, 1], bool))
    if not np.all(np.isfinite(t)):
        return None
    xo_p = n[1] / (n[0] + n[1])
    gw_o, go = org_lng(cnt, xo_p, T)[0]
    m14 = np.zeros(14); m14[S2.NAMES.index(c)] = n[2] / (n[0] * MW); m14[S2.NAMES.index(a)] = n[3] / (n[0] * MW)
    ew, ei = el_lng(m14, T)
    A_ = np.array([gw_o + ew[0], go, ei[0, S2.NAMES.index(c)], ei[0, S2.NAMES.index(a)]])
    return n, t, A_, Mo


def binary_sum(name, sub, c, a, n, T):
    lo = lm_for((name, None, None), sub, [])
    no = n[:2]; bo = lng_x(lo.ln_a(no, T), no, np.array([0, 0], bool))
    ls = lm_for((None, c, a), None, [c, a])
    ns = n[[0, 2, 3]]; bs = lng_x(ls.ln_a(ns, T), ns, np.array([0, 1, 1], bool))
    return np.array([bo[0] + bs[0], bo[1], bs[1], bs[2]])


def static(cnt, c, a, T):
    return np.concatenate([cnt @ W_EMB.T, [(T - 293.15) / 20], [float(k == c) for k in CATS], [float(k == a) for k in ANS]])


# ---------------------------------------------------------------- data (organics, salts and grid of Part 2)
orgs_all = json.load(open(P2 + "/organics.json"))
from rdkit import Chem                                                      # noqa: E402
EXCLUDE = {"malonic acid", "ethanol"}
oxy = [d for d in orgs_all if d["name"] not in EXCLUDE
       and any(at.GetSymbol() == "O" for at in Chem.MolFromSmiles(d["smiles"]).GetAtoms())][:120]
TEST_O = [d["name"] for i, d in enumerate(oxy) if i % 6 == 0]
CAL_O = [d["name"] for i, d in enumerate(oxy) if i % 6 == 3]
TRAIN_O = [d["name"] for i, d in enumerate(oxy) if i % 6 not in (0, 3)]
SMI = {d["name"]: d["smiles"] for d in orgs_all}
SMI.update({"malonic acid": "OC(=O)CC(=O)O", "ethanol": "CCO"})
SALTS_TRAIN = [("Na+", "Cl-"), ("Na+", "NO3-"), ("K+", "Cl-"), ("NH4+", "SO4--"), ("Ca2+", "NO3-"), ("H+", "Cl-"),
               ("Na+", "SO4--"), ("K+", "NO3-")]
SALTS_GEN = [("NH4+", "Cl-"), ("NH4+", "NO3-")]
CNT, SUBS, MISSING = {}, {}, []
for nm in list(SMI):
    if nm in TRAIN_O + TEST_O + CAL_O + ["malonic acid", "ethanol", "glucose", "citric acid", "glycerol", "sucrose",
                                         "levoglucosan"]:
        c_, s_ = counts(SMI[nm])
        if c_ is None:
            MISSING.append(nm)
        else:
            CNT[nm], SUBS[nm] = c_, s_
print(f"organics: train {len(TRAIN_O)}, calibration {len(CAL_O)}, test {len(TEST_O)}; outside the S1 vocabulary: "
      f"{MISSING}", flush=True)


def build(onames, salts, grid_o=(0.08, 0.20, 0.30), grid_s=(0.03, 0.08), grid_T=(283.15, 313.15)):
    rows = []
    for o in onames:
        if o not in CNT:
            continue
        for c, a in salts:
            for wo in grid_o:
                for ws in grid_s:
                    for T in grid_T:
                        p = point(o, SMI[o], SUBS[o], CNT[o], c, a, wo, ws, T)
                        if p is None:
                            continue
                        n, t, A_, Mo = p
                        rows.append({"organic": o, "salt": c + a, "cat": c, "an": a, "wo": wo, "ws": ws, "T": T,
                                     "n": n, "t": t, "A": A_, "s": static(CNT[o], c, a, T), "Mo": Mo})
    return rows


SETS = {}
for key, (on, sl, gs) in {"train": (TRAIN_O, SALTS_TRAIN, (0.0002, 0.002, 0.03, 0.08)), "unseen salt": (TRAIN_O, SALTS_GEN, None),
                      "unseen organic": (TEST_O, SALTS_TRAIN, None), "unseen organic and salt": (TEST_O, SALTS_GEN, None),
                      "calibration": (CAL_O, SALTS_TRAIN + SALTS_GEN, None)}.items():
    # the training organics and salts also get a dilute salt level (mass fractions 0.0002 and 0.002), so that the cross term is
    # learned towards infinite dilution of the salt (needed for mixed-solvent reference states)
    SETS[key] = build(on, sl) if gs is None else build(on, sl, grid_s=gs)
print({k: len(v) for k, v in SETS.items()}, f"({time.time()-T0:.0f}s)", flush=True)


def stack(rows):
    return (np.stack([r["n"] for r in rows]), np.stack([r["s"] for r in rows]), np.stack([r["t"] for r in rows]),
            np.stack([r["A"] for r in rows]))


# ---------------------------------------------------------------- Option A decomposition (324 points of Part 2)
A_ORGS = ["glucose", "citric acid", "glycerol", "sucrose", "levoglucosan", "ethanol"]
A_SALTS = [("Na+", "Cl-"), ("NH4+", "SO4--"), ("Ca2+", "NO3-")]
dec = []
for o in A_ORGS:
    if o not in CNT:
        continue
    for c, a in A_SALTS:
        for wo in (0.05, 0.15, 0.30):
            for ws in (0.02, 0.08):
                for T in (283.15, 298.15, 313.15):
                    p = point(o, SMI[o], SUBS[o], CNT[o], c, a, wo, ws, T)
                    if p is None:
                        continue
                    n, t, A_, _ = p
                    bsum = binary_sum(o, SUBS[o], c, a, n, T)
                    for k, sp in enumerate(SP):
                        dec.append({"organic": o, "salt": c + a, "wo": wo, "ws": ws, "T": T, "species": sp,
                                    "total": abs(t[k] - A_[k]), "own": abs(bsum[k] - A_[k]), "cross": abs(t[k] - bsum[k])})
dec = pd.DataFrame(dec); dec.to_csv(OUT + "s3_optionA_decomposition.csv", index=False)
dec["sp2"] = dec.species.replace({"cation": "ion", "anion": "ion"})
print(f"Option A decomposition: {dec[dec.species=='H2O'].shape[0]} ternary points")
print(dec.groupby("sp2")[["total", "own", "cross"]].mean().round(3).to_string(), flush=True)


# ---------------------------------------------------------------- cross term
class Cross(nn.Module):
    def __init__(self, n_static, h=96):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_static + 3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self.register_buffer("sm", torch.zeros(n_static)); self.register_buffer("ss", torch.ones(n_static))

    def G(self, n, s):
        nt = n.sum(1); x = n / nt[:, None]
        xo_p = n[:, 1] / (n[:, 0] + n[:, 1])
        mc = n[:, 2] / (n[:, 0] * MW); ma = n[:, 3] / (n[:, 0] * MW)
        z = torch.cat([(s - self.sm) / self.ss, xo_p[:, None], torch.log1p(mc)[:, None], torch.log1p(ma)[:, None]], 1)
        return nt * x[:, 1] * (x[:, 2] + x[:, 3]) * self.f(z)[:, 0] * SCALE

    def forward(self, n, s, create_graph=True):
        n = n.clone().requires_grad_(True)
        return torch.autograd.grad(self.G(n, s).sum(), n, create_graph=create_graph)[0]


TR = SETS["train"]
p_ = np.random.RandomState(0).permutation(len(TR)); n15 = int(0.15 * len(TR))
I_TEST, I_VAL, I_FIT = p_[:n15], p_[n15:2 * n15], p_[2 * n15:]
N_, S_, T_, A_ = stack(TR)
R_ = T_ - A_


def train_cross(seed, max_epochs=3000, patience=100, batch=256):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    net = Cross(S_.shape[1])
    net.sm.copy_(torch.tensor(S_[I_FIT].mean(0))); net.ss.copy_(torch.tensor(S_[I_FIT].std(0) + 1e-6))
    sd = torch.tensor(R_[I_FIT].std(0))
    Nt, St, Rt = torch.tensor(N_), torch.tensor(S_), torch.tensor(R_)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    L = lambda i, cg=True: ((((net(Nt[i], St[i], cg) - Rt[i]) / sd) ** 2).mean())
    best, st, bad = 1e9, None, 0
    for ep in range(max_epochs):
        net.train(); perm = I_FIT[rng.permutation(len(I_FIT))]
        for k in range(0, len(perm), batch):
            i = torch.from_numpy(perm[k:k + batch]); opt.zero_grad(); L(i).backward(); opt.step()
        net.eval(); v = L(torch.from_numpy(I_VAL), False).item()
        if v < best - 1e-6:
            best, st, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                break
    net.load_state_dict(st); net.eval()
    return net, ep + 1


def cross_pred(nets, rows):
    N, S, _, _ = stack(rows)
    return np.mean([net(torch.tensor(N), torch.tensor(S), False).detach().numpy() for net in nets], 0)


RES = []
for seed in range(3):
    t0 = time.time(); net, ep = train_cross(seed); RES.append(net)
    torch.save(net.state_dict(), OUT + f"s3_models/cross_seed{seed}.pt")
    print(f"cross seed {seed}: {ep} epochs ({time.time()-t0:.0f}s)", flush=True)

# OOD guard in the embedding space: nearest training organic; above the threshold the whole cross term is dropped
E_tr = np.stack([CNT[o] @ W_EMB.T for o in TRAIN_O if o in CNT]); e_m, e_s = E_tr.mean(0), E_tr.std(0) + 1e-6
nn_dist = lambda o: float(np.min(np.linalg.norm((E_tr - e_m) / e_s - (CNT[o] @ W_EMB.T - e_m) / e_s, axis=1)))

rows, thr = [], []
for seed, net in enumerate(RES):
    cal = SETS["calibration"]; r = cross_pred([net], cal); per = []
    df = pd.DataFrame({"o": [x["organic"] for x in cal]})
    for o, idx in df.groupby("o").groups.items():
        idx = np.asarray(idx)
        t = np.stack([cal[i]["t"] for i in idx]); a = np.stack([cal[i]["A"] for i in idx])
        per.append((nn_dist(o), np.abs(t[:, 1] - a[:, 1] - r[idx, 1]).mean() < np.abs(t[:, 1] - a[:, 1]).mean()))
    d = np.array([x[0] for x in per]); h = np.array([x[1] for x in per]); bt, bs = np.inf, -1
    for tt in np.r_[np.unique(d), np.inf]:
        sc = h[d <= tt].sum() + (~h[d > tt]).sum()
        if sc > bs:
            bs, bt = sc, tt
    thr.append((bt, int(bs), len(d)))
    for name, rset in [("interpolation", [TR[i] for i in I_TEST]), ("unseen salt", SETS["unseen salt"]),
                       ("unseen organic", SETS["unseen organic"]), ("unseen organic and salt", SETS["unseen organic and salt"])]:
        _, _, t, a = stack(rset); r = cross_pred([net], rset)
        for guard in (False, True):
            if guard and name in ("interpolation", "unseen salt"):
                continue
            keep = np.array([nn_dist(x["organic"]) <= bt for x in rset]) if guard else np.ones(len(rset), bool)
            c_ = a + r * keep[:, None]
            rows.append({"seed": seed, "case": name + (" (guarded)" if guard else ""), "n": len(rset),
                         **{f"{s} A": np.abs(t[:, k] - a[:, k]).mean() for k, s in enumerate(SP)},
                         **{f"{s} C": np.abs(t[:, k] - c_[:, k]).mean() for k, s in enumerate(SP)}})
C_res = pd.DataFrame(rows); C_res.to_csv(OUT + "s3_optionC_runs.csv", index=False)
print("guard thresholds:", [(round(t, 2), f"{s}/{n}") for t, s, n in thr])
summ = C_res.groupby("case", sort=False).mean(numeric_only=True).drop(columns="seed")
for s in SP:
    summ[f"{s} red%"] = 100 * (1 - summ[f"{s} C"] / summ[f"{s} A"])
print(summ.round(3).to_string(), flush=True)
THR = float(np.median([t for t, _, _ in thr]))
json.dump({"thresholds": [[float(t), s, n] for t, s, n in thr], "median": THR,
           "test_beyond": [o for o in TEST_O if o in CNT and nn_dist(o) > THR],
           "missing_vocab": MISSING}, open(OUT + "s3_guard.json", "w"), indent=1)

# per-organic effect (calibration + test organics, seed-mean)
per_org = []
for nm, rset in [("calibration", SETS["calibration"]), ("test", SETS["unseen organic"] + SETS["unseen organic and salt"])]:
    r = cross_pred(RES, rset); _, _, t, a = stack(rset)
    df = pd.DataFrame({"o": [x["organic"] for x in rset]})
    for o, idx in df.groupby("o").groups.items():
        idx = np.asarray(idx)
        ea = np.abs(t[idx, 1] - a[idx, 1]).mean(); ec = np.abs(t[idx, 1] - a[idx, 1] - r[idx, 1]).mean()
        per_org.append({"set": nm, "organic": o, "nn_dist": nn_dist(o), "mae_A": ea, "mae_C": ec, "reduction_%": 100 * (1 - ec / ea)})
pd.DataFrame(per_org).to_csv(OUT + "s3_per_organic.csv", index=False)


# ---------------------------------------------------------------- worked examples and Lopes et al. (1999)
def molal_ln_g(n, lgx, Mo):
    """mole-fraction-scale ion values -> AIOMFAC molal scale (molality per kg of salt-free solvent)."""
    x = n / n.sum(); m = n / (n[0] * MW + n[1] * Mo)
    return lgx[2:] + np.log(x[2:]) - math.log(MW) - np.log(m[2:])


def predict(o, c, a, wo, ws, T, guard=True):
    p = point(o, SMI[o], SUBS[o], CNT[o], c, a, wo, ws, T)
    n, t, A_, Mo = p
    row = {"n": n, "s": static(CNT[o], c, a, T)}
    r = cross_pred(RES, [{"n": n, "s": row["s"], "t": t, "A": A_}])[0]
    if guard and nn_dist(o) > THR:
        r = r * 0
    return n, t, A_, A_ + r, Mo


def worked(o, c, a, grid):
    out = []
    for wo, ws in grid:
        n, t, A_, C_, Mo = predict(o, c, a, wo, ws, 298.15)
        rec = {"wo": wo, "ws": ws}
        for k, s in enumerate(SP):
            rec.update({f"t_{s}": t[k], f"a_{s}": A_[k], f"c_{s}": C_[k]})
        for tag, v in (("t", t), ("a", A_), ("c", C_)):
            lm_ = molal_ln_g(n, v, Mo); rec[f"{tag}_lngpm"] = 0.5 * (lm_[0] + lm_[1])
        out.append(rec)
    return pd.DataFrame(out)


mal = worked("malonic acid", "NH4+", "SO4--", [(x, y) for x in (0.03, 0.07, 0.11, 0.15) for y in (0.02, 0.05, 0.08)])
eth = worked("ethanol", "K+", "Cl-", [(x, 0.03) for x in np.linspace(0.05, 0.45, 13)])
mal.to_csv(OUT + "s3_worked_malonic.csv", index=False); eth.to_csv(OUT + "s3_worked_ethanol_kcl.csv", index=False)
print(f"guard threshold {THR:.2f}; nn distance malonic acid {nn_dist('malonic acid'):.2f}, ethanol {nn_dist('ethanol'):.2f}")
for name, Dw in [("malonic acid + (NH4)2SO4", mal), ("ethanol + KCl", eth)]:
    print(f"{name} (n = {len(Dw)}): MAE Option A -> Option C")
    for s in SP:
        print(f"  {s:8s} {np.abs(Dw[f't_{s}'] - Dw[f'a_{s}']).mean():.3f} -> {np.abs(Dw[f't_{s}'] - Dw[f'c_{s}']).mean():.3f}")
for tag in ("a", "c"):
    g_t, g_m = np.exp(eth.t_lngpm), np.exp(eth[f"{tag}_lngpm"])
    print(f"ethanol + KCl gamma+- Option {tag.upper()}: MAE {np.abs(g_m-g_t).mean():.3f} ({100*(np.abs(g_m-g_t)/g_t).mean():.1f} %)")

LOPES = {0.05: (0.4112, 0.0393, 0.224, 0.0041), 0.10: (0.4335, 0.0527, 0.283, -0.0006),
         0.15: (0.4584, 0.0760, 0.116, -0.0104), 0.20: (0.4863, 0.0350, 0.181, 0.0105)}


def pitzer_ln_gpm(m, Aphi, b0, b1, cphi, alpha=2.0, b=1.2):
    sm = np.sqrt(m)
    f = -Aphi * (sm / (1 + b * sm) + (2 / b) * np.log(1 + b * sm))
    Bg = 2 * b0 + 2 * b1 / (alpha ** 2 * m) * (1 - (1 + alpha * sm - alpha ** 2 * m / 2) * np.exp(-alpha * sm))
    return f + m * Bg + m ** 2 * 1.5 * cphi


M_KCL = gtd.ION_MW["K+"] + gtd.ION_MW["Cl-"]


def ln_gpm_all(xe, m):
    tot = 1.0 + m * M_KCL
    n, t, A_, C_, Mo = predict("ethanol", "K+", "Cl-", xe / tot, m * M_KCL / tot, 298.15)
    return [0.5 * molal_ln_g(n, v, Mo).sum() for v in (t, A_, C_)]


lop = []
for xe, par in LOPES.items():
    ref = ln_gpm_all(xe, 1e-4)
    for m in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]:
        mod = ln_gpm_all(xe, m)
        lop.append({"ethanol_wt": xe, "m": m, "exp": float(np.exp(pitzer_ln_gpm(m, *par))),
                    "AIOMFAC": math.exp(mod[0] - ref[0]), "Option A": math.exp(mod[1] - ref[1]),
                    "Option C": math.exp(mod[2] - ref[2]), "lnref_AIOMFAC": ref[0], "lnref_A": ref[1], "lnref_C": ref[2],
                    "ln_AIOMFAC": mod[0], "ln_A": mod[1], "ln_C": mod[2]})
lop = pd.DataFrame(lop); lop.to_csv(OUT + "s3_lopes_comparison.csv", index=False)
for k in ("AIOMFAC", "Option A", "Option C"):
    e = (lop[k] - lop.exp).abs()
    print(f"{k:9s} vs Lopes: MAE {e.mean():.3f} ({100*(e/lop.exp).mean():.1f} %), max {100*(e/lop.exp).max():.1f} %")
for k, cc in (("Option A", "A"), ("Option C", "C")):
    e = (lop[k] - lop.AIOMFAC).abs()
    fin = (lop[f"ln_{cc}"] - lop.ln_AIOMFAC).abs().mean()
    r0 = (lop[f"lnref_{cc}"] - lop.lnref_AIOMFAC).groupby(lop.ethanol_wt).first().abs()
    print(f"{k} vs AIOMFAC: {100*(e/lop.AIOMFAC).mean():.1f} %; |dln g+-| finite m {fin:.3f}, reference term {r0.mean():.3f}")
print(f"S3 DONE ({time.time()-T0:.0f}s)")
