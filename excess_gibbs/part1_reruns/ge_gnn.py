"""GNN models with G^E heads.
  t3  : Table 3 (notebook 08): A_emb, A_desc and GNN/hybrids B-G on label set 2 (10 x 3 per BIMOG molecule), 5 seeds.
  mcm : Table 4 GNN rows (notebook 01 cell 9): GNN on MCM subsets of 400 and 2855 molecules, 3 seeds.
The GINE encoder is unchanged; the head is a G^E head (SiLU, 2 hidden layers of 128 as in the original heads)."""
import copy
import sys

import numpy as np
import pandas as pd

from aiomfac_py import ActivityModel, Component

from ge_common import *

WATER = Component(1, "Water", ((16, 1),))
ATOMS = ["C", "O", "N", "Cl", "Br", "F", "I", "S"]
BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def graph(smi):
    mol = Chem.MolFromSmiles(smi)
    x = [[float(a.GetSymbol() == s) for s in ATOMS] + [a.GetDegree() / 4, a.GetTotalNumHs() / 4, float(a.GetIsAromatic()),
         float(a.IsInRing()), a.GetFormalCharge()] for a in mol.GetAtoms()]
    ei, ea = [], []
    for b in mol.GetBonds():
        f = [float(b.GetBondType() == t) for t in BONDS]; i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); ei += [(i, j), (j, i)]; ea += [f, f]
    if not ei:
        ei, ea = [(0, 0)], [[0.0] * 4]
    return np.array(x, np.float32), np.array(ei, np.int64).T, np.array(ea, np.float32)


class Bank:
    def __init__(self, graphs):
        xs, eis, eas, own, off = [], [], [], [], 0
        for k, (x, ei, ea) in enumerate(graphs):
            xs.append(x); eis.append(ei + off); eas.append(ea); own.append(np.full(len(x), k)); off += len(x)
        self.x = torch.tensor(np.concatenate(xs)); self.ei = torch.tensor(np.concatenate(eis, 1)); self.ea = torch.tensor(np.concatenate(eas))
        self.own = torch.tensor(np.concatenate(own)); self.n = len(graphs)


class GINE(nn.Module):
    def __init__(self, h):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, h, ei, e):
        return self.mlp(h + torch.zeros_like(h).index_add_(0, ei[1], torch.relu(h[ei[0]] + e)))


class Encoder(nn.Module):
    def __init__(self, h=64, emb=64, L=3):
        super().__init__()
        self.ap = nn.Linear(len(ATOMS) + 5, h); self.ep = nn.ModuleList([nn.Linear(4, h) for _ in range(L)])
        self.cv = nn.ModuleList([GINE(h) for _ in range(L)]); self.bn = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(L)])
        self.out = nn.Linear(2 * h, emb)

    def forward(self, b):
        h = self.ap(b.x)
        for c, n, e in zip(self.cv, self.bn, self.ep):
            h = torch.relu(n(c(h, b.ei, e(b.ea))))
        s = torch.zeros(b.n, h.shape[1]).index_add_(0, b.own, h); cnt = torch.zeros(b.n).index_add_(0, b.own, torch.ones(len(h))).clamp(min=1)
        return self.out(torch.cat([s / cnt[:, None], s], 1))


def label(ids, n_comp, n_temp, seed, pool=POOL):
    rng = np.random.RandomState(seed); rows = []
    for oi in ids:
        m = ActivityModel([WATER, Component(2, "o", tuple(map(tuple, pool[oi]["subgroups"])))]); e = np.linspace(0.02, 0.98, n_comp + 1)
        for w in rng.uniform(e[:-1], e[1:]):
            for T in rng.uniform(273.15, 313.15, n_temp):
                try:
                    lg = m.evaluate([w, 1 - w], T, basis="mass").ln_gamma
                    if np.isfinite(lg[:2]).all():
                        rows.append((oi, w, (T - 293.15) / 20, lg[0], lg[1]))
                except Exception:
                    pass
    a = np.array(rows)
    return a[:, 0].astype(int), a[:, 1:3].astype(np.float32), a[:, 3:5].astype(np.float32)


def train_graph(enc, seed, pts, val, Ymat, EX, PMv, Mmol, bank_tr, loc_t, D=None, train_encoder=True, drop=0.0,
                max_epochs=200, patience=15, batch=128, emb_eval=None, seed_offset=1000):
    torch.manual_seed(seed_offset + seed); rng = np.random.RandomState(seed)
    Dn = np.zeros((len(PMv), 0), np.float32) if D is None else D[PMv]
    if Dn.shape[1]:
        dm, ds = Dn[pts].mean(0), Dn[pts].std(0); ds = np.where(ds < 1e-3, 1.0, ds); Dn = (Dn - dm) / ds
    else:
        dm = ds = None
    Dt = torch.tensor(Dn.astype(np.float32))
    xo = torch.tensor(x_of_w(EX[:, 0], Mmol[PMv]).astype(np.float32)); Mt = torch.tensor(Mmol[PMv].astype(np.float32))
    Tt = torch.tensor(EX[:, 1]); Yt = torch.tensor(Ymat); ys = torch.tensor(Ymat[pts].std(0) + 1e-6)
    head = GEHead(64 + Dn.shape[1], L=2, drop=drop); head.set_norm(EX[pts, 0], EX[pts, 1], Ymat[pts, 1].std())
    params = list(head.parameters()) + (list(enc.parameters()) if train_encoder else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    if not train_encoder:
        enc.eval()
        with torch.no_grad():
            EMB_TR = enc(bank_tr)

    def fwd(i, training, cg=True):
        if train_encoder:
            enc.train(training); emb = enc(bank_tr)
        else:
            emb = EMB_TR
        f = torch.cat([emb[loc_t[i]], Dt[i]], 1)
        return head(f, xo[i], Mt[i], Tt[i], create_graph=cg)
    best, st, bad = 1e9, None, 0; val_t = torch.tensor(val)
    for ep in range(max_epochs):
        head.train(); perm = torch.from_numpy(pts[rng.permutation(len(pts))])
        for s in range(0, len(perm), batch):
            i = perm[s:s + batch]; opt.zero_grad()
            (((fwd(i, True) - Yt[i]) / ys) ** 2).mean().backward(); opt.step()
        head.eval()
        v = (((fwd(val_t, False, cg=False).detach() - Yt[val_t]) / ys) ** 2).mean().item()
        if v < best - 1e-5:
            best, bad = v, 0
            st = ({k: t.clone() for k, t in head.state_dict().items()}, {k: t.clone() for k, t in enc.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(st[0]); enc.load_state_dict(st[1]); head.eval(); enc.eval()

    def predict(bank, loc, idx, PMe, EXe, De=None):
        with torch.no_grad():
            emb = enc(bank)
        Dd = np.zeros((len(idx), 0), np.float32) if De is None else ((De[PMe[idx]] - dm) / ds).astype(np.float32)
        f = torch.cat([emb[loc[idx]], torch.tensor(Dd)], 1)
        x = torch.tensor(x_of_w(EXe[idx, 0], Mmol[PMe[idx]]).astype(np.float32))
        return head(f, x, torch.tensor(Mmol[PMe[idx]].astype(np.float32)), torch.tensor(EXe[idx, 1]),
                    create_graph=False).detach().numpy()
    return predict, ep + 1


task = sys.argv[1]
if task == "t3":
    isB = SRC == "bimog"; bids = np.where(isB)[0]
    PM2, EX2, Y2 = label(bids, 10, 3, 0)
    gen_pts = np.where(TEST[PM2])[0]; tr_all = np.where(~TEST[PM2])[0]
    p0 = np.random.RandomState(0).permutation(len(tr_all))
    interp_pts = tr_all[p0[:int(0.2 * len(tr_all))]]; fit_all = tr_all[p0[int(0.2 * len(tr_all)):]]
    nv = int(0.1 * len(fit_all)); val_pts, fit_pts = fit_all[:nv], fit_all[nv:]
    MGS8 = sorted({sg_ for i in bids for sg_ in [__import__("ge_common")._sg.main_group(int(s)) for s, q in POOL[i]["subgroups"]]})
    DESC8 = np.array([desc(o, MGS8) if SRC[i] == "bimog" else np.zeros(len(MGS8) + 4) for i, o in enumerate(POOL)], np.float32)
    FEATS["R_desc8"] = DESC8
    print(f"{len(Y2)} points: fit {len(fit_pts)}, val {len(val_pts)}, interp {len(interp_pts)}, gen {len(gen_pts)}", flush=True)

    def errs(P):
        E = np.abs(P - Y2)
        return {"gen_w": E[gen_pts, 0].mean(), "gen_org": E[gen_pts, 1].mean(), "interp_org": E[interp_pts, 1].mean()}
    # synthetic pretraining pool (notebook 08 cell 7)
    from rdkit import Chem as _C
    MONO = {"alcohol": lambda c: c + "O", "ketone": lambda c: "CC(=O)" + c[1:] if len(c) > 1 else None,
            "ester_methyl": lambda c: c + "C(=O)OC", "ether_methyl": lambda c: c + "OC", "amine": lambda c: c + "N",
            "nitro": lambda c: c + "[N+](=O)[O-]", "acid": lambda c: c + "C(=O)O", "aldehyde": lambda c: c + "C=O",
            "chloro": lambda c: c + "Cl", "diol": lambda c: c + "O" if len(c) < 2 else "OC" + c[1:] + "O",
            "nitrate": lambda c: c + "O[N+](=O)[O-]", "hydroperoxide": lambda c: c + "OO", "ether_ethyl": lambda c: c + "OCC"}
    DI = {"hydroxy_acid": lambda c: "OC" + c[1:] + "C(=O)O", "keto_acid": lambda c: "CC(=O)" + c[1:] + "C(=O)O" if len(c) > 1 else None,
          "hydroxy_ketone": lambda c: "OC" + c[1:] + "C(C)=O", "diacid": lambda c: "OC(=O)" + c + "C(=O)O",
          "hydroxy_nitrate": lambda c: "OC" + c[1:] + "O[N+](=O)[O-]", "keto_aldehyde": lambda c: "CC(=O)" + c[1:] + "C=O" if len(c) > 1 else None}

    def chain(n, br):
        if br == "linear": return "C" * n
        if br == "methyl_branch": return "C" * (n // 2) + "C(C)" + "C" * (n - n // 2 - 1) if n >= 3 else None
        if br == "ring": return "C1" + "C" * (n - 2) + "C1" if n >= 3 else None
    bcanon = {o["canon"] for o in POOL if o["src"] == "bimog"}
    seen, cands = set(bcanon), []
    for n in range(1, 11):
        for br in ("linear", "methyl_branch", "ring"):
            c = chain(n, br)
            if c is None: continue
            temps = list(MONO.items()) + (list(DI.items()) if br == "linear" else [])
            for nm, fn in temps:
                try: smi = fn(c)
                except Exception: smi = None
                if not smi: continue
                mol = _C.MolFromSmiles(smi)
                if mol is None: continue
                can = _C.MolToSmiles(mol)
                if can in seen: continue
                seen.add(can); cands.append((f"{nm}_C{n}_{br}", can))
    from aiomfac_py.s2as import smiles_to_components
    SYN = []
    for nm, s in cands:
        r = smiles_to_components([s], water_as_component1=True)
        if r.removed: continue
        try:
            if not (ActivityModel(r.components).evaluate([0.9, 0.1], 298.15, basis="mass").gamma_neutral > 0).all(): continue
        except Exception:
            continue
        SYN.append({"name": nm, "smiles": s, "subgroups": [list(map(int, t)) for t in r.components[1].subgroups],
                    "M_kg": float(ActivityModel([WATER, Component(2, "o", tuple(r.components[1].subgroups))]).mixture.mmass[1])})
    SPM, SEX, SY = label(range(len(SYN)), 10, 2, 42, pool=SYN)
    MSYN = np.array([o["M_kg"] for o in SYN])
    print(f"synthetic pool {len(SYN)} molecules, {len(SY)} points ({time.time()-T0:.0f}s)", flush=True)
    GB = {i: graph(POOL[i]["smiles"]) for i in bids}; GS = [graph(o["smiles"]) for o in SYN]
    train_mols = np.unique(PM2[tr_all]); BANK_TR = Bank([GB[i] for i in train_mols]); BANK_ALL = Bank([GB[i] for i in bids])
    LOC_TR = {m: k for k, m in enumerate(train_mols)}; LOC_ALL = {m: k for k, m in enumerate(bids)}
    loc_tr = torch.tensor([LOC_TR.get(m, -1) for m in PM2]); loc_all = torch.tensor([LOC_ALL[m] for m in PM2])
    ALLP = np.arange(len(Y2))

    def gm(enc, seed, **kw):
        return train_graph(enc, seed, fit_pts, val_pts, Y2, EX2, PM2, MAIO, BANK_TR, loc_tr, **kw)

    def pretrain(seed):
        p = np.random.RandomState(1).permutation(len(SY)); n5 = max(1, int(0.05 * len(SY)))
        enc = Encoder(); torch.manual_seed(2000 + seed)
        _, ep = train_graph(enc, seed, p[n5:], p[:n5], SY, SEX, SPM, MSYN, Bank(GS), torch.tensor(SPM),
                            max_epochs=100, patience=10, batch=256)
        return enc, ep
    res = []
    for seed in range(5):
        t0 = time.time()
        pred, ep = fit_ge("R_emb", tr_all, seed, interp_split=True, max_epochs=300, patience=25, label=(PM2, EX2, Y2))
        res.append({"model": "A_emb", "seed": seed, **errs(pred(PM2, EX2)), "epochs": ep})
        pred, ep = fit_ge("R_desc8", tr_all, seed, interp_split=True, max_epochs=300, patience=25, label=(PM2, EX2, Y2))
        res.append({"model": "A_desc", "seed": seed, **errs(pred(PM2, EX2)), "epochs": ep})
        torch.manual_seed(seed); p_, ep = gm(Encoder(), seed)
        res.append({"model": "B", "seed": seed, **errs(p_(BANK_ALL, loc_all, ALLP, PM2, EX2)), "epochs": ep})
        enc_pre, ep_pre = pretrain(seed)
        p_, ep = gm(copy.deepcopy(enc_pre), seed)
        res.append({"model": "C", "seed": seed, **errs(p_(BANK_ALL, loc_all, ALLP, PM2, EX2)), "epochs": ep})
        torch.manual_seed(seed); p_, ep = gm(Encoder(), seed, D=DESC8)
        res.append({"model": "D", "seed": seed, **errs(p_(BANK_ALL, loc_all, ALLP, PM2, EX2, DESC8)), "epochs": ep})
        p_, ep = gm(copy.deepcopy(enc_pre), seed, D=DESC8)
        res.append({"model": "E", "seed": seed, **errs(p_(BANK_ALL, loc_all, ALLP, PM2, EX2, DESC8)), "epochs": ep})
        for nm, enc0 in [("F", copy.deepcopy(enc_pre)), ("G", (torch.manual_seed(3000 + seed), Encoder())[1])]:
            p_, ep = gm(enc0, seed, D=DESC8, train_encoder=False, drop=0.2)
            res.append({"model": nm, "seed": seed, **errs(p_(BANK_ALL, loc_all, ALLP, PM2, EX2, DESC8)), "epochs": ep})
        print(f"  seed {seed}: pretraining {ep_pre} epochs; " + ", ".join(f"{r['model']} {r['gen_org']:.3f}" for r in res if r["seed"] == seed)
              + f" ({time.time()-t0:.0f}s; total {time.time()-T0:.0f}s)", flush=True)
        pd.DataFrame(res).to_csv("results/GE_table3_runs.csv", index=False)
    R3 = pd.DataFrame(res)
    order = ["A_emb", "A_desc", "B", "C", "D", "E", "F", "G"]
    print(R3.groupby("model")[["gen_w", "gen_org", "interp_org"]].agg(fmt).loc[order].to_string())
    bc = R3.pivot(index="seed", columns="model", values="gen_org")
    print("C better than B in", int((bc["C"] < bc["B"]).sum()), "of", len(bc), "seeds; G <= F in", int((bc["G"] <= bc["F"]).sum()))

elif task == "mcm":
    isB = SRC == "bimog"; isM = SRC == "mcm"
    mcm_tr = np.where(isM & ~TEST)[0]; evals = {"MCM": isM & TEST, "BIMOG": isB & TEST}
    GR = {}
    rows = []
    for seed in range(3):
        order = np.random.RandomState(100 + seed).permutation(mcm_tr)
        for n in (400, len(order)):
            sub = order[:n]; m = np.zeros(len(POOL), bool); m[sub] = True
            tr = pts_of(m); rng = np.random.RandomState(seed); tr = tr.copy(); rng.shuffle(tr)
            nv = int(0.1 * len(tr)); val, fit = tr[:nv], tr[nv:]
            for i in sub:
                if i not in GR: GR[i] = graph(POOL[i]["smiles"])
            bank = Bank([GR[i] for i in sub]); loc = -np.ones(len(POOL), int); loc[sub] = np.arange(len(sub))
            torch.manual_seed(seed)
            pred, ep = train_graph(Encoder(), seed, fit, val, Y, EXTRA, PM, MAIO, bank, torch.tensor(loc[PM]),
                                   max_epochs=40, patience=6, batch=1024, seed_offset=0)
            rec = {"model": "GNN", "n": n, "seed": seed, "epochs": ep}
            for name, em_ in evals.items():
                ids = np.where(em_)[0]
                for i in ids:
                    if i not in GR: GR[i] = graph(POOL[i]["smiles"])
                b = Bank([GR[i] for i in ids]); l2 = -np.ones(len(POOL), int); l2[ids] = np.arange(len(ids))
                g = pts_of(em_); P = pred(b, torch.tensor(l2[PM]), g, PM, EXTRA)
                e = np.abs(P - Y[g]); rec[f"{name} org"] = e[:, 1].mean(); rec[f"{name} w"] = e[:, 0].mean()
            rows.append(rec)
            print(f"  seed {seed} n={n}: GNN MCM {rec['MCM org']:.3f} BIMOG {rec['BIMOG org']:.3f} ({time.time()-T0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv("results/GE_table4_gnn_runs.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("n")[["MCM org", "BIMOG org", "MCM w"]].agg(fmt).to_string())
print(f"DONE gnn {task} ({time.time()-T0:.0f}s)")
