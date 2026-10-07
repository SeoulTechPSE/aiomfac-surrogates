"""Sect. 3.9 / Table 7 / Fig. 6 with G^E heads (notebook 02 cell 14): M1-M4, three seeds."""
import numpy as np
import pandas as pd

from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components

from ge_common import *

WATER = Component(1, "Water", ((16, 1),))
TARGETS = {"Glycerol": "OCC(O)CO", "1,2,4-Butanetriol": "OCCC(O)CO", "2,5-Hexanediol": "CC(O)CCC(C)O",
           "Ethanol": "CCO", "2-Propanol": "CC(C)O", "Malonic acid": "OC(=O)CC(=O)O"}
W19 = np.linspace(0.05, 0.95, 19)
isB = SRC == "bimog"; isM = SRC == "mcm"
cooh = Chem.MolFromSmarts("C(=O)[OH]")
diacid = np.array([SRC[i] == "mcm" and len(Chem.MolFromSmiles(POOL[i]["smiles"]).GetSubstructMatches(cooh)) >= 2
                   for i in range(len(POOL))])
MODELS = {"M1": isB & ~TEST, "M2": (isB | isM) & ~TEST, "M3": isB | isM, "M4": (isB | isM) & ~TEST & ~diacid}
truth, Q = {}, {}
for nm, s in TARGETS.items():
    r = smiles_to_components([s], water_as_component1=True)
    comp = r.components[1]
    m = ActivityModel([WATER, Component(2, "o", tuple(comp.subgroups))])
    truth[nm] = np.array([m.evaluate([x, 1 - x], 298.15, basis="mass").ln_gamma[:2] for x in W19])
    cnt = np.zeros(len(VOCAB), np.float32)
    for sgr, q in comp.subgroups:
        cnt[VOCAB.index(int(sgr))] = q
    Q[nm] = (cnt, float(m.mixture.mmass[1]))
lit, curves = [], {}
for mn, mask in MODELS.items():
    for seed in range(3):
        pred, _ = fit_ge("R_emb", pts_of(mask), seed, max_epochs=300)
        emb, head, fm, fs = pred.modules
        for nm in TARGETS:
            cnt, M = Q[nm]
            f = emb(torch.tensor(np.tile(cnt, (19, 1))))
            x = torch.tensor(x_of_w(W19, M).astype(np.float32))
            p = head(f, x, torch.full((19,), M), torch.full((19,), (298.15 - 293.15) / 20), create_graph=False).detach().numpy()
            e = np.abs(p - truth[nm]); curves[f"{mn}|{seed}|{nm}"] = p
            lit.append({"model": mn, "seed": seed, "compound": nm, "w": e[:, 0].mean(), "org": e[:, 1].mean(), "max": e.max()})
    print(f"  {mn} done ({time.time()-T0:.0f}s)", flush=True)
lit = pd.DataFrame(lit); lit.to_csv("results/GE_literature_runs.csv", index=False)
print(lit.groupby(["compound", "model"], sort=False).agg(w=("w", "mean"), org=("org", "mean"), org_sd=("org", "std"))
      .round(3).unstack("model").to_string())
np.savez_compressed("results/GE_literature_curves.npz", W19=W19, **{f"truth|{k}": v for k, v in truth.items()}, **curves)
print(f"DONE lit ({time.time()-T0:.0f}s)")
