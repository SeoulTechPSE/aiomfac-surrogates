"""S1 data: the Part 1 molecule pool and AIOMFAC labels, reproduced from aiomfac-surrogates v2.0.1
part1_organic_surrogate/01_unified_reruns_and_checkpoint.ipynb (cells 1-2: featurize, pool, hash split, label), plus
  * the molar mass of every organic (AIOMFAC), to convert water mass fraction w to organic mole fraction x_o;
  * derivative labels d ln gamma_i / d x_o at every label point (central differences in x_o, step 1e-5 relative to
    min(x_o, 1 - x_o), AIOMFAC), for the Sobolev variant.
Writes ../results/s1_labels.npz and ../results/s1_pool.json.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
import aiomfac_py.s2as._mapping as _m

_m.print = lambda *a, **k: None
RDLogger.DisableLog("rdApp.*")
WATER = Component(1, "Water", ((16, 1),))
hash100 = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 100
P1 = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "part1_organic_surrogate")
MCM_TSV = P1 + "/mcm_v331_species.tsv"   # written by part1_organic_surrogate/fetch_mcm_species.py
T0 = time.time()


def featurize(name, smi, cat):
    r = smiles_to_components([smi], water_as_component1=True)
    if r.removed:
        return None
    try:
        if not (ActivityModel(r.components).evaluate([0.9, 0.1], 298.15, basis="mass").gamma_neutral > 0).all():
            return None
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smi)
    return {"name": name, "smiles": smi, "canon": Chem.MolToSmiles(mol), "cat": cat,
            "subgroups": [list(map(int, t)) for t in r.components[1].subgroups], "unm": int(r.n_unmatched_atoms[0])}


def bimog_cat(d):
    n, h = d["#N"] != 0, d["#Hal"] != 0
    return "n_and_hal" if n and h else "n_only" if n else "hal_only" if h else "cho"


bim_raw = json.load(open(f"{P1}/bimog_unified.json"))
bimog = [o for d in bim_raw if (o := featurize(d["name"], d["smiles"], bimog_cat(d)))]
bcanon = {o["canon"] for o in bimog}
synth = []
add = lambda s, n: synth.append((n, s))
for n in range(1, 11): add("C" * n + "N", f"amine1-C{n}")
for n in range(1, 7): add("C" * n + "N" + "C" * n, f"amine2s-C{n}")
for n in range(1, 6):
    for m in range(n + 1, 7): add("C" * n + "N" + "C" * m, f"amine2-C{n}C{m}")
for n in range(1, 9): add("C" * n + "[N+](=O)[O-]", f"nitro-C{n}")
add("c1ccccc1[N+](=O)[O-]", "nitrobenzene")
for n in range(1, 5): add("C" * n + "c1ccccc1[N+](=O)[O-]", f"nitroarom-C{n}")
for n in range(1, 11): add("C" * n + "O[N+](=O)[O-]", f"nitrate-C{n}")
for n in range(2, 8):
    add("C" * (n - 1) + "C(O)O[N+](=O)[O-]", f"hn-a-C{n}"); add("OC" + "C" * (n - 2) + "CO[N+](=O)[O-]", f"hn-b-C{n}")
for n in range(2, 8): add("NC" + "C" * (n - 2) + "CO", f"aminoalc-C{n}")
for s, n in [("CC(C)N", "isopropylamine"), ("CC(C)(C)N", "tBuNH2"), ("CC(C)CN", "iBuNH2"), ("CC(N)CC", "sBuNH2"),
             ("C1CCC(CC1)N", "cyhexNH2"), ("C1CCC(C1)N", "cypentNH2"), ("CC(C)(C)CN", "neopentNH2"), ("COCCN", "MeOEtNH2"),
             ("CCOCCCN", "EtOPrNH2"), ("CC(=O)CCN", "aminobutanone"), ("CCOC(=O)CCN", "Et-aminopropanoate"),
             ("NCCOCCN", "bisaminoethylether"), ("NCCCCN", "putrescine"), ("NCCCCCN", "cadaverine"), ("NCCCCCCN", "HMDA"),
             ("NC(C)CN", "diaminopropane"), ("CC(C)[N+](=O)[O-]", "2-nitropropane"), ("C1CCC(CC1)[N+](=O)[O-]", "nitrocyclohexane"),
             ("CC(C)(C)[N+](=O)[O-]", "tBuNO2"), ("CC(=O)CC[N+](=O)[O-]", "nitrobutanone"), ("COCC[N+](=O)[O-]", "MeO-nitroethane"),
             ("CC(=O)CO[N+](=O)[O-]", "nitrooxypropanone"), ("COCCO[N+](=O)[O-]", "MeOEt-nitrate"), ("CC(C)O[N+](=O)[O-]", "iPr-nitrate"),
             ("C1CCC(CC1)O[N+](=O)[O-]", "cyhex-nitrate"), ("CC(C)NC(C)C", "diisopropylamine"), ("C1CCNCC1", "piperidine"),
             ("C1CCNC1", "pyrrolidine")]:
    add(s, n)
synth = [o for n, s in synth if (o := featurize(n, s, "synth_n"))]
mraw = pd.read_csv(MCM_TSV, sep="\t", comment="*")
mcm = []
for _, r in mraw.iterrows():
    mol = Chem.MolFromSmiles(str(r.Smiles))
    if (mol is None or str(r.Excited).lower() == "true" or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            or not {a.GetSymbol() for a in mol.GetAtoms()} <= {"C", "H", "O", "N"} or mol.GetNumHeavyAtoms() < 2
            or "C" not in {a.GetSymbol() for a in mol.GetAtoms()}):
        continue
    o = featurize(r.Name, str(r.Smiles), "mcm")
    if o and o["unm"] == 0 and o["canon"] not in bcanon:
        mcm.append(o)
POOL = bimog + synth + mcm
SRC = ["bimog"] * len(bimog) + ["synth"] * len(synth) + ["mcm"] * len(mcm)
for o, s in zip(POOL, SRC):
    o["src"] = s
    o["hash100"] = hash100(o["smiles"])
    o["M_kg"] = float(ActivityModel([WATER, Component(2, "o", tuple(map(tuple, o["subgroups"])))]).mixture.mmass[1])
VOCAB = sorted({int(s) for o in POOL for s, q in o["subgroups"]})
print(f"BIMOG {len(bimog)}, synthetic N {len(synth)}, MCM {len(mcm)}; vocabulary {len(VOCAB)} ({time.time()-T0:.0f}s)",
      flush=True)

MW = 0.01801528


def x_of_w(w, M):
    a, b = (1 - w) / M, w / MW
    return a / (a + b)


def w_of_x(x, M):
    a, b = (1 - x) * MW, x * M
    return a / (a + b)


rows = []
rng = np.random.RandomState(0)                       # same sampling as the notebook (label(..., seed=0))
for oi, o in enumerate(POOL):
    m = ActivityModel([WATER, Component(2, "o", tuple(map(tuple, o["subgroups"])))])
    e = np.linspace(0.02, 0.98, 21)
    for w in rng.uniform(e[:-1], e[1:]):
        for T in rng.uniform(273.15, 313.15, 4):
            try:
                lg = m.evaluate([w, 1 - w], T, basis="mass").ln_gamma
                if not np.isfinite(lg[:2]).all():
                    continue
                x = x_of_w(w, o["M_kg"])
                h = 1e-5 * min(x, 1 - x)
                lp = m.evaluate([w_of_x(x + h, o["M_kg"]), 1 - w_of_x(x + h, o["M_kg"])], T, basis="mass").ln_gamma
                lm = m.evaluate([w_of_x(x - h, o["M_kg"]), 1 - w_of_x(x - h, o["M_kg"])], T, basis="mass").ln_gamma
                d = (lp[:2] - lm[:2]) / (2 * h)
                rows.append((oi, w, x, (T - 293.15) / 20, lg[0], lg[1], d[0], d[1]))
            except Exception:
                pass
    if oi % 500 == 0:
        print(f"  {oi}/{len(POOL)} ({time.time()-T0:.0f}s)", flush=True)
a = np.array(rows)
CNT = np.zeros((len(POOL), len(VOCAB)), np.float32)
for i, o in enumerate(POOL):
    for s, q in o["subgroups"]:
        CNT[i, VOCAB.index(int(s))] = q
np.savez_compressed("../results/s1_labels.npz", pm=a[:, 0].astype(int), w=a[:, 1], x=a[:, 2], tn=a[:, 3],
                    y=a[:, 4:6], dy=a[:, 6:8], cnt=CNT, vocab=np.array(VOCAB))
json.dump({"pool": POOL, "vocab": VOCAB}, open("../results/s1_pool.json", "w"))
print(f"{len(a)} labelled points ({time.time()-T0:.0f}s)")
