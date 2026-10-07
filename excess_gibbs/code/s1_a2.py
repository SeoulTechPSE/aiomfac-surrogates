"""S1 / A2: paper_2 binaries (T = 289 K) with the S1 G^E release ensemble vs AIOMFAC and vs the Part 1 R_emb release
surrogate (raw).  Binodal by convex hull (AIOMFAC, G^E) or equal activities (R_emb raw), and PhaseEquilibrium runs
at a_w* -/+ 0.0005 (AIOMFAC onset and each model's own onset) plus an RH scan 0.95 ... 0.9999.
Writes ../results/s1_a2.json."""
import json
import time

import numpy as np
import torch

from aiomfac_py.phase_equilibrium import ExplicitLiquidModel, PhaseEquilibrium
from aiomfac_py.s2as import smiles_to_components

import s1_train as S
from s1_llps import binodal
from surrogates import Part1Ensemble
from surrogate_pe import GEPhaseEquilibrium, SurrogateBinaryLiquidModel, SurrogatePhaseEquilibrium

T = 289.0
CAND = {"pinonaldehyde": "CC(=O)C1CC(C=O)C1(C)C", "pinonic_acid": "CC(=O)C1CC(C1(C)C)CC(=O)O",
        "pinic_acid": "CC1(C(CC1C(=O)O)CC(=O)O)C", "terpenylic_acid": "OC(=O)C1(C)CCC(=O)C1CC(=O)O",
        "MBTCA": "OC(=O)CC(C(=O)O)C(C)(C(=O)O)C"}
VOCAB = [int(v) for v in np.load("../results/s1_labels.npz")["vocab"]]
nets = []
for seed in range(3):
    net = S.Net(len(VOCAB), "ge", "silu")
    net.load_state_dict(torch.load(f"../results/s1_models/ge_silu_sob_release_seed{seed}.pt")); net.eval()
    nets.append(net)
ens = Part1Ensemble()
SCAN = [0.95, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.9985, 0.999, 0.9995, 0.9999]


def _rand(pe):
    pe.inner_method = "rand"
    return pe


def lna_of(lm):
    return lambda xs: np.array([lm.ln_a(np.array([1 - x, x]), T)[:2] - lm._c[:2] for x in xs])


def pe_point(make, org, rh):
    pe = make()
    t0 = time.time()
    r = pe.solve({org.name: 1.0}, rh, solids="none")
    return {"rh": rh, "status": r.status, "n_liquids": r.n_liquids,
            "x_org": [float(L.mole_fractions[1]) for L in r.liquids], "n_eval": int(pe.lm.n_eval),
            "n_ood": int(getattr(pe.lm, "n_ood", 0)), "time_s": round(time.time() - t0, 3), "message": r.message}


OUT = {}
for name, smi in CAND.items():
    org = smiles_to_components([smi], names=[name]).components[1]
    cnt = np.zeros(len(VOCAB))
    for s, q in org.subgroups:
        cnt[VOCAB.index(int(s))] = q
    M = float(ExplicitLiquidModel([org], [])._mm[1])
    makers = {"aiomfac": lambda: PhaseEquilibrium([org], [], T_K=T),
              "ge_release": lambda: GEPhaseEquilibrium(org, T, nets, cnt, M),
              "ge_release_rand": lambda: _rand(GEPhaseEquilibrium(org, T, nets, cnt, M)),
              "remb_release_raw": lambda: SurrogatePhaseEquilibrium(org, T, ens, mode="raw")}
    res = {"binodal": {}, "pe": {}}
    for tag, mk in makers.items():
        res["binodal"][tag] = (res["binodal"]["ge_release"] if tag == "ge_release_rand"
                               else binodal(lna_of(mk().lm), tag != "remb_release_raw"))
    for tag, mk in makers.items():
        pts = {}
        for label in ("aiomfac", tag.replace("_rand", "")):
            b = res["binodal"][label]
            if isinstance(b, dict):
                for d in (-0.0005, 0.0005):
                    pts[f"{label}_onset{d:+.4f}"] = pe_point(mk, org, min(b["aw"] + d, 0.99995))
        pts["scan"] = [pe_point(mk, org, rh) for rh in SCAN]
        res["pe"][tag] = pts
    OUT[name] = res
    line = f"{name:16s}"
    for tag in makers:
        b = res["binodal"][tag]
        line += f" | {tag}: " + (f"aw*={b['aw']:.5f} x'={b['x1']:.2e} x''={b['x2']:.3f}" if isinstance(b, dict) else str(b))
    print(line, flush=True)
    for tag in makers:
        sc = res["pe"][tag]["scan"]
        print(f"    {tag:17s} scan: " + " ".join(f"{p['rh']}:{p['n_liquids']}{'' if p['status']=='converged' else '*'}"
                                             f"({p['x_org'][0]:.3f})" for p in sc), flush=True)
json.dump(OUT, open("../results/s1_a2.json", "w"), indent=1)
print("S1 A2 DONE")
