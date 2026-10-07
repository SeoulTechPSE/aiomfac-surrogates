"""Table 4 / Fig. 3 with G^E heads: R_emb learning curves and transfer (notebook 02 cell 10) and the R_desc MLP on
the MCM subsets (notebook 01 cell 9, R_desc part).  The GNN rows are in ge_gnn_mcm.py."""
import sys

import numpy as np
import pandas as pd

from ge_common import *

isB = SRC == "bimog"; isM = SRC == "mcm"
mcm_tr = np.where(isM & ~TEST)[0]; mcm_te = isM & TEST
b_tr = np.where(isB & ~TEST)[0]; b_te = isB & TEST
which = sys.argv[1] if len(sys.argv) > 1 else "remb"
SEEDS = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
SUF = f"_s{sys.argv[2].replace(',', '')}" if len(sys.argv) > 2 else ""
rows = []
for seed in SEEDS:
    order = np.random.RandomState(100 + seed).permutation(mcm_tr)
    if which == "remb":
        configs = [("MCM only", order[:n]) for n in (100, 200, 400, 800, 1600, len(order))] + \
                  [("BIMOG train + MCM", np.r_[b_tr, order[:n]]) for n in (0, 400, 1600, len(order))]
        rep = "R_emb"
    else:
        configs = [("MCM only", order[:n]) for n in (400, len(order))]
        rep = "R_desc"
    for reg, sub in configs:
        m = np.zeros(len(POOL), bool); m[sub] = True
        pred, ep = fit_ge(rep, pts_of(m), seed, max_epochs=300)
        rec = {"model": f"{rep} MLP", "regime": reg, "n_train_mol": len(sub), "seed": seed, "epochs": ep}
        for tname, tmask in [("MCM", mcm_te), ("BIMOG", b_te)]:
            g = pts_of(tmask); e = np.abs(pred(PM[g], EXTRA[g]) - Y[g])
            rec[f"{tname} org"] = e[:, 1].mean(); rec[f"{tname} w"] = e[:, 0].mean()
        rows.append(rec)
        print(f"  seed {seed} {rep} {reg:18s} n={len(sub):5d}: MCM {rec['MCM org']:.3f} BIMOG {rec['BIMOG org']:.3f} "
              f"({time.time()-T0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(f"results/GE_learning_curve_{which}{SUF}_runs.csv", index=False)
df = pd.DataFrame(rows)
print(df.groupby(["regime", "n_train_mol"])[["MCM org", "BIMOG org", "MCM w"]].agg(fmt).to_string())
print(f"DONE t4 {which} ({time.time()-T0:.0f}s)")
