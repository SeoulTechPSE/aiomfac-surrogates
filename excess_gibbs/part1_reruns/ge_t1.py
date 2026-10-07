"""Table 1 (hash split and CHO cross-validation), Table 2 (training pools) and Sect. 3.3 (coverage cross-validation)
with G^E heads; protocols of notebooks 01 (cells 5, 7) and 02 (cells 4, 8)."""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from ge_common import *

isB = SRC == "bimog"
task = sys.argv[1] if len(sys.argv) > 1 else "all"

if task in ("hash", "all"):
    t1 = []
    for pname, pmask in [("CHO", isB & (CAT == "cho")), ("BIMOG", isB)]:
        for rep in REPS:
            for seed in range(5):
                pred, ep = fit_ge(rep, pts_of(pmask & ~TEST), seed, interp_split=True)
                g = pts_of(pmask & TEST); e = np.abs(pred(PM[g], EXTRA[g]) - Y[g])
                t1.append({"pool": pname, "rep": rep, "seed": seed, "gen_w": e[:, 0].mean(), "gen_org": e[:, 1].mean(),
                           "epochs": ep})
            print(f"  hash {pname} {rep} done ({time.time()-T0:.0f}s)", flush=True)
    t1 = pd.DataFrame(t1); t1.to_csv("results/GE_table1_hash_runs.csv", index=False)
    print(t1.groupby(["pool", "rep"], sort=False)[["gen_w", "gen_org"]].agg(fmt).to_string(), flush=True)

if task in ("cv", "all"):
    cho_ids = np.where(isB & (CAT == "cho"))[0]; fold = H % 5
    cv = []
    for rep in REPS:
        for seed in range(3):
            per = {}
            for k in range(5):
                trm = np.zeros(len(POOL), bool); trm[cho_ids[fold[cho_ids] != k]] = True
                tem = np.zeros(len(POOL), bool); tem[cho_ids[fold[cho_ids] == k]] = True
                pred, _ = fit_ge(rep, pts_of(trm), 1000 * seed + k, interp_split=True)
                g = pts_of(tem); e = np.abs(pred(PM[g], EXTRA[g]) - Y[g])[:, 1]
                for m, v in pd.Series(e).groupby(PM[g]).mean().items():
                    per[m] = v
            cv.append({"rep": rep, "seed": seed, "oof_org": np.mean(list(per.values())), **{f"m{m}": v for m, v in per.items()}})
        print(f"  CV {rep} done ({time.time()-T0:.0f}s)", flush=True)
    cv = pd.DataFrame(cv); cv.to_csv("results/GE_table1_cv_runs.csv", index=False)
    print(cv.groupby("rep", sort=False)["oof_org"].agg(fmt).to_string())
    mcols = [c for c in cv.columns if c.startswith("m")]
    mm = cv.groupby("rep")[mcols].mean().T
    for a, b in [("R_emb", "R_desc"), ("R_desc", "R_count"), ("R_count'", "R_desc")]:
        d = mm[a] - mm[b]
        print(f"{a} vs {b}: mean d {d.mean():+.3f} ({100*d.mean()/mm[b].mean():+.0f}%), {a} better for {(d<0).mean()*100:.0f}% "
              f"of molecules, Wilcoxon p = {stats.wilcoxon(mm[a], mm[b]).pvalue:.2g}", flush=True)

if task in ("pools", "all"):
    POOLS = {"A: CHO only": isB & (CAT == "cho"), "B: full BIMOG": isB, "C: B + synthetic N": isB | (SRC == "synth")}
    t2 = []
    for pname, pmask in POOLS.items():
        for seed in range(5):
            pred, ep = fit_ge("R_emb", pts_of(pmask & ~TEST), seed, interp_split=True)
            g = pts_of(isB & TEST); e = np.abs(pred(PM[g], EXTRA[g]) - Y[g])
            rec = {"pool": pname, "seed": seed, "gen_w": e[:, 0].mean(), "gen_org": e[:, 1].mean(), "epochs": ep}
            for c in ["cho", "n_only", "hal_only", "n_and_hal"]:
                sel = CAT[PM[g]] == c
                if sel.any():
                    rec[f"gen_org_{c}"] = e[sel, 1].mean()
            t2.append(rec)
        print(f"  pools {pname} done ({time.time()-T0:.0f}s)", flush=True)
    t2 = pd.DataFrame(t2); t2.to_csv("results/GE_table2_pools_runs.csv", index=False)
    cols = ["gen_w", "gen_org", "gen_org_cho", "gen_org_n_only", "gen_org_hal_only", "gen_org_n_and_hal"]
    print(pd.DataFrame({p: {c: fmt(g[c].dropna()) if g[c].notna().any() else "-" for c in cols}
                        for p, g in t2.groupby("pool", sort=False)}).T.to_string(), flush=True)

if task in ("coverage", "all"):
    fold = H % 5; oof = {}
    for seed in range(3):
        for k in range(5):
            trm, tem = isB & (fold != k), isB & (fold == k)
            pred, _ = fit_ge("R_emb", pts_of(trm), 1000 * seed + k)
            g = pts_of(tem); e = np.abs(pred(PM[g], EXTRA[g]) - Y[g])[:, 1]
            for oi, v in pd.Series(e).groupby(PM[g]).mean().items():
                oof.setdefault(oi, []).append(v)
        print(f"  coverage seed {seed} done ({time.time()-T0:.0f}s)", flush=True)
    per = pd.DataFrame({"mol": list(oof), "err": [np.mean(v) for v in oof.values()]})
    per["cat"] = CAT[per.mol]; per["unmatched"] = [POOL[m]["unm"] > 0 for m in per.mol]
    print(per.groupby("cat").err.agg(["mean", "count"]).round(3).to_string())
    print(per.groupby("unmatched").err.agg(["mean", "count"]).round(3).to_string())
    bids = np.where(isB)[0]
    Z = CNT[bids]; Z = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-6, 1, Z.std(0)); pos = {m: i for i, m in enumerate(bids)}
    per["nn"] = [np.min(np.linalg.norm(Z[[pos[j] for j in bids if fold[j] != fold[m]]] - Z[pos[m]], axis=1)) for m in per.mol]
    per["mag"] = [np.abs(Y[PM == m, 1]).mean() for m in per.mol]
    A_ = np.column_stack([np.ones(len(per)), per.nn, per.mag, per.unmatched.astype(float)]); y_ = per.err.values
    beta = np.linalg.lstsq(A_, y_, rcond=None)[0]; rs = np.random.RandomState(0)
    boots = np.array([np.linalg.lstsq(A_[i], y_[i], rcond=None)[0] for i in [rs.randint(0, len(y_), len(y_)) for _ in range(2000)]])
    lo, hi = np.percentile(boots, [2.5, 97.5], 0)
    print(f"error = {beta[0]:.3f} + {beta[1]:.3f} nn [{lo[1]:.3f},{hi[1]:.3f}] + {beta[2]:.3f} |lng| [{lo[2]:.3f},{hi[2]:.3f}] "
          f"+ {beta[3]:+.3f} unmatched [{lo[3]:+.3f},{hi[3]:+.3f}]", flush=True)
    per.to_csv("results/GE_coverage_oof.csv", index=False)
print(f"DONE {task} ({time.time()-T0:.0f}s)")
