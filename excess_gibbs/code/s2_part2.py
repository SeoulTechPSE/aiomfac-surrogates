"""Revised Part 2, electrolyte part (Sects. 3.1-3.5): the free-species excess-Gibbs-energy surrogate (S2, label set
s2v2) evaluated as in the earlier Part 2 analysis, with the dissociation equilibria solved by speciation.py.

  t1   accuracy on free-species labels (s2v2_runs_*.csv from s2_train.py) and per-species errors
  e2e  end-to-end on the Part 2 nominal-composition data (training_data.parquet; the interpolation and
       unseen-combination test sets of Part 2), surrogate + speciation vs AIOMFAC; also AIOMFAC's own free-species
       activity model + speciation.py (fidelity of the speciation step)
  fort Fortran reference cases c005, c006
  bis  bisulfate speciation of the 300 closed-form samples of Part 2
  cost single-thread cost
  sobol first- and total-order Sobol indices, surrogate vs AIOMFAC

Usage: python s2_part2.py [task ...].  Writes ../results/P2GE_*.csv / .json."""
import contextlib
import io
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("S2_TAG", "s2v2")
P2 = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "part2_inorganic_surrogate")
sys.path.insert(0, P2); sys.path.insert(0, P2 + "/fortran_reference")
import generate_training_data as gtd                                     # noqa: E402
import s2_train as S2                                                     # noqa: E402
import s2_data as SD                                                      # noqa: E402
import speciation as SPC                                                  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
OUT = "../results/"
NAMES = S2.NAMES
OUT_COLS = ["ln_gamma_H2O"] + [f"ln_gamma_{n}" for n in NAMES]
fmt = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
T0 = time.time()


def load(kind, seed, release=False):
    net = {"gex": S2.GEXNet, "ge": S2.GENet, "direct": S2.DirectNet}[kind]()
    net.load_state_dict(torch.load(f"../results/s2v2_models/{kind}{'_release' if release else ''}_seed{seed}.pt"))
    return net.double().eval()


def model_fn(nets):
    """(m (14,), T) -> (ln a_w, ln gamma_i molal (14,)), mean over the members (mean of G for the G^E models)."""
    def f(m, T):
        A, b = S2.dh_params([T])
        args = (torch.tensor(np.asarray(m, float))[None, :], torch.tensor([(T - 293.15) / 20]), torch.tensor(A), torch.tensor(b))
        aw = gi = 0
        for net in nets:
            a_, g_ = net(*args, create_graph=False)
            aw, gi = aw + float(a_.detach()[0]), gi + g_.detach().numpy()[0]
        return aw / len(nets), gi / len(nets)
    return f


def aiomfac_free(m, T):
    lgw, lgi = SD.ln_gamma(np.asarray(m, float), T)
    return lgw + math.log(1.0 / (1.0 + SPC.MW * float(np.sum(m)))), lgi


def outputs(res):
    """speciation result -> AIOMFAC-style outputs: ln gamma_w (mole fraction, dissociated basis) and ion ln gamma."""
    m = res["m"]
    lnxw = -math.log1p(SPC.MW * float(m.sum()))
    return np.concatenate([[res["ln_a_w"] - lnxw], res["ln_gamma"]])


# ---------------------------------------------------------------- t1
def task_t1():
    runs = pd.concat([pd.read_csv(OUT + f"s2v2_runs_{k}.csv") for k in ("direct", "ge", "gex")])
    runs.to_csv(OUT + "P2GE_table1_runs.csv", index=False)
    cols = ["mae_lng_ion", "mae_lnaw", "gd_rel_median", "mae_lng_ion_unseen", "mae_lnaw_unseen"]
    print(runs.groupby("model")[cols].agg(lambda s: f"{s.mean():.4f} ± {s.std(ddof=1):.4f}").to_string())
    # per-species MAE (random test and unseen combinations), seed means, for direct and gex
    M, T = S2.M, S2.T
    HOLD = [("Ca2+", "CO3--"), ("NH4+", "HSO4-"), ("K+", "IO3-")]
    held = np.zeros(len(M), bool)
    for a_, b_ in HOLD:
        held |= (M[:, NAMES.index(a_)] > 0) & (M[:, NAMES.index(b_)] > 0)
    rest = np.flatnonzero(~held); perm = np.random.RandomState(123).permutation(rest)
    te, tu = perm[:int(0.15 * len(rest))], np.flatnonzero(held)
    ps = {}
    for kind in ("direct", "gex"):
        for split, idx in (("test", te), ("unseen", tu)):
            acc = []
            for seed in range(5):
                net = load(kind, seed)
                m, tn, A, b, aw, yi, pr = S2.tens(idx)
                a_, g_ = net(m.double(), tn.double(), A.double(), b.double(), create_graph=False)
                e = np.abs(g_.detach().numpy() - yi.numpy()); p = pr.numpy()
                ew = np.abs(a_.detach().numpy() - aw.numpy())
                acc.append(np.concatenate([[ew.mean()], [e[p[:, j], j].mean() if p[:, j].any() else np.nan for j in range(14)]]))
            ps[f"{kind} {split}"] = np.nanmean(np.stack(acc), 0)
    pd.DataFrame(ps, index=OUT_COLS).to_csv(OUT + "P2GE_per_species.csv")
    print(pd.DataFrame(ps, index=OUT_COLS).round(3).to_string())


# ---------------------------------------------------------------- e2e
def part2_split():
    df = pd.read_parquet(P2 + "/training_data.parquet")
    df = df[df[OUT_COLS[1:]].notna().any(axis=1)].reset_index(drop=True)
    HELD = [frozenset({"Ca2+", "CO3--"}), frozenset({"NH4+", "HSO4-"}), frozenset({"K+", "IO3-"})]
    combo = df.apply(lambda r: frozenset(n for n in NAMES if r[f"m_{n}"] > 0), axis=1)
    is_gen = combo.isin(HELD).values
    rest = np.where(~is_gen)[0]; p = np.random.RandomState(0).permutation(len(rest))
    return df, rest[p[:int(0.1 * len(rest))]], np.where(is_gen)[0]


def e2e_errors(fn, df, idx):
    Y = df[OUT_COLS].to_numpy(float); rows = []
    for i in idx:
        r = df.iloc[i]
        res = SPC.speciate(fn, r[[f"m_{n}" for n in NAMES]].to_numpy(float), float(r["T_K"]))
        P = outputs(res)
        ok = np.isfinite(Y[i])
        rows.append((np.where(ok, np.abs(P - Y[i]), np.nan), res["converged"]))
    E = np.stack([r[0] for r in rows]); conv = np.array([r[1] for r in rows])
    per = np.nanmean(E, 0)
    return {"pooled": np.nanmean(E), "species_mean": np.nanmean(per[np.isfinite(per)]), "median": np.nanmedian(E),
            "converged": float(conv.mean())}, per


def task_e2e():
    df, interp, gen = part2_split()
    rows, per = [], {}
    fid = {}
    for split, idx in (("interp", interp), ("gen", gen)):
        r, p = e2e_errors(aiomfac_free, df, idx); fid[split] = r; per[f"AIOMFAC-free {split}"] = p
        print(f"AIOMFAC free-species + speciation vs AIOMFAC ({split}, n = {len(idx)}): {r}", flush=True)
    for kind in ("direct", "gex"):
        for seed in range(5):
            fn = model_fn([load(kind, seed)])
            for split, idx in (("interp", interp), ("gen", gen)):
                r, p = e2e_errors(fn, df, idx)
                rows.append({"model": kind, "seed": seed, "split": split, **r}); per.setdefault(f"{kind} {split}", []).append(p)
            print(f"  {kind} seed {seed} ({time.time()-T0:.0f}s): {rows[-2]} {rows[-1]}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT + "P2GE_e2e_runs.csv", index=False)
    json.dump(fid, open(OUT + "P2GE_e2e_fidelity.json", "w"), indent=1)
    pd.DataFrame({k: (np.nanmean(np.stack(v), 0) if isinstance(v, list) else v) for k, v in per.items()},
                 index=OUT_COLS).to_csv(OUT + "P2GE_e2e_per_species.csv")
    print(R.groupby(["model", "split"])[["pooled", "species_mean", "median", "converged"]].agg(fmt).to_string())


# ---------------------------------------------------------------- fort
def task_fort():
    from aiomfac_py import ActivityModel, read_input_file
    from reference_utils import REF, parse_dump
    SUB2ION = {s: n for n, s, _ in gtd.CATIONS + gtd.ANIONS}
    rows = []
    for c in ("c005", "c006"):
        case = read_input_file(REF / "carb_sulf" / "inputs" / f"input_{c}.txt")
        pts = parse_dump(REF / "carb_sulf" / "dumps" / f"debug_terms_{c}.txt.gz")
        sr = ActivityModel(case.components).mixture.sr
        for k, p in enumerate(pts):
            fr = np.asarray(case.fractions[k], float)
            w_water = 1 - fr[1:].sum() if len(fr) == len(case.components) else 1 - fr.sum()
            fr_s = fr[1:] if len(fr) == len(case.components) else fr
            tot = {n: 0.0 for n in NAMES}
            for comp, f in zip(case.components[1:], fr_s):
                mw = sum(gtd.ION_MW[SUB2ION[s]] * q for s, q in comp.subgroups)
                for s, q in comp.subgroups:
                    tot[SUB2ION[s]] += f / w_water / mw * q
            mn = np.array([tot[n] for n in NAMES]); T = float(case.T_K[k])
            truth = {"ln_gamma_H2O": p["lnactcoeff_n"][0]}
            for j, s in enumerate(sr.cations):
                truth[f"ln_gamma_{SUB2ION[s]}"] = p["lnactcoeff_c"][j]
            for j, s in enumerate(sr.anions):
                truth[f"ln_gamma_{SUB2ION[s]}"] = p["lnactcoeff_a"][j]
            preds = {"AIOMFAC-free": outputs(SPC.speciate(aiomfac_free, mn, T))}
            for seed in range(5):
                preds[f"gex release {seed}"] = outputs(SPC.speciate(model_fn([load("gex", seed, True)]), mn, T))
            for src, P in preds.items():
                for col, t in truth.items():
                    if t > -9000 and col in OUT_COLS:
                        rows.append({"case": c, "point": k, "source": src, "species": col, "truth": t,
                                     "pred": float(P[OUT_COLS.index(col)])})
    F = pd.DataFrame(rows); F["abs_err"] = (F.pred - F.truth).abs(); F.to_csv(OUT + "P2GE_fortran.csv", index=False)
    g = F[F.source.str.startswith("gex")].groupby(["case", "source"]).abs_err.mean().groupby("case").agg(fmt)
    print("surrogate + speciation vs Fortran:", g.to_dict())
    print("AIOMFAC free-species + speciation vs Fortran:", F[F.source == "AIOMFAC-free"].groupby("case").abs_err.mean().round(4).to_dict())
    print(F[F.source.str.startswith("gex")].groupby("species").abs_err.mean().round(3).sort_values().to_string())


# ---------------------------------------------------------------- bis
def task_bis():
    import closed_form_pipeline as cfp
    with contextlib.redirect_stdout(io.StringIO()):
        samples = cfp.make_samples(300, seed=0)
    rows = []
    for seed in range(5):
        fn = model_fn([load("gex", seed, True)])
        e = []
        for s in samples:
            row = s["mlp_input_row"]
            mn = np.array([row.get(f"m_{n}", 0.0) for n in NAMES]); T = s["T_K"]
            res = SPC.speciate(fn, mn, T)
            m_hso4 = res["m"][NAMES.index("HSO4-")] * res["n_w"] * SPC.MW        # per initial kg water
            e.append((abs(m_hso4 - s["m_hso4_true"]), max(s["m_h_max"], s["m_sulf_max"])))
        e = np.array(e)
        rows.append({"seed": seed, "mae_m_hso4": e[:, 0].mean(), "rel_err_mean_%": 100 * (e[:, 0] / e[:, 1]).mean(),
                     "rel_err_median_%": 100 * np.median(e[:, 0] / e[:, 1])})
    # fidelity: AIOMFAC's free-species model in the same speciation step
    ef = []
    for s in samples:
        row = s["mlp_input_row"]; mn = np.array([row.get(f"m_{n}", 0.0) for n in NAMES])
        res = SPC.speciate(aiomfac_free, mn, s["T_K"])
        ef.append(abs(res["m"][NAMES.index("HSO4-")] * res["n_w"] * SPC.MW - s["m_hso4_true"]))
    R = pd.DataFrame(rows); R.to_csv(OUT + "P2GE_bisulfate.csv", index=False)
    print(R.drop(columns="seed").agg(fmt).to_string(), f"\nAIOMFAC-free speciation: max |dm_HSO4| {max(ef):.2e}")
    json.dump({"aiomfac_free_max_abs": float(max(ef)), "n": len(samples)}, open(OUT + "P2GE_bisulfate_fidelity.json", "w"))


# ---------------------------------------------------------------- cost
def task_cost():
    from aiomfac_py import ActivityModel
    torch.set_num_threads(1)

    def med(fn, n, warm=5):
        for _ in range(warm):
            fn()
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
        return float(np.median(t))
    rng = np.random.default_rng(1)
    comps, fr, tot, _ = gtd._build_system({("Na+", "Cl-"): 1.0}, 298.15); mA = ActivityModel(comps)
    t_fixed = med(lambda: mA.evaluate([0.0] + list(fr), 298.15, basis="mass"), 300)
    combos = [gtd.sample_combo(rng) for _ in range(300)]

    def aio(sm_T):
        c, f, t, _ = gtd._build_system(*sm_T); ActivityModel(c).evaluate([0.0] + list(f), sm_T[1], basis="mass")
    t_vary = float(np.median([med(lambda s=s: aio(s), 1, warm=0) for s in combos]))
    vec = lambda sm_T: np.array([gtd._build_system(*sm_T)[2][n] for n in NAMES])
    t_vec = float(np.median([med(lambda s=s: vec(s), 1, warm=0) for s in combos]))
    net = load("gex", 0, True); fn = model_fn([net])
    m1 = vec(combos[0]); t_one = med(lambda: fn(m1, 298.15), 500)
    t_e2e = float(np.median([med(lambda s=s: SPC.speciate(fn, vec(s), s[1]), 1, warm=0) for s in combos]))
    flags = np.array([any(SPC.flags(vec(s))) for s in combos])
    MB = torch.tensor(np.stack([vec(s) for s in combos] * 34)[:10000]); TB = np.full(len(MB), 298.15)
    A, b = S2.dh_params(TB)
    argsB = (MB, torch.full((len(MB),), 0.25), torch.tensor(A), torch.tensor(b))
    t_batch = med(lambda: net(*argsB, create_graph=False), 10, warm=2) / len(MB)
    cost = {"fixed": t_fixed, "vary": t_vary, "vec": t_vec, "one": t_one, "batch": t_batch, "e2e_vary": t_e2e,
            "frac_speciated": float(flags.mean())}
    json.dump(cost, open(OUT + "P2GE_cost.json", "w"), indent=1)
    for k, v in cost.items():
        print(f"{k:15s} {1e6*v if k != 'frac_speciated' else v:10.2f}")
    print(f"speed-up varying combination: one at a time (no speciation) {t_vary/(t_vec+t_one):.1f}x, with speciation "
          f"{t_vary/t_e2e:.1f}x, batched {t_vary/(t_vec+t_batch):.0f}x; fixed system: one at a time {t_fixed/(t_vec+t_one):.1f}x, "
          f"batched {t_fixed/(t_vec+t_batch):.1f}x")


# ---------------------------------------------------------------- sobol
def task_sobol():
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample
    CASES = [("NaCl", [("Na+", "Cl-")], ["ln_gamma_H2O", "ln_gamma_Na+"], 512),
             ("NaCl+NaNO3+KCl+KNO3", [("Na+", "Cl-"), ("Na+", "NO3-"), ("K+", "Cl-"), ("K+", "NO3-")],
              ["ln_gamma_H2O", "ln_gamma_Na+", "ln_gamma_K+"], 256),
             ("Ca(IO3)2+NaHSO4+NaHCO3", [("Ca2+", "IO3-"), ("Na+", "HSO4-"), ("Na+", "HCO3-")],
              ["ln_gamma_H2O", "ln_gamma_Ca2+", "ln_gamma_HSO4-"], 256)]
    fn = model_fn([load("gex", s, True) for s in range(5)])
    gsa = []
    for name, salts, outs, N in CASES:
        prob = {"num_vars": len(salts) + 1, "names": [f"m_{c}{a}" for c, a in salts] + ["T_K"],
                "bounds": [[-3.0, 0.3]] * len(salts) + [[275.0, 325.0]]}
        Xs = sobol_sample.sample(prob, N, calc_second_order=False, seed=0)
        truth = np.full((len(Xs), 15), np.nan); pred = np.full((len(Xs), 15), np.nan)
        for i, row in enumerate(Xs):
            sm = {s: 10 ** row[k] for k, s in enumerate(salts)}
            c, f, tot, _ = gtd._build_system(sm, row[-1]); r = gtd._evaluate(c, f, row[-1])
            if r is not None:
                truth[i] = [r[0].get(col, np.nan) for col in OUT_COLS]
            pred[i] = outputs(SPC.speciate(fn, np.array([tot[n] for n in NAMES]), row[-1]))
        for col in outs:
            j = OUT_COLS.index(col)
            for src, Y in (("surrogate", pred[:, j]), ("AIOMFAC", truth[:, j])):
                Y = np.where(np.isfinite(Y), Y, np.nanmean(Y))
                Si = sobol_analyze.analyze(prob, Y.astype(float), calc_second_order=False, seed=0)
                for k, f in enumerate(prob["names"]):
                    gsa.append({"case": name, "output": col, "source": src, "factor": f, "S1": Si["S1"][k],
                                "S1_conf": Si["S1_conf"][k], "ST": Si["ST"][k]})
            gsa.append({"case": name, "output": col, "source": "mae", "factor": "", "S1": float(np.nanmean(np.abs(pred[:, j] - truth[:, j])))})
        print(f"  {name}: {np.isnan(truth[:, OUT_COLS.index(outs[0])]).sum()} failed AIOMFAC points of {len(Xs)} "
              f"({time.time()-T0:.0f}s)", flush=True)
    G = pd.DataFrame(gsa); G.to_csv(OUT + "P2GE_sobol.csv", index=False)
    print(G[G.source != "mae"].pivot_table(index=["case", "output", "factor"], columns="source", values=["S1", "ST"]).round(3).to_string())
    print(G[G.source == "mae"][["case", "output", "S1"]].to_string())


if __name__ == "__main__":
    for t in sys.argv[1:] or ["t1", "e2e", "fort", "bis", "cost", "sobol"]:
        print(f"===== {t}", flush=True)
        globals()[f"task_{t}"]()
