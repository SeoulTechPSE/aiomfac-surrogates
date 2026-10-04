"""Figures 1-4 and 6 for the representation-C manuscript (Fig. 5 unchanged, from make_figs.py)."""
import re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path
HERE = Path(__file__).resolve().parent
P = str(HERE / "results") + "/"; OUT = str(HERE / "figures") + "/"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
panel = lambda ax, s: ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, color=INK)
EB = {"lw": 0.8, "ecolor": INK}

# ---------- Fig 1: training pools (representation C) ----------
t1 = pd.read_csv(P + "C_table1_runs.csv")
cats = [("gen_org_cho", "CHO"), ("gen_org_n_only", "N only"), ("gen_org_hal_only", "Halogen only"), ("gen_org_n_and_hal", "N + halogen")]
pools = ["A: CHO only", "B: full BIMOG", "C: B + synthetic N"]
fig, ax = plt.subplots(figsize=(6.3, 3.0)); w = 0.26
for j, pool in enumerate(pools):
    g = t1[t1.pool == pool]
    if pool.startswith("A"): g = g.assign(**{c: np.nan for c, _ in cats[1:]})   # the CHO-only pool is compared on CHO molecules only
    m = [g[c].mean() if c in g and g[c].notna().any() else np.nan for c, _ in cats]
    s = [g[c].std(ddof=1) if c in g and g[c].notna().any() else np.nan for c, _ in cats]
    ax.bar(np.arange(4) + (j - 1) * w, m, w * 0.92, yerr=s, color=C[j], label=pool, capsize=2, error_kw=EB)
ax.set_xticks(range(4), [l for _, l in cats]); ax.set_ylabel("Generalization MAE, ln γ$_{org}$"); ax.set_ylim(0, 0.36)
ax.legend(title="Training pool", fontsize=8, title_fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
plt.savefig(OUT + "fig1_training_pools.png"); plt.close()

# ---------- Fig 2: representations and models on BIMOG (Part II configuration) ----------
t3 = pd.read_csv(P + "table3_all_models_runs.csv").groupby("model")
labels = ["A$_{emb}$", "A$_{desc}$", "B", "C", "D", "E", "F", "G"]
keys = ["A_emb", "A_desc", "B", "C", "D", "E", "F", "G"]
water = [(t3.gen_w.mean()[k], t3.gen_w.std(ddof=1)[k]) for k in keys]
org = [(t3.gen_org.mean()[k], t3.gen_org.std(ddof=1)[k]) for k in keys]
cols = [C[0], C[2], C[6], C[6], C[1], C[1], C[1], C[1]]
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
for ax, data, lab, bat, tag in [(axes[0], water, "ln γ$_w$", 0.130, "(a)"), (axes[1], org, "ln γ$_{org}$", 1.388, "(b)")]:
    ax.bar(range(8), [d[0] for d in data], 0.7, yerr=[d[1] for d in data], color=cols, capsize=2, error_kw=EB)
    ax.set_xticks(range(8), labels, fontsize=7.5); ax.axhline(bat, color=INK, lw=1, ls="--")
    ax.text(7.4, bat, "BAT", va="bottom", ha="right", fontsize=8); ax.set_ylabel(f"Generalization MAE, {lab}"); panel(ax, tag)
fig.legend(handles=[Patch(color=C[0], label="MLP, learned subgroup embedding"), Patch(color=C[2], label="MLP, hand-crafted descriptors"),
                        Patch(color=C[6], label="GNN"), Patch(color=C[1], label="GNN + descriptor hybrid")], fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.12, 1, 1)); plt.savefig(OUT + "fig2_representations_bimog.png"); plt.close()

# ---------- Fig 3: learning curves ----------
lcC = pd.read_csv(P + "C_learning_curve_runs.csv"); u4 = pd.read_csv(P + "U_table4_desc_gnn_runs.csv").rename(columns={"n": "n_train_mol"})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
for ax, test, bat, tag in [(axes[0], "MCM", 1.093, "(a)"), (axes[1], "BIMOG", 0.805, "(b)")]:
    series = [(lcC[lcC.regime == "MCM only"], f"{test} org", C[0], "o", "-", "Embedding MLP, MCM only"),
              (lcC[lcC.regime == "BIMOG train + MCM"], f"{test} org", C[1], "D", "-", "Embedding MLP, BIMOG train + MCM"),
              (u4[u4.model == "GNN"], f"{test} org", C[6], "s", "--", "GNN, MCM only"),
              (u4[u4.model == "R_desc MLP"], f"{test} org", C[2], "^", ":", "Descriptor MLP, MCM only")]
    for df, col, c, mk, ls, lab in series:
        g = df.groupby("n_train_mol")[col].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], color=c, marker=mk, ms=4, ls=ls, lw=1.5, capsize=2, label=lab)
    ax.axhline(bat, color=INK, lw=1, ls="-."); ax.axvline(312, color=MUTED, lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    yt = [0.03, 0.1, 0.3, 1.0] if test == "MCM" else [0.1, 0.3, 1.0, 2.0]
    ax.set_yticks(yt, [f"{v:g}" for v in yt]); ax.minorticks_off(); ax.set_xticks([100, 300, 1000, 3000], ["100", "300", "1000", "3000"])
    ax.set_xlabel("Training molecules"); ax.set_ylabel("Generalization MAE, ln γ$_{org}$")
    ax.set_title("Held-out MCM molecules" if test == "MCM" else "Held-out BIMOG molecules", fontsize=9, color=INK); panel(ax, tag)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(OUT + "fig3_mcm_learning_curves.png"); plt.close()

# ---------- Fig 4: viscosity ----------
vC = pd.read_csv(P + "C_viscosity_runs.csv").groupby("inputs")["gen MAE"].agg(["mean", "std"])
vC_dry = pd.read_csv(P + "C_viscosity_runs.csv").groupby("inputs")["gen MAE dry near-glassy"].mean()
vis = [("Global mean", 2.267, np.nan, MUTED), ("T$_g$ only", 1.388, 0.008, C[3]), ("Descriptors (B)", 1.267, 0.034, C[2]),
       ("T$_g$ + O:C", 1.160, 0.020, C[3]), ("Embedding (C)", vC.loc["structure (C) only", "mean"], vC.loc["structure (C) only", "std"], C[0]),
       ("Descriptors (B) + T$_g$", 0.801, 0.044, C[2]), ("Embedding (C) + T$_g$", vC.loc["structure (C) + Tg", "mean"], vC.loc["structure (C) + Tg", "std"], C[0])]
prop = pd.read_csv(P + "viscosity_tg_sensitivity.csv")
prop["bin"] = pd.cut(prop["T - Tg"], [-200, 0, 50, 100, 400], labels=["T < Tg", "0–50", "50–100", "> 100"])
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
ax.barh([v[0] for v in vis], [v[1] for v in vis], xerr=[v[2] for v in vis], color=[v[3] for v in vis], height=0.6, capsize=2, error_kw=EB)
ax.set_xlabel("Generalization MAE (log$_{10}$ Pa s)"); ax.invert_yaxis(); panel(ax, "(a)")
ax = axes[1]; best = vC.loc["structure (C) + Tg", "mean"]
for k, wv in enumerate((0.05, 0.2, 0.5)):
    g = prop[prop.w_water == wv].groupby("bin", observed=True)["Δ (Tg+10 K)"].apply(lambda s: np.median(np.abs(s)))
    ax.plot(range(len(g)), g.values, marker="o", ms=4.5, color=C[k], label=f"w$_{{water}}$ = {wv}")
dry = vC_dry.loc["structure (C) + Tg"]
ax.axhline(best, color=INK, lw=1, ls="--"); ax.text(3, best + 0.03, "emulator MAE, all", ha="right", fontsize=7.5)
ax.axhline(dry, color=INK, lw=1, ls=":"); ax.text(3, dry + 0.03, "emulator MAE, dry and near T$_g$", ha="right", fontsize=7.5)
ax.set_xticks(range(4), ["T < T$_g$", "0–50", "50–100", "> 100"]); ax.set_xlabel("T − T$_g$ (K)")
ax.set_ylabel("Median |Δlog$_{10}$η| for T$_g$ + 10 K"); ax.set_ylim(0, 2.0)
ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.6)); panel(ax, "(b)")
plt.tight_layout(); plt.savefig(OUT + "fig4_viscosity.png"); plt.close()

# ---------- Fig 6: literature compounds (representation C, M1 vs M3) ----------
z = np.load(P + "C_literature_curves.npz"); W19 = z["W19"]
names = ["Glycerol", "1,2,4-Butanetriol", "2,5-Hexanediol", "Ethanol", "2-Propanol", "Malonic acid"]
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True)
for ax, nm in zip(axes.flat, names):
    for k, (lab, c) in enumerate([("ln γ$_w$", C[0]), ("ln γ$_{org}$", C[1])]):
        ax.plot(W19, z[f"truth|{nm}"][:, k], color=c, lw=2, label=f"{lab}, AIOMFAC")
        for mn, ls in [("M1", "--"), ("M3", ":")]:
            p = np.mean([z[f"{mn}|{s}|{nm}"][:, k] for s in range(3)], 0)
            ax.plot(W19, p, color=c, lw=1.6, ls=ls, label=f"{lab}, {mn}")
    ax.set_title(nm, fontsize=9)
for ax in axes[1]: ax.set_xlabel("Water mass fraction")
for ax in axes[:, 0]: ax.set_ylabel("ln γ")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(OUT + "fig6_literature_compounds_reeval.png"); plt.close()
# ---------- Fig 5: sensitivity analyses ----------
gsa = pd.read_csv(P + "sobol_descriptor_indices.csv")
gsa = gsa[gsa.N == 4096]
anova = pd.read_csv(P + "anova_real_molecules.csv")
pg = pd.read_csv(P + "param_gsa_meanST.csv")
GROUP = {"1": "CHn", "7": "H₂O", "69": "OH", "9": "ketone", "13": "ether", "72": "CH₂OOH", "73": "C(O)OOH",
         "8": "ACOH", "3": "ACH", "20": "COOH", "67": "CHn[alc]", "2": "C=C", "71": "CH₂ONO₂", "10": "CHO"}
def pname(s):
    a, b = re.findall(r"(\d+):", s)
    return f"{GROUP.get(a, a)} → {GROUP.get(b, b)}"
LBL = {"wtf_water": "w$_{water}$", "log1p_total_q": "size, ln(1+q)", "o_to_c": "O:C", "frac_carboxylic_COOH": "COOH",
       "log_molar_mass": "ln M", "frac_hydroxyl_OH": "OH", "frac_ether_CH2O": "ether", "frac_ketone_CH2CO": "ketone",
       "frac_ester_CCOO": "ester", "frac_polyol_CH": "CH[OH]", "frac_amine_CNH": "amine", "n_to_c": "N:C",
       "T_norm": "T", "frac_aromatic_ACH": "ACH"}

fig = plt.figure(figsize=(6.3, 7.4))
gsp = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.75, 1.15], hspace=0.65, wspace=0.75)
for k, (out, lab) in enumerate([("ln_gamma_water", "ln γ$_w$"), ("ln_gamma_organic", "ln γ$_{org}$")]):
    ax = fig.add_subplot(gsp[0, k])
    g = gsa[gsa.output == out].sort_values("ST", ascending=True).tail(8)
    y = np.arange(len(g))
    ax.barh(y + 0.2, g.ST, 0.38, xerr=g.ST_conf, color=C[0], label="S$_T$", capsize=1.5, error_kw={"lw": 0.7})
    ax.barh(y - 0.2, g.S1.clip(lower=0), 0.38, xerr=g.S1_conf, color=C[1], label="S$_1$", capsize=1.5,
            error_kw={"lw": 0.7})
    ax.set_yticks(y, [LBL.get(f, f) for f in g.factor]); ax.set_xlabel(f"Sobol index, {lab}")
    if k == 0:
        ax.legend(fontsize=7.5, loc="lower right"); panel(ax, "(a)")
ax = fig.add_subplot(gsp[1, :])
labels, comps = [], ["S_molecule", "S_w", "S_T", "interactions"]
bars = anova.assign(lbl=anova.pool + ", " + anova.output.map({"ln_gamma_water": "ln γ$_w$", "ln_gamma_organic": "ln γ$_{org}$"}))
left = np.zeros(len(bars))
for c_, comp, nm in zip([C[0], C[1], C[3], MUTED], comps, ["molecule", "composition", "temperature", "interactions"]):
    ax.barh(bars.lbl, bars[comp].clip(lower=0), left=left, color=c_, label=nm, height=0.6, edgecolor="white", lw=1)
    left += bars[comp].clip(lower=0)
ax.set_xlim(0, 1); ax.set_xlabel("Share of variance (real molecules, AIOMFAC)"); ax.invert_yaxis()
ax.legend(ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0)); panel(ax, "(b)")
for k, lab in enumerate(["ln γ$_w$", "ln γ$_{org}$"]):
    ax = fig.add_subplot(gsp[2, k])
    g = pg[pg.output == ["ln_gamma_water", "ln_gamma_organic"][k]].sort_values("mean_ST").tail(8)
    ax.barh([pname(n) for n in g.parameter], g.mean_ST, color=C[2], height=0.6)
    ax.set_xlabel(f"Mean S$_T$ over outputs, {lab}")
    if k == 0:
        panel(ax, "(c)")
plt.savefig(OUT + "fig5_sensitivity.png"); plt.close()


print("figures written to", OUT)
