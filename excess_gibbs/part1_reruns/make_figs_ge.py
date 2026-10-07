"""Figures 1, 2, 3, 5, 6 and 7 of the excess-Gibbs-energy (G^E) manuscript (Fig. 4 unchanged: figures_revised)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_H = __import__("pathlib").Path(__file__).resolve().parent
# P: Part 1 v2 material (anova_real_molecules.csv from part1_organic_surrogate/results; param_gsa_raw.npz, 373 MB, is
# produced by part1_organic_surrogate/04_sensitivity_viscosity_analyses.ipynb and not archived); fig4_viscosity.png is
# the unchanged viscosity figure of v2 (part1_organic_surrogate/figures).
P = str(_H.parents[1] / "part1_organic_surrogate" / "results") + "/"; G = str(_H / "results") + "/"; OUT = str(_H / "figures") + "/"
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
t1 = pd.read_csv(G + "GE_table2_pools_runs.csv")
cats = [("gen_org_cho", "CHO"), ("gen_org_n_only", "N only"), ("gen_org_hal_only", "Halogen only"), ("gen_org_n_and_hal", "N + halogen")]
pools = ["A: CHO only", "B: full BIMOG", "C: B + synthetic N"]
fig, ax = plt.subplots(figsize=(6.3, 3.0)); w = 0.26
for j, pool in enumerate(pools):
    g = t1[t1.pool == pool]
    if pool.startswith("A"): g = g.assign(**{c: np.nan for c, _ in cats[1:]})   # the CHO-only pool is compared on CHO molecules only
    m = [g[c].mean() if c in g and g[c].notna().any() else np.nan for c, _ in cats]
    s = [g[c].std(ddof=1) if c in g and g[c].notna().any() else np.nan for c, _ in cats]
    ax.bar(np.arange(4) + (j - 1) * w, m, w * 0.92, yerr=s, color=C[j], label=pool, capsize=2, error_kw=EB)
ax.set_xticks(range(4), [l for _, l in cats]); ax.set_ylabel("Generalization MAE, ln γ$_{org}$"); ax.set_ylim(0, None)
ax.legend(title="Training pool", fontsize=8, title_fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
plt.savefig(OUT + "fig1_training_pools.png"); plt.close()

# ---------- Fig 2: representations and models on BIMOG (Part II configuration) ----------
t3 = pd.read_csv(G + "GE_table3_runs.csv").groupby("model")
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
lcC = pd.concat([pd.read_csv(G + "GE_learning_curve_remb_runs.csv").query("seed < 2"), pd.read_csv(G + "GE_learning_curve_remb_s2_runs.csv")])
u4 = pd.concat([pd.read_csv(G + "GE_table4_gnn_runs.csv").rename(columns={"n": "n_train_mol"}),
                pd.read_csv(G + "GE_learning_curve_desc_runs.csv").assign(model="R_desc MLP")])
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

# ---------- Fig 6: literature compounds (representation C, M1 vs M3) ----------
z = np.load(G + "GE_literature_curves.npz"); W19 = z["W19"]
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
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(OUT + "fig6_literature_compounds.png"); plt.close()


# ---------- Fig 5: sensitivity analyses (panel a: G^E R_desc surrogate; b, c: AIOMFAC directly, unchanged) ----------
import re
gsa = pd.read_csv(G + "GE_sobol_descriptor_indices.csv"); gsa = gsa[gsa.N == 4096]
anova = pd.read_csv(P + "anova_real_molecules.csv")
d = np.load(P + "param_gsa_raw.npz"); fA, fB, fAB, pnames = d["fA"], d["fB"], d["fAB"], d["names"]
Vv = np.var(np.concatenate([fA, fB]), 0); STn = 0.5 * np.mean((fA[None] - fAB) ** 2, 1)
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
    g = gsa[gsa.output == out].sort_values("ST", ascending=True).tail(8); y = np.arange(len(g))
    ax.barh(y + 0.2, g.ST, 0.38, xerr=g.ST_conf, color=C[0], label="S$_T$", capsize=1.5, error_kw={"lw": 0.7})
    ax.barh(y - 0.2, g.S1.clip(lower=0), 0.38, xerr=g.S1_conf, color=C[1], label="S$_1$", capsize=1.5, error_kw={"lw": 0.7})
    ax.set_yticks(y, [LBL.get(f, f) for f in g.factor]); ax.set_xlabel(f"Sobol index, {lab}")
    if k == 0:
        ax.legend(fontsize=7.5, loc="lower right"); panel(ax, "(a)")
ax = fig.add_subplot(gsp[1, :])
comps = ["S_molecule", "S_w", "S_T", "interactions"]
bars = anova.assign(lbl=anova.pool + ", " + anova.output.map({"ln_gamma_water": "ln γ$_w$", "ln_gamma_organic": "ln γ$_{org}$"}))
left = np.zeros(len(bars))
for c_, comp, nm in zip([C[0], C[1], C[3], MUTED], comps, ["molecule", "composition", "temperature", "interactions"]):
    ax.barh(bars.lbl, bars[comp].clip(lower=0), left=left, color=c_, label=nm, height=0.6, edgecolor="white", lw=1)
    left += bars[comp].clip(lower=0)
ax.set_xlim(0, 1); ax.set_xlabel("Share of variance (real molecules, AIOMFAC)"); ax.invert_yaxis()
ax.legend(ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0)); panel(ax, "(b)")
for k, lab in enumerate(["ln γ$_w$", "ln γ$_{org}$"]):
    ax = fig.add_subplot(gsp[2, k]); keep = Vv[:, k] > 1e-8
    st = (STn[:, keep, k] / Vv[keep, k]).mean(1); o = np.argsort(-st)[:8][::-1]
    ax.barh([pname(pnames[i]) for i in o], st[o], color=C[2], height=0.6); ax.set_xlabel(f"Mean S$_T$ over outputs, {lab}")
    if k == 0:
        panel(ax, "(c)")
plt.savefig(OUT + "fig5_sensitivity.png"); plt.close()
print("figures 1, 2, 3, 5, 6 written to", OUT)
