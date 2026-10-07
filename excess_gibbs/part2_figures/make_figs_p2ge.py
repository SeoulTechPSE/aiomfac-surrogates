"""Figures of the revised Part 2 (excess-Gibbs-energy surrogates) from the result files of research/paper_5."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "results") + "/"
OLD = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "part2_inorganic_surrogate" / "results") + "/"
OUT = str(__import__("pathlib").Path(__file__).resolve().parent / "figures") + "/"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
panel = lambda ax, s: ax.text(-0.1, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, color=INK)
lab = lambda c: (c.replace("ln_gamma_", "").replace("HSO4", "HSO₄").replace("HCO3", "HCO₃").replace("--", "²⁻")
                 .replace("2+", "²⁺").replace("+", "⁺").replace("-", "⁻").replace("H2O", "H₂O").replace("NH4", "NH₄")
                 .replace("NO3", "NO₃").replace("SO4", "SO₄").replace("CO3", "CO₃").replace("IO3", "IO₃"))
SP = ["H2O", "organic", "cation", "anion"]

# Fig 1: per-species MAE
ps = pd.read_csv(R + "P2GE_per_species.csv", index_col=0)
e2e = pd.read_csv(R + "P2GE_e2e_per_species.csv", index_col=0)
old = pd.read_csv(OLD + "P2_per_species_mae.csv", index_col=0)
fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.0))
x = np.arange(15); w = 0.38
ax = axes[0]
ax.bar(x - w / 2, ps["direct test"], w, color=C[1], label="unconstrained network")
ax.bar(x + w / 2, ps["gex test"], w, color=C[0], label="mole-fraction G$^E$")
ax.set_xticks(x, [lab(c) for c in ps.index], fontsize=8, rotation=45, ha="right"); ax.set_ylabel("MAE, ln γ"); ax.legend(fontsize=8)
ax.set_title("Free-species activity coefficients, new compositions", fontsize=9); panel(ax, "(a)")
ax = axes[1]
ax.bar(x - w / 2, old["corrected interp"].reindex(e2e.index), w, color=C[6], label="earlier surrogate (nominal molalities)")
ax.bar(x + w / 2, e2e["gex interp"], w, color=C[0], label="mole-fraction G$^E$ + speciation")
ax.set_xticks(x, [lab(c) for c in e2e.index], fontsize=8, rotation=45, ha="right"); ax.set_ylabel("MAE, ln γ"); ax.legend(fontsize=8)
ax.set_title("End to end from nominal compositions (earlier interpolation test set)", fontsize=9); panel(ax, "(b)")
plt.tight_layout(); plt.savefig(OUT + "p2_fig1_species_mae.png"); plt.close()

# Fig 2: Sobol S1, surrogate vs AIOMFAC
gsa = pd.read_csv(R + "P2GE_sobol.csv"); gsa = gsa[gsa.source != "mae"]
SALT = {"m_Na+NO3-": "NaNO₃", "m_Na+Cl-": "NaCl", "m_K+NO3-": "KNO₃", "m_K+Cl-": "KCl", "m_Na+HSO4-": "NaHSO₄",
        "m_Na+HCO3-": "NaHCO₃", "m_Ca2+IO3-": "Ca(IO₃)₂", "T_K": "T"}
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
for ax, (case, out, t) in zip(axes, [("NaCl+NaNO3+KCl+KNO3", "ln_gamma_Na+", "(a)"),
                                     ("Ca(IO3)2+NaHSO4+NaHCO3", "ln_gamma_HSO4-", "(b)"),
                                     ("Ca(IO3)2+NaHSO4+NaHCO3", "ln_gamma_Ca2+", "(c)")]):
    g = gsa[(gsa.case == case) & (gsa.output == out)].pivot(index="factor", columns="source", values="S1")
    y = np.arange(len(g))
    ax.barh(y + 0.2, g["AIOMFAC"], 0.38, color=C[2], label="AIOMFAC"); ax.barh(y - 0.2, g["surrogate"], 0.38, color=C[0], label="surrogate")
    ax.set_yticks(y, [SALT.get(f, f) for f in g.index], fontsize=8); ax.set_xlabel(f"S₁, ln γ({lab(out)})"); panel(ax, t)
axes[0].legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig(OUT + "p2_fig2_sobol.png"); plt.close()

# Fig 3: phase equilibrium of inorganic particles
pe = json.load(open(R + "s2v2_pe_gex.json"))
names = ["(NH4)2SO4", "NaCl", "NH4NO3", "NH4HSO4"]
fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4), sharey=False)
for ax, nm in zip(axes, names):
    for tag, c, mk, ls, lb in (("aiomfac", INK, "o", "-", "AIOMFAC"), ("gex", C[0], "s", "--", "G$^E$ surrogate"),
                               ("direct", C[1], "x", ":", "unconstrained")):
        if tag not in pe[nm]:
            continue
        rows = pe[nm][tag]
        rh = np.array([r["rh"] for r in rows]); w = np.array([r["water_mol"] for r in rows])
        ok = np.array([r["status"] in ("converged", "dry") for r in rows])
        liq = w > 0
        if tag == "direct":
            ax.plot(rh[liq & ok], w[liq & ok], ls="none", color=c, marker="x", ms=4, label=lb + " (converged)")
            ax.plot(rh[liq & ~ok], w[liq & ~ok], ls="none", color=c, marker="o", mfc="none", ms=4, label=lb + " (not converged)")
            nconv = ok.sum()
        else:
            ax.plot(rh[liq], w[liq], ls=ls, color=c, marker=mk, ms=3.5, lw=1.2 if tag != "aiomfac" else 1.8, label=lb)
    ax.set_title(lab(nm).replace("(NH₄)2SO₄", "(NH₄)₂SO₄"), fontsize=9)
    ax.text(0.04, 0.95, f"unconstrained:\n{nconv}/12 converged", transform=ax.transAxes, va="top", fontsize=7, color=C[1])
    ax.set_xlabel("RH"); ax.set_yscale("log"); ax.set_xlim(0.27, 0.98); ax.set_ylim(1e-3, 100)
axes[0].set_ylabel("Particle water (mol)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
plt.tight_layout(); plt.savefig(OUT + "p2_fig3_phase_equilibrium.png"); plt.close()

# Fig 4: Option C
C_res = pd.read_csv(R + "s3_optionC_runs.csv"); summ = C_res.groupby("case", sort=False).mean(numeric_only=True)
per_org = pd.read_csv(R + "s3_per_organic.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
cases = ["interpolation", "unseen salt", "unseen organic", "unseen organic and salt"]
sp_lbl = {"H2O": "H₂O", "organic": "organic", "cation": "cation", "anion": "anion"}
ax = axes[0]; x = np.arange(4); w = 0.2
for k, s in enumerate(SP):
    ax.bar(x + (k - 1.5) * w, [summ.loc[c, f"{s} C"] / summ.loc[c, f"{s} A"] for c in cases], w, color=C[k], label=sp_lbl[s])
ax.set_xticks(x, ["interp.", "unseen\nsalt", "unseen\norganic", "unseen\nboth"], fontsize=8)
ax.set_ylabel("MAE ratio, Option C / Option A"); ax.legend(fontsize=7, ncol=2); panel(ax, "(a)")
ax = axes[1]
for nm, mk, c in [("calibration", "o", C[6]), ("test", "s", C[1])]:
    g = per_org[per_org.set == nm]
    ax.scatter(g.nn_dist, g["reduction_%"], marker=mk, s=22, color=c, label=f"{nm} organics")
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xlabel("Distance to nearest training organic"); ax.set_ylabel("Organic error reduction (%)")
ax.legend(fontsize=7); panel(ax, "(b)")
plt.tight_layout(); plt.savefig(OUT + "p2_fig4_option_c.png"); plt.close()

# Fig 5: worked examples
eth = pd.read_csv(R + "s3_worked_ethanol_kcl.csv"); mal = pd.read_csv(R + "s3_worked_malonic.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
ax = axes[0]
for p_, c, ls, nm in [("t", INK, "-", "AIOMFAC"), ("a", C[1], "--", "Option A"), ("c", C[0], ":", "Option C")]:
    ax.plot(eth.wo, np.exp(eth[f"{p_}_lngpm"]), color=c, ls=ls, lw=2, marker="o" if p_ == "t" else None, ms=3, label=nm)
ax.set_xlabel("Ethanol mass fraction (KCl mass fraction 0.03)"); ax.set_ylabel("γ± (KCl)"); ax.legend(fontsize=8); panel(ax, "(a)")
ax = axes[1]
for k, s in enumerate(SP):
    ax.scatter(mal[f"t_{s}"], mal[f"a_{s}"], marker="x", s=18, color=C[k])
    ax.scatter(mal[f"t_{s}"], mal[f"c_{s}"], marker="o", s=14, color=C[k],
               label=sp_lbl[s].replace("cation", "NH₄⁺").replace("anion", "SO₄²⁻"))
v = mal[[f"t_{s}" for s in SP]].to_numpy(); lim = [v.min() - 0.2, v.max() + 0.2]
ax.plot(lim, lim, color=MUTED, lw=0.8); ax.set_xlabel("AIOMFAC ln γ (mole-fraction scale)")
ax.set_ylabel("Predicted ln γ (×: A, ○: C)"); ax.legend(fontsize=7); panel(ax, "(b)")
plt.tight_layout(); plt.savefig(OUT + "p2_fig5_worked_examples.png"); plt.close()

# Fig 6: Lopes et al. (1999)
lop = pd.read_csv(R + "s3_lopes_comparison.csv")
fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.5), sharey=True)
for ax, (xe, g) in zip(axes, lop.groupby("ethanol_wt")):
    ax.plot(g.m, g.exp, "o", color=INK, ms=4, label="Lopes et al. (1999)")
    ax.plot(g.m, g.AIOMFAC, "-", color=C[2], lw=2, label="AIOMFAC")
    ax.plot(g.m, g["Option C"], ":", color=C[0], lw=2, label="Option C")
    ax.plot(g.m, g["Option A"], "--", color=C[1], lw=1.5, label="Option A")
    ax.set_title(f"{int(round(100 * xe))} % ethanol", fontsize=9); ax.set_xlabel("m(KCl) / mol kg⁻¹")
axes[0].set_ylabel("γ± (mixed-solvent reference)"); axes[0].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(OUT + "p2_fig6_lopes.png"); plt.close()
print("figures written")
