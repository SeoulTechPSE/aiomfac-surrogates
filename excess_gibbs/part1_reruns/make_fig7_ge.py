"""Figure 7 (consistency and phase behaviour) from research/paper_5 S1 results."""
import json
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "code"))
import os
os.chdir(_HERE.parent / "code")
import s1_train as S
from aiomfac_py.io import Component
from aiomfac_py.phase_equilibrium import ExplicitLiquidModel

OUT = str(_HERE / "figures") + "/"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.8, "savefig.dpi": 300, "savefig.bbox": "tight"})
panel = lambda ax, s: ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, color=INK)


def net(m, seed):
    kind, act = ("remb", "relu") if m == "remb_relu" else ("ge", "silu")
    n = S.Net(S.CNT.shape[1], kind, act); n.load_state_dict(torch.load(f"../results/s1_models/{m}_bimog_seed{seed}.pt")); n.eval()
    return n


T = 298.15
o = next(p for p in S.POOL if p["name"] == "acetone")
cnt = torch.tensor(S.CNT[S.POOL.index(o)])
xs = np.unique(np.concatenate([np.logspace(-4, -1, 200), np.linspace(0.1, 0.995, 400)]))
lm = ExplicitLiquidModel([Component(2, "acetone", tuple(map(tuple, o["subgroups"])))], [])
la_ref = np.array([lm.ln_a(np.array([1 - x, x]), T)[0] for x in xs])
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
ax = axes[0]
ax.plot(xs, la_ref, color=INK, lw=2.2, label="AIOMFAC")
for mname, c, lab in (("remb_relu", C[1], "unconstrained (ReLU)"), ("ge_silu", C[0], "excess Gibbs energy")):
    n = net(mname, 0); x = torch.tensor(xs, dtype=torch.float32).requires_grad_(True)
    lg, _ = n(cnt.expand(len(xs), -1), x, torch.full((len(xs),), (T - 293.15) / 20), torch.full((len(xs),), o["M_kg"]))
    ax.plot(xs, lg.detach().numpy()[:, 0] + np.log(1 - xs), color=c, label=lab)
ax.set_xlabel("Organic mole fraction x$_{o}$"); ax.set_ylabel("ln a$_w$"); ax.set_title("Acetone (held out), 298.15 K", fontsize=9)
ax.legend(fontsize=7.5); panel(ax, "(a)")
ins = ax.inset_axes([0.52, 0.42, 0.45, 0.36])
for mname, c in (("remb_relu", C[1]), ("ge_silu", C[0])):
    n = net(mname, 0); x = torch.tensor(xs, dtype=torch.float32).requires_grad_(True)
    lg, _ = n(cnt.expand(len(xs), -1), x, torch.full((len(xs),), (T - 293.15) / 20), torch.full((len(xs),), o["M_kg"]))
    la = lg.detach().numpy()[:, 0] + np.log(1 - xs)
    ins.plot(xs, np.gradient(la, xs), color=c, lw=1.2)
ins.plot(xs, np.gradient(la_ref, xs), color=INK, lw=1.2); ins.axhline(0, color=MUTED, lw=0.6)
ins.set_xlim(0.1, 0.99); ins.set_ylim(-6, 1); ins.text(0.04, 0.06, "d ln a$_w$/dx$_o$", transform=ins.transAxes, fontsize=7); ins.tick_params(labelsize=6)
ax = axes[1]
R = json.load(open("../results/s1_llps.json"))
for mname, c, mk, lab in (("remb_relu", C[1], "s", "unconstrained (ReLU)"), ("ge_silu_sob", C[0], "o", "excess Gibbs energy + derivative loss")):
    a_, b_ = [], []
    for r in R:
        if isinstance(r["aiomfac"], dict):
            for b in r[mname]:
                if isinstance(b, dict) and b["aw"] < 1 and r["aiomfac"]["aw"] < 1:
                    a_.append(1 - r["aiomfac"]["aw"]); b_.append(1 - b["aw"])
    ax.scatter(a_, b_, s=9, color=c, marker=mk, alpha=0.6, label=lab, lw=0)
lim = [1e-7, 0.1]
ax.plot(lim, lim, color=INK, lw=0.8); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("1 − a$_w^*$, AIOMFAC"); ax.set_ylabel("1 − a$_w^*$, surrogate"); ax.set_title("LLPS onset, held-out molecules", fontsize=9)
ax.axhline(1e-6, color=MUTED, lw=0.6, ls=":")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, markerscale=2.5, handletextpad=0.3); panel(ax, "(b)")
plt.tight_layout(); plt.savefig(OUT + "fig7_consistency_llps.png"); plt.close()
print("fig7 written")
