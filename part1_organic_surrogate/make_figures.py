import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "/tmp/claude-0/-home-claude/eecda288-448a-508f-8b60-1c1c2ce4350c/scratchpad/surrogate_paper/figs"
plt.rcParams.update({"font.size": 10, "figure.dpi": 200})

# ---------- Figure 1: Table 1 -- training-pool growth MAE ----------
pools = ["A: CHO-only\n(n=243)", "B: full BIMOG\n(n=368)", "C: +homologous\n+diverse N (n=468)"]
mae_n_only = [np.nan, 0.476, 0.561]
mae_n_hal  = [np.nan, 0.598, 0.375]
mae_cho    = [0.230, 0.208, 0.204]

x = np.arange(len(pools))
w = 0.25
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.bar(x - w, mae_cho, width=w, label="held-out CHO-only MAE", color="#4C72B0")
ax.bar(x,     mae_n_only, width=w, label="held-out N-containing MAE", color="#DD8452")
ax.bar(x + w, mae_n_hal, width=w, label="held-out N+halogen MAE", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(pools)
ax.set_ylabel("Generalization MAE (log10 activity coeff.)")
ax.set_title("Figure 1. Surrogate MAE vs. training-pool composition (Table 1)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_pool_growth_mae.png")
plt.close(fig)

# ---------- Figure 2: Table 2 -- Model A-G comparison ----------
models = ["A. Hand-crafted\nMLP", "B. GNN,\nno pretrain", "C. GNN,\nsynth-pretrained",
          "D. Hybrid,\nno pretrain", "E. Hybrid,\nGNN pretrained\n(overfit)",
          "F. Hybrid, frozen\npretrained+reg", "G. Hybrid, frozen\nrandom encoder"]
mae_water   = [0.025, 0.080, 0.061, 0.098, 0.210, 0.102, 0.072]
mae_water_e = [0.0,   0.007, 0.008, 0.015, 0.082, 0.009, 0.007]
mae_organic   = [0.298, 0.847, 0.715, 0.791, 2.465, 1.039, 0.791]
mae_organic_e = [0.0,   0.138, 0.075, 0.090, 0.175, 0.044, 0.036]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=False)
colors = ["#4C72B0"] + ["#888888"]*6
colors[0] = "#4C72B0"  # best hand-crafted model highlighted

axes[0].barh(models, mae_water, xerr=mae_water_e, color=colors)
axes[0].invert_yaxis()
axes[0].set_xlabel("Gen. MAE (log10 activity coeff., water)")
axes[0].set_title("(a) Water activity coefficient")
axes[0].grid(axis="x", alpha=0.3)

axes[1].barh(models, mae_organic, xerr=mae_organic_e, color=colors)
axes[1].invert_yaxis()
axes[1].set_yticklabels([])
axes[1].set_xlabel("Gen. MAE (log10 activity coeff., organic)")
axes[1].set_title("(b) Organic activity coefficient")
axes[1].grid(axis="x", alpha=0.3)

fig.suptitle("Figure 2. Feature-representation comparison, Models A-G (Table 2)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_model_comparison.png")
plt.close(fig)

# ---------- Figure 3: Table 3 -- viscosity-pipeline MAE ----------
vmodels = ["Naive global-mean\nbaseline", "Hand-crafted MLP,\nwithout Tg", "Hand-crafted MLP,\nwith Tg (best)", "VTF-coefficient\ndirect prediction\n(failed)"]
vmae = [2.267, 1.286, 0.823, 14.8]
vmae_e = [0.0, 0.019, 0.020, 5.5]
vcolors = ["#AAAAAA", "#DD8452", "#4C72B0", "#C44E52"]

fig, ax = plt.subplots(figsize=(6.2, 4.2))
bars = ax.bar(vmodels, vmae, yerr=vmae_e, color=vcolors, capsize=4)
ax.set_ylabel("Generalization MAE (log10 Pa s)")
ax.set_title("Figure 3. Viscosity-surrogate architecture comparison (Table 3)")
ax.set_yscale("log")
ax.grid(axis="y", which="both", alpha=0.3)
for b, v in zip(bars, vmae):
    ax.text(b.get_x()+b.get_width()/2, v*1.15, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3_viscosity_mae.png")
plt.close(fig)

# ---------- Figure 4: Sobol GSA (real results from aiomfac_gsa/sobol_results.json) ----------
with open("/tmp/claude-0/-home-claude/eecda288-448a-508f-8b60-1c1c2ce4350c/scratchpad/aiomfac_gsa/sobol_results.json") as f:
    sob = json.load(f)

names = sob["factor_names"]
targets = [k for k in sob.keys() if k.startswith("ln_gamma")]
print("targets:", targets)

fig, axes = plt.subplots(1, len(targets), figsize=(6.5*len(targets), 5.5), sharey=True)
if len(targets) == 1:
    axes = [axes]
for ax, tgt in zip(axes, targets):
    S1 = np.array(sob[tgt]["S1"])
    ST = np.array(sob[tgt]["ST"])
    order = np.argsort(ST)[::-1]
    y = np.arange(len(names))
    ax.barh(y, ST[order], color="#C44E52", alpha=0.6, label="Total-order (ST)")
    ax.barh(y, S1[order], color="#4C72B0", label="First-order (S1)")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index")
    ax.set_title(tgt.replace("ln_gamma_", "ln γ ("+")").replace("(water)","(water)") if False else tgt)
    ax.grid(axis="x", alpha=0.3)
axes[0].legend(fontsize=9, loc="lower right")
fig.suptitle("Figure 4. Sobol global sensitivity indices for the AIOMFAC-surrogate activity coefficients")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4_sobol_gsa.png")
plt.close(fig)

print("wrote figures to", FIG_DIR)

# ---------- Figure 5: speed benchmark (Table 4) ----------
with open(f"{FIG_DIR}/../timing_results.json") as f:
    timing = json.load(f)

labels = ["AIOMFAC,\ndirect call", "Surrogate (PyTorch),\none-at-a-time", "Surrogate (PyTorch),\nbatched",
          "Surrogate (NumPy),\none-at-a-time", "Surrogate (NumPy),\nbatched"]
per_call_us = [
    timing["aiomfac_direct"]["median_s"] * 1e6,
    timing["surrogate_single_call"]["median_s"] * 1e6,
    timing["surrogate_batched"]["per_point_s"] * 1e6,
    timing["surrogate_numpy_single_call"]["median_s"] * 1e6,
    timing["surrogate_numpy_batched"]["per_point_s"] * 1e6,
]
colors5 = ["#C44E52", "#DD8452", "#4C72B0", "#55A868", "#8172B2"]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
bars = axes[0].bar(range(len(labels)), per_call_us, color=colors5)
axes[0].set_xticks(range(len(labels)))
axes[0].set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
axes[0].set_yscale("log")
axes[0].set_ylabel("Median time per evaluation (microseconds)")
axes[0].set_title("(a) Per-evaluation cost")
axes[0].grid(axis="y", which="both", alpha=0.3)
for b, v in zip(bars, per_call_us):
    axes[0].text(b.get_x() + b.get_width()/2, v*1.3, f"{v:.2f}", ha="center", fontsize=8)

scaling = timing["surrogate_batch_scaling_per_point_s"]
ns = sorted(int(k) for k in scaling.keys())
us_per_point = [scaling[str(n)] * 1e6 for n in ns]
axes[1].plot(ns, us_per_point, "o-", color="#4C72B0", label="Surrogate, PyTorch (batched)")
axes[1].axhline(timing["aiomfac_direct"]["median_s"] * 1e6, color="#C44E52", ls="--", label="AIOMFAC (direct, per-call)")
axes[1].axhline(timing["surrogate_numpy_batched"]["per_point_s"] * 1e6, color="#55A868", ls=":",
                 label="Surrogate, NumPy (batched)")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Batch size (number of points per surrogate call)")
axes[1].set_ylabel("Time per evaluation (microseconds)")
axes[1].set_title("(b) Surrogate cost vs. batch size")
axes[1].legend(fontsize=8)
axes[1].grid(True, which="both", alpha=0.3)

fig.suptitle(f"Figure 5. Evaluation speed: AIOMFAC direct vs. surrogate "
             f"(up to {timing['speedup_numpy_batched_vs_aiomfac']:.0f}x speedup, NumPy + batched)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig5_speed_benchmark.png")
plt.close(fig)
print("wrote fig5_speed_benchmark.png")

# ---------- Figure 6: GPU benchmark run (Table 5), measured by the user on Colab T4 ----------
with open(f"{FIG_DIR}/../timing_results_gpu_run.json") as f:
    gpu = json.load(f)

labels6 = ["AIOMFAC,\ndirect call", "PyTorch,\none-at-a-time", "PyTorch,\nbatched",
           "NumPy,\none-at-a-time", "NumPy,\nbatched", "GPU (T4),\none-at-a-time", "GPU (T4),\nbatched"]
per_call_us6 = [
    gpu["aiomfac_direct_us"], gpu["surrogate_pytorch_single_us"], gpu["surrogate_pytorch_batched_us"],
    gpu["surrogate_numpy_single_us"], gpu["surrogate_numpy_batched_us"],
    gpu["surrogate_gpu_single_us"], gpu["surrogate_gpu_batched_us"],
]
colors6 = ["#C44E52", "#DD8452", "#4C72B0", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
bars = ax.bar(range(len(labels6)), per_call_us6, color=colors6)
ax.set_xticks(range(len(labels6)))
ax.set_xticklabels(labels6, fontsize=8, rotation=20, ha="right")
ax.set_yscale("log")
ax.set_ylabel("Median time per evaluation (microseconds)")
ax.set_title("Figure 6. GPU benchmark run (Google Colab, NVIDIA T4 GPU runtime, user-executed)\n"
              "-- note: different, slower-CPU hardware than Figure 5; not directly comparable in absolute terms")
ax.grid(axis="y", which="both", alpha=0.3)
for b, v in zip(bars, per_call_us6):
    ax.text(b.get_x() + b.get_width()/2, v*1.3, f"{v:.2f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig6_gpu_benchmark.png")
plt.close(fig)
print("wrote fig6_gpu_benchmark.png")
