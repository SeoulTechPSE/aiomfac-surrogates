"""Benchmark: direct AIOMFAC evaluation vs. the trained hand-crafted-feature surrogate,
for the same binary water-organic activity-coefficient calls used throughout the paper.
Produces Table 4 (timing) and Figure 5 (speedup bar chart) for the surrogate paper.
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/tmp/claude-0/-home-claude/eecda288-448a-508f-8b60-1c1c2ce4350c/scratchpad/nb_exec_test3/aiomfac-python/src")
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components

OUT_DIR = "/tmp/claude-0/-home-claude/eecda288-448a-508f-8b60-1c1c2ce4350c/scratchpad/surrogate_paper"
GSA_DIR = "/tmp/claude-0/-home-claude/eecda288-448a-508f-8b60-1c1c2ce4350c/scratchpad/aiomfac_gsa"


# ---------------------------------------------------------------------------
# 1) Load the trained surrogate (same checkpoint delivered to the user: Table 1
#    row B / Table 2 Model A -- hand-crafted features, full 368-molecule BIMOG pool)
# ---------------------------------------------------------------------------
with open(f"{GSA_DIR}/surrogate_meta.json") as f:
    meta = json.load(f)
norm = np.load(f"{GSA_DIR}/surrogate_norm.npz")
x_mean, x_std = norm["x_mean"], norm["x_std"]
y_mean, y_std = norm["y_mean"], norm["y_std"]
n_features = meta["n_features"]
main_groups = meta["main_groups"]


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_hidden=128, n_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


model = SurrogateMLP(n_features)
state = torch.load(f"{GSA_DIR}/surrogate_weights.pt", map_location="cpu")
# try loading directly; if the architecture guess above doesn't match, report shapes for debugging
try:
    model.load_state_dict(state)
except Exception as e:
    print("state_dict keys:", list(state.keys()))
    print("shapes:", {k: tuple(v.shape) for k, v in state.items()})
    raise
model.eval()
print("surrogate loaded OK,", sum(p.numel() for p in model.parameters()), "parameters")
torch.set_num_threads(1)  # single-thread CPU, same as AIOMFAC's single-threaded numpy solve -- a fair, like-for-like comparison

# ---------------------------------------------------------------------------
# 2) Build N_TEST real (molecule, composition, T) query points using actual BIMOG-style
#    organics (reuse the two alpha-pinene SOA proxies + a few more varied structures already
#    used elsewhere in this project, to keep the benchmark grounded in real compounds)
# ---------------------------------------------------------------------------
TEST_SMILES = [
    "CC(=O)C1CC(C1(C)C)CC(=O)O",        # pinonic acid
    "CC1(C(CC1C(=O)O)CC(=O)O)C",        # pinic acid
    "OCC(O)CO",                           # glycerol
    "CC(=O)O",                            # acetic acid
    "OCC(O)C(O)C(O)C(O)CO",              # sorbitol
    "c1ccccc1C(=O)O",                     # benzoic acid
]
WATER = Component(1, "Water", ((16, 1),))

organics = []
for smi in TEST_SMILES:
    r = smiles_to_components([smi], water_as_component1=True)
    organics.append(Component(2, smi, r.components[1].subgroups))

N_PER_MOL = 2000  # water-fraction/temperature grid points per molecule -> matches a realistic large screening sweep / GSA sample count
wtf_grid = np.linspace(0.05, 0.95, N_PER_MOL)
T_grid = np.linspace(280.0, 310.0, N_PER_MOL)

N_TOTAL = len(organics) * N_PER_MOL
print(f"benchmarking {len(organics)} molecules x {N_PER_MOL} composition/T points = {N_TOTAL} total evaluations")

# ---------------------------------------------------------------------------
# 3) Time direct AIOMFAC calls (one ActivityModel.evaluate() per query point --
#    exactly how a screening sweep or Sobol sampler would call it point-by-point;
#    AIOMFAC has no batched/vectorized multi-composition API, so point-by-point
#    is the only way it is ever actually used)
# ---------------------------------------------------------------------------
# warm-up (exclude first-call Python import/JIT/cache effects from the timed loop)
for organic in organics[:1]:
    model_a = ActivityModel([WATER, organic])
    for _ in range(5):
        model_a.evaluate([0.5, 0.5], 298.15, basis="mass")

aiomfac_times = []
for organic in organics:
    model_a = ActivityModel([WATER, organic])
    for wtf_w, T_K in zip(wtf_grid, T_grid):
        t0 = time.perf_counter()
        _ = model_a.evaluate([wtf_w, 1.0 - wtf_w], T_K, basis="mass")
        aiomfac_times.append(time.perf_counter() - t0)
aiomfac_times = np.array(aiomfac_times)

# ---------------------------------------------------------------------------
# 4) Time the surrogate: (a) one-at-a-time (apples-to-apples per-call latency),
#    and (b) fully batched (the realistic way a surrogate is actually used in a sweep/GSA)
# ---------------------------------------------------------------------------


def build_feature_row(organic_subgroups, wtf_w, T_K):
    x = np.zeros(n_features, dtype=np.float32)
    total_q = 0.0
    for mg, count in organic_subgroups:
        if mg in main_groups:
            x[main_groups.index(mg)] += count
            total_q += count
    s = x[:len(main_groups)].sum()
    if s > 0:
        x[:len(main_groups)] /= s
    o_count = sum(c for mg, c in organic_subgroups if mg in (66, 67, 68, 69, 72))  # rough O-bearing groups proxy
    # NOTE: o_to_c / n_to_c here are approximate screening-level placeholders for timing purposes only
    # (the actual Part I notebook computes these precisely from RDKit); timing is insensitive to this.
    x[len(main_groups)] = 0.3       # o_to_c placeholder
    x[len(main_groups) + 1] = 0.0   # n_to_c placeholder
    x[len(main_groups) + 2] = np.log(150.0)   # log_molar_mass placeholder
    x[len(main_groups) + 3] = np.log1p(total_q)
    x[len(main_groups) + 4] = wtf_w
    x[len(main_groups) + 5] = (T_K - 293.15) / 20.0
    return x


rows = []
for organic in organics:
    for wtf_w, T_K in zip(wtf_grid, T_grid):
        rows.append(build_feature_row(organic.subgroups, wtf_w, T_K))
X = np.stack(rows).astype(np.float32)
Xn = (X - x_mean) / x_std
Xt = torch.tensor(Xn, dtype=torch.float32)

# warm-up the surrogate too (first torch call pays a one-off dispatch/cache cost)
with torch.no_grad():
    for _ in range(20):
        model(Xt[0:1])

# (a) one-at-a-time surrogate calls
surrogate_times_single = []
with torch.no_grad():
    for i in range(Xt.shape[0]):
        t0 = time.perf_counter()
        _ = model(Xt[i:i + 1])
        surrogate_times_single.append(time.perf_counter() - t0)
surrogate_times_single = np.array(surrogate_times_single)

# (b) fully batched surrogate call (all N_TOTAL points in one forward pass), repeated
# a few times and take the median total time (single-shot batch timing is noisy)
batch_run_times = []
with torch.no_grad():
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model(Xt)
        batch_run_times.append(time.perf_counter() - t0)
batched_total_time = float(np.median(batch_run_times))
batched_per_point = batched_total_time / N_TOTAL

# (c) batched-call scaling: how per-point cost drops as batch size grows
# (the realistic regime for a Sobol GSA -- tens of thousands of samples per run)
scaling = {}
for n in [1, 10, 100, 1000, N_TOTAL]:
    n = min(n, N_TOTAL)
    sub = Xt[:n]
    with torch.no_grad():
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = model(sub)
            times.append(time.perf_counter() - t0)
    scaling[n] = float(np.median(times)) / n

# ---------------------------------------------------------------------------
# 4b) Does a framework-free forward pass close the single-call gap? Re-implement the
#     exact same trained weights as plain NumPy matmuls (no PyTorch dispatch overhead)
#     and verify numerical equivalence before trusting its timing.
# ---------------------------------------------------------------------------
W_np = [state[f"net.{i}.weight"].numpy().astype(np.float32) for i in (0, 2, 4, 6)]
B_np = [state[f"net.{i}.bias"].numpy().astype(np.float32) for i in (0, 2, 4, 6)]


def numpy_forward(x):
    h = x
    for i in range(3):
        h = np.maximum(h @ W_np[i].T + B_np[i], 0.0)
    return h @ W_np[3].T + B_np[3]


with torch.no_grad():
    torch_check = model(Xt[:5]).numpy()
numpy_check = numpy_forward(Xn[:5])
max_abs_diff = float(np.abs(torch_check - numpy_check).max())
print(f"numpy-vs-torch numerical check, max abs diff over 5 rows: {max_abs_diff:.2e}")
assert max_abs_diff < 1e-4, "numpy reimplementation does not match the trained torch model"

for _ in range(20):
    numpy_forward(Xn[0])  # warm-up (numpy has no JIT, but keeps this loop symmetric with the torch one)

np_single_times = []
for i in range(Xn.shape[0]):
    t0 = time.perf_counter()
    numpy_forward(Xn[i])
    np_single_times.append(time.perf_counter() - t0)
np_single_times = np.array(np_single_times)

np_batch_times = []
for _ in range(10):
    t0 = time.perf_counter()
    numpy_forward(Xn)
    np_batch_times.append(time.perf_counter() - t0)
np_batched_per_point = float(np.median(np_batch_times)) / N_TOTAL

# ---------------------------------------------------------------------------
# 5) Report + save
# ---------------------------------------------------------------------------
results = {
    "n_total_evaluations": int(N_TOTAL),
    "aiomfac_direct": {
        "mean_s": float(aiomfac_times.mean()), "median_s": float(np.median(aiomfac_times)),
        "std_s": float(aiomfac_times.std()), "total_s": float(aiomfac_times.sum()),
    },
    "surrogate_single_call": {
        "mean_s": float(surrogate_times_single.mean()), "median_s": float(np.median(surrogate_times_single)),
        "std_s": float(surrogate_times_single.std()), "total_s": float(surrogate_times_single.sum()),
    },
    "surrogate_batched": {
        "per_point_s": float(batched_per_point), "total_s": float(batched_total_time),
    },
    "surrogate_batch_scaling_per_point_s": scaling,
    "surrogate_numpy_single_call": {
        "mean_s": float(np_single_times.mean()), "median_s": float(np.median(np_single_times)),
        "std_s": float(np_single_times.std()),
    },
    "surrogate_numpy_batched": {"per_point_s": np_batched_per_point},
    "numpy_vs_torch_max_abs_diff": max_abs_diff,
}
results["speedup_single_vs_aiomfac_mean"] = results["aiomfac_direct"]["mean_s"] / results["surrogate_single_call"]["mean_s"]
results["speedup_single_vs_aiomfac_median"] = results["aiomfac_direct"]["median_s"] / results["surrogate_single_call"]["median_s"]
results["speedup_batched_vs_aiomfac"] = results["aiomfac_direct"]["median_s"] / results["surrogate_batched"]["per_point_s"]
results["speedup_numpy_single_vs_aiomfac"] = results["aiomfac_direct"]["median_s"] / results["surrogate_numpy_single_call"]["median_s"]
results["speedup_numpy_batched_vs_aiomfac"] = results["aiomfac_direct"]["median_s"] / results["surrogate_numpy_batched"]["per_point_s"]

print(json.dumps(results, indent=2))
with open(f"{OUT_DIR}/timing_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {OUT_DIR}/timing_results.json")
