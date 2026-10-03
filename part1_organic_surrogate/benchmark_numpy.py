"""Does a pure-NumPy forward pass (no PyTorch dispatch overhead) close the gap between
single-point surrogate calls and AIOMFAC? Reuses the exact trained weights."""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/tmp/claude-0/-home-claude/eecda288-448a-508f-8b60-1c1c2ce4350c/scratchpad/nb_exec_test3/aiomfac-python/src")
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components

GSA_DIR = "/tmp/claude-0/-home-claude/eecda288-448a-508f-8b60-1c1c2ce4350c/scratchpad/aiomfac_gsa"

with open(f"{GSA_DIR}/surrogate_meta.json") as f:
    meta = json.load(f)
norm = np.load(f"{GSA_DIR}/surrogate_norm.npz")
x_mean, x_std = norm["x_mean"].astype(np.float32), norm["x_std"].astype(np.float32)
y_mean, y_std = norm["y_mean"].astype(np.float32), norm["y_std"].astype(np.float32)
main_groups = meta["main_groups"]

state = torch.load(f"{GSA_DIR}/surrogate_weights.pt", map_location="cpu")
W = [state[f"net.{i}.weight"].numpy().astype(np.float32) for i in (0, 2, 4, 6)]
B = [state[f"net.{i}.bias"].numpy().astype(np.float32) for i in (0, 2, 4, 6)]


def numpy_forward(x):
    # x: (n_features,) single row -- plain numpy matmuls + manual ReLU, no framework overhead
    h = x
    for i in range(3):
        h = h @ W[i].T + B[i]
        h = np.maximum(h, 0.0)
    return h @ W[3].T + B[3]


def numpy_forward_batch(X):
    h = X
    for i in range(3):
        h = h @ W[i].T + B[i]
        h = np.maximum(h, 0.0)
    return h @ W[3].T + B[3]


TEST_SMILES = ["CC(=O)C1CC(C1(C)C)CC(=O)O", "CC1(C(CC1C(=O)O)CC(=O)O)C", "OCC(O)CO",
               "CC(=O)O", "OCC(O)C(O)C(O)C(O)CO", "c1ccccc1C(=O)O"]
WATER = Component(1, "Water", ((16, 1),))
organics = []
for smi in TEST_SMILES:
    r = smiles_to_components([smi], water_as_component1=True)
    organics.append(Component(2, smi, r.components[1].subgroups))

N_PER_MOL = 2000
wtf_grid = np.linspace(0.05, 0.95, N_PER_MOL)
T_grid = np.linspace(280.0, 310.0, N_PER_MOL)
n_features = len(main_groups) + 6


def build_row(subgroups, wtf_w, T_K):
    x = np.zeros(n_features, dtype=np.float32)
    total_q = 0.0
    for mg, count in subgroups:
        if mg in main_groups:
            x[main_groups.index(mg)] += count
            total_q += count
    s = x[:len(main_groups)].sum()
    if s > 0:
        x[:len(main_groups)] /= s
    x[len(main_groups):len(main_groups) + 4] = [0.3, 0.0, np.log(150.0), np.log1p(total_q)]
    x[len(main_groups) + 4] = wtf_w
    x[len(main_groups) + 5] = (T_K - 293.15) / 20.0
    return x


rows = [build_row(o.subgroups, w, t) for o in organics for w, t in zip(wtf_grid, T_grid)]
X = np.stack(rows).astype(np.float32)
Xn = (X - x_mean) / x_std

# warm-up
for i in range(20):
    numpy_forward(Xn[0])

# AIOMFAC timing (same as before, for reference)
m0 = ActivityModel([WATER, organics[0]])
for _ in range(5):
    m0.evaluate([0.5, 0.5], 298.15, basis="mass")
aiomfac_times = []
for organic in organics:
    m = ActivityModel([WATER, organic])
    for wtf_w, T_K in zip(wtf_grid, T_grid):
        t0 = time.perf_counter()
        m.evaluate([wtf_w, 1.0 - wtf_w], T_K, basis="mass")
        aiomfac_times.append(time.perf_counter() - t0)
aiomfac_times = np.array(aiomfac_times)

# pure-numpy single-call timing
np_single_times = []
for i in range(Xn.shape[0]):
    t0 = time.perf_counter()
    numpy_forward(Xn[i])
    np_single_times.append(time.perf_counter() - t0)
np_single_times = np.array(np_single_times)

# pure-numpy batched timing (sanity check -- should be even faster than torch batched)
batch_times = []
for _ in range(10):
    t0 = time.perf_counter()
    numpy_forward_batch(Xn)
    batch_times.append(time.perf_counter() - t0)
np_batched_per_point = np.median(batch_times) / Xn.shape[0]

print(f"AIOMFAC direct:              median {np.median(aiomfac_times)*1e6:8.3f} us/eval")
print(f"Surrogate, numpy single-call: median {np.median(np_single_times)*1e6:8.3f} us/eval  "
      f"({np.median(aiomfac_times)/np.median(np_single_times):.1f}x vs AIOMFAC)")
print(f"Surrogate, numpy batched:            {np_batched_per_point*1e6:8.3f} us/eval  "
      f"({np.median(aiomfac_times)/np_batched_per_point:.1f}x vs AIOMFAC)")

# verify correctness against the torch model (sanity: outputs should match closely)
import torch.nn as nn
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

tm = SurrogateMLP(n_features)
tm.load_state_dict(state)
tm.eval()
with torch.no_grad():
    torch_out = tm(torch.tensor(Xn[:5])).numpy()
numpy_out = numpy_forward_batch(Xn[:5])
print("\nmax abs diff (numpy vs torch, first 5 rows):", np.abs(torch_out - numpy_out).max())
