"""Predict ln(gamma_water), ln(gamma_organic) for binary water-organic mixtures with the R_emb release ensemble.

Usage:
    python predict_remb.py "CC(=O)C1CC(CC(=O)O)C1(C)C" 0.5 298.15

Requires aiomfac_py (with the s2as extra) and PyTorch. Valid within the training ranges stored in
remb_bimog_mcm_meta.json (w_water 0.02-0.98, T 273.15-313.15 K) and for molecules whose AIOMFAC subgroups are all
in the stored vocabulary; molecules with atoms S2AS cannot match are flagged.
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from aiomfac_py.s2as import smiles_to_components

HERE = pathlib.Path(__file__).resolve().parent
META = json.loads((HERE / "remb_bimog_mcm_meta.json").read_text())
VOCAB = META["subgroup_vocabulary"]


class EmbedNet(nn.Module):
    def __init__(self, n_vocab, n_extra=2, emb=16):
        super().__init__()
        self.n_vocab = n_vocab
        self.emb = nn.Linear(n_vocab, emb, bias=False)
        self.head = nn.Sequential(nn.Linear(emb + n_extra, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(torch.cat([self.emb(x[:, :self.n_vocab]), x[:, self.n_vocab:]], 1))


def load_ensemble():
    members = []
    for m in META["members"]:
        net = EmbedNet(len(VOCAB))
        net.load_state_dict(torch.load(HERE / m["file"]))
        net.eval()
        members.append((net, np.array(m["extra_mean"]), np.array(m["extra_std"]),
                        np.array(m["y_mean"]), np.array(m["y_std"])))
    return members


def subgroup_counts(smiles):
    r = smiles_to_components([smiles], water_as_component1=True)
    if r.removed:
        raise ValueError(f"S2AS could not decompose {smiles!r}: {r.removed[0][2]}")
    counts = np.zeros(len(VOCAB), np.float32)
    for s, q in r.components[1].subgroups:
        if int(s) not in VOCAB:
            raise ValueError(f"subgroup {s} of {smiles!r} is outside the training vocabulary")
        counts[VOCAB.index(int(s))] = q
    return counts, int(r.n_unmatched_atoms[0])


def predict(smiles, w_water, T_K, members=None):
    """Ensemble mean and spread of (ln gamma_water, ln gamma_organic); w_water and T_K may be arrays."""
    members = members or load_ensemble()
    counts, n_unmatched = subgroup_counts(smiles)
    w = np.atleast_1d(np.asarray(w_water, float)); T = np.broadcast_to(np.asarray(T_K, float), w.shape)
    extra = np.column_stack([w, (T - 293.15) / 20.0])
    preds = []
    for net, em, es, ym, ys in members:
        x = np.concatenate([np.tile(counts, (len(w), 1)), (extra - em) / es], 1).astype(np.float32)
        with torch.no_grad():
            preds.append(net(torch.tensor(x)).numpy() * ys + ym)
    preds = np.array(preds)
    return preds.mean(0), preds.std(0), n_unmatched


if __name__ == "__main__":
    smi, w, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    mean, sd, n_unm = predict(smi, w, T)
    print(f"ln gamma_water   = {mean[0, 0]:+.4f} (ensemble sd {sd[0, 0]:.4f})")
    print(f"ln gamma_organic = {mean[0, 1]:+.4f} (ensemble sd {sd[0, 1]:.4f})")
    if n_unm:
        print(f"warning: {n_unm} atom(s) not matched by S2AS; AIOMFAC (and hence the surrogate) ignores them")
