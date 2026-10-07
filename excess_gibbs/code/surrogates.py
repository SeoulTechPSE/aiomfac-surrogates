"""Neural-network surrogates of aiomfac-surrogates v2.0.1, loaded as differentiable float64 torch functions.

Part 1 (binary water-organic): the release R_emb ensemble (3 seeds) of ``part1_organic_surrogate/checkpoints``.
Inputs: subgroup counts (vocabulary of the checkpoint), water mass fraction w, T_norm = (T - 293.15)/20 (both
standardized per member); outputs ln gamma_water, ln gamma_organic on AIOMFAC's mole-fraction scale.

The networks were trained in float32; here they are evaluated in float64 (weights cast) so that the phase-equilibrium
Newton iterations see a smooth, deterministic function instead of float32 round-off (about 1e-7 in ln gamma).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

SURR_ROOT = Path(os.environ.get("AIOMFAC_SURROGATES", Path(__file__).resolve().parents[2]))
P1_DIR = SURR_ROOT / "part1_organic_surrogate" / "checkpoints"
MW_WATER = 18.01528e-3                     # kg/mol


class EmbedNet(nn.Module):
    """Architecture of the Part 1 release checkpoint (predict_remb.py)."""

    def __init__(self, n_vocab, n_extra=2, emb=16):
        super().__init__()
        self.n_vocab = n_vocab
        self.emb = nn.Linear(n_vocab, emb, bias=False)
        self.head = nn.Sequential(nn.Linear(emb + n_extra, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(torch.cat([self.emb(x[:, :self.n_vocab]), x[:, self.n_vocab:]], 1))


class Part1Ensemble:
    """ln gamma (water, organic) of a binary water-organic mixture; ensemble mean or single members."""

    def __init__(self, ckpt_dir: Path = P1_DIR):
        meta = json.loads((ckpt_dir / "remb_bimog_mcm_meta.json").read_text())
        self.meta = meta
        self.vocab = [int(v) for v in meta["subgroup_vocabulary"]]
        self.w_range = tuple(meta["training_ranges"]["w_water"])
        self.T_range = tuple(meta["training_ranges"]["T_K"])
        self.members = []
        for m in meta["members"]:
            net = EmbedNet(len(self.vocab))
            net.load_state_dict(torch.load(ckpt_dir / m["file"], map_location="cpu"))
            net = net.double().eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self.members.append((net, torch.tensor(m["extra_mean"]), torch.tensor(m["extra_std"]),
                                 torch.tensor(m["y_mean"]), torch.tensor(m["y_std"])))

    def counts(self, subgroups) -> torch.Tensor:
        c = torch.zeros(len(self.vocab))
        for s, q in subgroups:
            if int(s) not in self.vocab:
                raise ValueError(f"subgroup {s} is outside the Part 1 vocabulary")
            c[self.vocab.index(int(s))] = float(q)
        return c

    def ln_gamma(self, counts: torch.Tensor, w_water: torch.Tensor, T_K: float, member: int | None = None):
        """(ln gamma_w, ln gamma_org) for water mass fractions ``w_water`` (tensor, any shape; differentiable)."""
        w = w_water.reshape(-1)
        tn = torch.full_like(w, (float(T_K) - 293.15) / 20.0)
        extra = torch.stack([w, tn], 1)
        sel = self.members if member is None else [self.members[member]]
        out = 0.0
        for net, em, es, ym, ys in sel:
            x = torch.cat([counts.expand(len(w), -1), (extra - em) / es], 1)
            out = out + net(x) * ys + ym
        out = out / len(sel)
        return out[:, 0].reshape(w_water.shape), out[:, 1].reshape(w_water.shape)
