"""Electrolyte surrogates inside aiomfac_py's PhaseEquilibrium, compared with AIOMFAC for inorganic particles
(298.15 K, RH 0.95-0.30): phase state (liquid, solids) and water content.

  ge     the mole-fraction excess-Gibbs-energy surrogate (GEXNet release ensemble, mean of G over 5 members)
  direct the unconstrained free-species network (5-member mean of its ln gamma); its "chemical potentials" are not
         derivatives of a Gibbs energy, so the solver works with the potentials and a finite-difference Jacobian

Writes ../results/s2v2_pe_gex.json."""
import json
import sys
import time

import numpy as np
import torch

from aiomfac_py.lr import debye_huckel_parameters
from aiomfac_py.phase_equilibrium import ExplicitLiquidModel, PhaseEquilibrium

import s2_train as S2

torch.set_default_dtype(torch.float64)
ALIAS = {"Ca++": "Ca2+"}


def load(kind, release):
    out = []
    import os
    for seed in [int(v) for v in os.environ.get("S2_REL_SEEDS", "0,1,2,3,4").split(",")]:
        n = {"gex": S2.GEXNet, "direct": S2.DirectNet}[kind]()
        n.load_state_dict(torch.load(f"../results/s2v2_models/{kind}{'_release' if release else ''}_seed{seed}.pt"))
        out.append(n.double().eval())
    return out


class SurrogateElectrolyteLiquid(ExplicitLiquidModel):
    def __init__(self, ions, nets, kind):
        super().__init__([], ions)
        self.idx = torch.tensor([S2.NAMES.index(ALIAS.get(i, i)) for i in self.sp_ions])
        self.nets, self.kind = nets, kind
        self._ref = {}

    def _consts(self, T):
        A, b = debye_huckel_parameters(float(T))
        return torch.full((1,), (float(T) - 293.15) / 20), torch.tensor([A]), torch.tensor([b])

    def _ln_a_t(self, n, T):
        tn, A, b = self._consts(T)
        nw = n[0]
        ni = torch.zeros(S2.NI, dtype=n.dtype).index_put((self.idx,), n[1:1 + len(self.idx)])
        lnxw = -torch.log1p(S2.MW * ni.sum() / (nw * S2.MW))
        m = ni / (nw * S2.MW)
        if self.kind == "gex":
            G = sum(net.G(nw.reshape(1), ni[None, :], tn, A, b)[0] for net in self.nets) / len(self.nets)
            key = round(float(T), 6)
            if key not in self._ref:
                z = torch.zeros(1, S2.NI, requires_grad=True)
                with torch.enable_grad():
                    Gz = sum(net.G(torch.full((1,), 1.0 / S2.MW), z, tn, A, b)[0] for net in self.nets) / len(self.nets)
                    self._ref[key] = torch.autograd.grad(Gz, z)[0][0].detach()
            gn = torch.autograd.grad(G, n, create_graph=True)[0]
            lnaw = lnxw + gn[0]
            ions = lnxw + gn[1:1 + len(self.idx)] - self._ref[key][self.idx] + torch.log(m[self.idx])
        else:
            lnaw = lnxw * 0
            aw = gi = 0
            for net in self.nets:
                a_, g_ = net(m[None, :], tn, A, b)
                aw, gi = aw + a_[0], gi + g_[0]
            lnaw = aw / len(self.nets)
            ions = gi[self.idx] / len(self.nets) + torch.log(m[self.idx])
        return torch.cat([lnaw.reshape(1), ions])

    def ln_a(self, n, T):
        self.n_eval += 1
        t = torch.tensor(np.asarray(n, dtype=float), requires_grad=True)
        with torch.enable_grad():
            v = self._ln_a_t(t, T)
        return v.detach().numpy() + self._c

    def hessian_ad(self, n, T, active=None):
        self.n_jac += 1
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(lambda t: self._ln_a_t(t, T), torch.tensor(np.asarray(n, float)),
                                                   create_graph=False).numpy()
        act = np.ones(self.N, bool) if active is None else np.asarray(active, bool)
        H = np.zeros((self.N, self.N)); i = np.flatnonzero(act); H[np.ix_(i, i)] = J[np.ix_(i, i)]
        return 0.5 * (H + H.T)


class SurrogatePE(PhaseEquilibrium):
    def __init__(self, ions, T, nets, kind, **kw):
        super().__init__([], ions, T, **kw)
        self.lm = SurrogateElectrolyteLiquid(ions, nets, kind)
        self.hess_scheme = "ad"

    def _use_ad(self):
        return True


CASES = {"(NH4)2SO4": (["NH4+", "SO4--"], {"NH4+": 2.0, "SO4--": 1.0}),
         "NaCl": (["Na+", "Cl-"], {"Na+": 1.0, "Cl-": 1.0}),
         "NH4NO3": (["NH4+", "NO3-"], {"NH4+": 1.0, "NO3-": 1.0}),
         "NH4HSO4": (["NH4+", "H+", "SO4--"], {"NH4+": 1.0, "H+": 1.0, "SO4--": 1.0}),
         "NaCl + (NH4)2SO4": (["Na+", "NH4+", "Cl-", "SO4--"], {"Na+": 1.0, "Cl-": 1.0, "NH4+": 2.0, "SO4--": 1.0})}
RH = [0.95, 0.9, 0.85, 0.82, 0.8, 0.78, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3]
T = 298.15

if __name__ == "__main__":
    kinds = sys.argv[1:] or ["gex", "direct"]
    nets = {k: load(k, release=(k == "gex")) for k in kinds}
    OUT = {}
    only = __import__("os").environ.get("PE_CASES")
    for name, (ions, feed) in CASES.items():
        if only and name not in only.split(";"):
            continue
        rows = {}
        makers = [("aiomfac", lambda: PhaseEquilibrium([], ions, T))] + \
                 [(k, lambda k=k: SurrogatePE(ions, T, nets[k], k)) for k in kinds]
        for tag, mk in makers:
            r_ = []
            for rh in RH:
                t0 = time.time()
                try:
                    r = mk().solve(feed, rh)
                    w = float(sum(L.amounts[0] for L in r.liquids))
                    r_.append({"rh": rh, "status": r.status, "n_liquids": r.n_liquids,
                               "solids": {k: round(v, 6) for k, v in r.solids.items()}, "water_mol": w,
                               "time_s": time.time() - t0})
                except Exception as e:                                   # noqa: BLE001
                    r_.append({"rh": rh, "status": f"error: {e}"[:80], "n_liquids": -1, "solids": {}, "water_mol": float("nan"),
                               "time_s": time.time() - t0})
            rows[tag] = r_
        OUT[name] = rows
        print(name, flush=True)
        for i, a in enumerate(rows["aiomfac"]):
            line = f"  RH {a['rh']:.2f}: AIOMFAC {a['status'][:4]} L{a['n_liquids']} {sorted(a['solids'])} w={a['water_mol']:.4g}"
            for k in kinds:
                b = rows[k][i]
                same = set(a["solids"]) == set(b["solids"]) and a["n_liquids"] == b["n_liquids"]
                line += f" | {k} {b['status'][:4]} L{b['n_liquids']} {sorted(b['solids'])} w={b['water_mol']:.4g}{'' if same else ' <--'}"
            print(line, flush=True)
        json.dump(OUT, open(__import__("os").environ.get("PE_OUT", "../results/s2v2_pe_gex.json"), "w"), indent=1)
