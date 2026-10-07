"""PhaseEquilibrium with the AIOMFAC activities of the liquid phases replaced by a neural-network surrogate.

Binary water-organic liquids (Part 1 surrogate).  Two ways of using the surrogate:

``mode="raw"``         ln a_i = ln x_i + ln gamma_i(surrogate): the surrogate's own activity coefficients.  They need
                       not satisfy Gibbs-Duhem, so the potentials are not exactly the gradient of one Gibbs function.
``mode="consistent"``  G^E/RT = sum_i n_i ln gamma_i(surrogate) is taken as the excess Gibbs energy and the activity
                       coefficients are its derivatives, ln gamma_i = dG^E/dn_i (torch autograd): Gibbs-Duhem holds
                       exactly, at the price of activity coefficients that differ from the surrogate's own.  The
                       derivatives of the network enter, and they were never trained (ReLU network: piecewise
                       constant derivatives), so this variant can be far off.
``mode="gd_org"``      ln gamma_org from the surrogate; ln gamma_w from the Gibbs-Duhem equation integrated by parts,
                       ln gamma_w(x) = -x/(1-x) ln gamma_o(x) + int_0^x ln gamma_o(t)/(1-t)^2 dt  (x = x_org),
                       with pure water as the reference (ln gamma_w(0) = 0).  Consistent, and only values of the
                       network are used (Gauss-Legendre quadrature, ``n_quad`` nodes).

Everything else (Newton iterations, stability test, warm starts) is the unchanged aiomfac_py solver; the Hessian
d ln a / d n is the exact torch Jacobian (``hess_scheme = "ad"``).  Starting points of the solver (water guesses,
component patterns of the stability test) still come from AIOMFAC; they do not enter the equilibrium conditions.
"""
from __future__ import annotations

import numpy as np
import torch

from aiomfac_py.phase_equilibrium import ExplicitLiquidModel, PhaseEquilibrium

from surrogates import MW_WATER, Part1Ensemble


class SurrogateBinaryLiquidModel(ExplicitLiquidModel):
    def __init__(self, organic, ensemble: Part1Ensemble, mode: str = "raw", member: int | None = None,
                 clamp: bool = False):
        """``clamp``: evaluate the network at the water mass fraction clipped to its training range (constant
        extrapolation; towards infinite dilution ln gamma_org approaches a constant, which a ReLU network would
        otherwise continue linearly)."""
        super().__init__([organic], [])
        self.clamp = clamp
        if mode not in ("raw", "consistent", "gd_org"):
            raise ValueError("mode must be 'raw', 'consistent' or 'gd_org'")
        u, wq = np.polynomial.legendre.leggauss(64)
        self._u, self._wq = torch.tensor(u), torch.tensor(wq)
        self.ens, self.mode, self.member = ensemble, mode, member
        self.cnt = ensemble.counts(organic.subgroups)
        self.M = torch.tensor(np.asarray(self._mm, dtype=float))        # kg/mol (water, organic)
        self.n_ood = 0                                                    # evaluations outside the w_water range
        self.w_seen = [1.0, 0.0]

    # ------------------------------------------------------------------------------------------------
    def _ln_gamma_t(self, n: torch.Tensor, T: float) -> torch.Tensor:
        m = n * self.M
        w = self._clip((m[0] / m.sum()).reshape(1))
        lgw, lgo = self.ens.ln_gamma(self.cnt, w, T, self.member)
        return torch.cat([lgw, lgo])

    def _clip(self, w: torch.Tensor) -> torch.Tensor:
        if not self.clamp:
            return w
        lo, hi = self.ens.w_range
        return torch.clamp(w, lo, hi)

    def _w_of_x(self, xo: torch.Tensor) -> torch.Tensor:
        mw, mo = (1 - xo) * self.M[0], xo * self.M[1]
        return self._clip(mw / (mw + mo))

    def _ln_gamma_gd(self, n: torch.Tensor, T: float) -> torch.Tensor:
        xo = n[1] / n.sum()
        t = 0.5 * xo * (self._u + 1.0)                      # Gauss-Legendre nodes on (0, x)
        wt = 0.5 * xo * self._wq
        pts = torch.cat([xo.reshape(1), t])
        _, lgo = self.ens.ln_gamma(self.cnt, self._w_of_x(pts), T, self.member)
        lgo_x, lgo_t = lgo[0], lgo[1:]
        lgw = -xo / (1 - xo) * lgo_x + torch.sum(wt * lgo_t / (1 - t) ** 2)
        return torch.stack([lgw, lgo_x])

    def _ln_a_t(self, n: torch.Tensor, T: float) -> torch.Tensor:
        x = n / n.sum()
        if self.mode == "gd_org":
            lg = self._ln_gamma_gd(n, T)
        elif self.mode == "raw":
            lg = self._ln_gamma_t(n, T)
        else:
            nn_ = n.detach().clone().requires_grad_(True) if not n.requires_grad else n
            GE = torch.dot(nn_, self._ln_gamma_t(nn_, T))
            lg = torch.autograd.grad(GE, nn_, create_graph=True)[0]
        return torch.log(x) + lg

    def _track(self, n):
        w = float(n[0] * self._mm[0] / np.dot(n, self._mm))
        self.w_seen = [min(self.w_seen[0], w), max(self.w_seen[1], w)]
        lo, hi = self.ens.w_range
        if not lo <= w <= hi:
            self.n_ood += 1

    def ln_a(self, n: np.ndarray, T: float) -> np.ndarray:
        self.n_eval += 1
        n = np.asarray(n, dtype=float)
        self._track(n)
        with torch.enable_grad():
            v = self._ln_a_t(torch.tensor(n), T)
        return v.detach().numpy() + self._c

    def ln_gamma(self, n: np.ndarray, T: float) -> np.ndarray:
        """Activity coefficients as used (raw or consistent)."""
        n = np.asarray(n, dtype=float)
        return self.ln_a(n, T) - self._c - np.log(n / n.sum())

    def hessian_ad(self, n: np.ndarray, T: float, active: np.ndarray | None = None) -> np.ndarray:
        n = np.asarray(n, dtype=float)
        self.n_jac += 1
        J = torch.autograd.functional.jacobian(lambda t: self._ln_a_t(t, T), torch.tensor(n)).numpy()
        act = np.ones(self.N, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        H = np.zeros((self.N, self.N))
        idx = np.flatnonzero(act)
        H[np.ix_(idx, idx)] = J[np.ix_(idx, idx)]
        return 0.5 * (H + H.T)


class SurrogatePhaseEquilibrium(PhaseEquilibrium):
    """PhaseEquilibrium of water + one organic with surrogate activities (see module docstring)."""

    def __init__(self, organic, T_K: float, ensemble: Part1Ensemble, *, mode: str = "raw",
                 member: int | None = None, clamp: bool = False, **kw):
        super().__init__([organic], [], T_K, **kw)
        self.lm = SurrogateBinaryLiquidModel(organic, ensemble, mode, member, clamp)
        self.hess_scheme = "ad"

    def _use_ad(self) -> bool:
        return self.hess_scheme == "ad"


# ====================================================================================================================
# S1: excess-Gibbs-energy surrogate (Gibbs-Duhem consistent by construction)
# ====================================================================================================================

class GEBinaryLiquidModel(ExplicitLiquidModel):
    """Water + one organic with ln gamma from an ensemble of S1 G^E networks (s1_train.Net, kind 'ge').

    The ensemble mean of g^E is used (a mean of consistent models is consistent); ln gamma and the Hessian follow by
    autograd in float64."""

    def __init__(self, organic, nets, counts: np.ndarray, M_kg: float):
        super().__init__([organic], [])
        self.nets = [n.double().eval() for n in nets]
        self.c = torch.tensor(counts, dtype=torch.float64).reshape(1, -1)
        self.Mo = float(M_kg)
        self.n_ood = 0
        self.w_range = (0.02, 0.98)

    def _gE(self, x: torch.Tensor, T: float) -> torch.Tensor:
        tn = torch.full_like(x, (float(T) - 293.15) / 20.0)
        g = 0.0
        for net in self.nets:
            e = net.emb(self.c.expand(len(x), -1))
            z = torch.stack([(x - net.em[0]) / net.es[0], (tn - net.em[1]) / net.es[1]], 1)
            g = g + x * (1 - x) * net.head(torch.cat([e, z], 1))[:, 0] * net.ys[1]
        return g / len(self.nets)

    def _ln_a_t(self, n: torch.Tensor, T: float) -> torch.Tensor:
        N = n.sum()
        x = (n[1] / N).reshape(1)
        GE = N * self._gE(x, T)[0]                        # G^E/RT of the liquid
        if not n.requires_grad:
            n = n.clone().requires_grad_(True)
            N = n.sum(); x = (n[1] / N).reshape(1); GE = N * self._gE(x, T)[0]
        lg = torch.autograd.grad(GE, n, create_graph=True)[0]
        return torch.log(n / n.sum()) + lg

    def ln_a(self, n: np.ndarray, T: float) -> np.ndarray:
        self.n_eval += 1
        n = np.asarray(n, dtype=float)
        w = n[0] * self._mm[0] / np.dot(n, self._mm)
        if not self.w_range[0] <= w <= self.w_range[1]:
            self.n_ood += 1
        with torch.enable_grad():
            v = self._ln_a_t(torch.tensor(n), T)
        return v.detach().numpy() + self._c

    def hessian_ad(self, n: np.ndarray, T: float, active: np.ndarray | None = None) -> np.ndarray:
        n = np.asarray(n, dtype=float)
        self.n_jac += 1
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(lambda t: self._ln_a_t(t, T), torch.tensor(n)).numpy()
        act = np.ones(self.N, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        H = np.zeros((self.N, self.N))
        idx = np.flatnonzero(act)
        H[np.ix_(idx, idx)] = J[np.ix_(idx, idx)]
        return 0.5 * (H + H.T)


class GEPhaseEquilibrium(PhaseEquilibrium):
    """PhaseEquilibrium of water + one organic with the S1 G^E surrogate."""

    def __init__(self, organic, T_K: float, nets, counts, M_kg, **kw):
        super().__init__([organic], [], T_K, **kw)
        self.lm = GEBinaryLiquidModel(organic, nets, counts, M_kg)
        self.hess_scheme = "ad"

    def _use_ad(self) -> bool:
        return self.hess_scheme == "ad"
