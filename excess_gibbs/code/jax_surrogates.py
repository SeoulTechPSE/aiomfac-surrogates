"""The excess-Gibbs-energy surrogates of Parts 1 and 2 as JAX Gibbs functions for aiomfac_py's phase-equilibrium solver.

The torch networks (``s1_train.Net`` kind "ge", ``s2_train.GEXNet``) are transcribed to JAX with the trained weights
(float64), wrapped in :class:`aiomfac_py.gibbs_model.GibbsFunction`, and used through
:class:`aiomfac_py.gibbs_model.GibbsLiquidModel`.  Compared with the torch classes of ``surrogate_pe.py`` and
``s2_pe_gex.py``, which differentiate the network with torch autograd at every activity evaluation and build the
Hessian by repeated reverse passes:

  1. the gradient (ln a), the Hessian and Hessian-vector products are jit-compiled once and reused for every call,
     every phase, every relative humidity and every temperature (temperature enters as an argument);
  2. the Hessian is forward-over-reverse (jacfwd of grad: N Hessian-vector products in one compiled call), and a
     single Hessian-vector product is available for matrix-free use;
  3. the networks are unchanged (same weights), so the results are those of the torch implementation to round-off
     (checked by ``jax_check.py``).

Requires aiomfac_py with ``aiomfac_py.gibbs_model`` (after v1.1.0) and jax.
"""
from __future__ import annotations

import math

import numpy as np

from aiomfac_py.gibbs_model import GibbsFunction, GibbsLiquidModel, ideal_mixing
from aiomfac_py.lr import debye_huckel_parameters
from aiomfac_py.phase_equilibrium import ExplicitLiquidModel

import jax
import jax.numpy as jnp

MW = 0.01801528
# Part 2 species basis (s2_train.NAMES) with aiomfac_py's ion keys
P2_NAMES = ["Na+", "K+", "NH4+", "Ca2+", "H+", "Cl-", "NO3-", "SO4--", "HSO4-", "HCO3-", "CO3--", "IO3-", "I-", "OH-"]
P2_Z = np.array([1, 1, 1, 2, 1, -1, -1, -2, -1, -1, -2, -1, -1, -1], dtype=float)
AIOMFAC_KEY = {"Ca2+": "Ca++"}


def _linear_layers(seq):
    """[(W (in, out), b (out,)), ...] of the nn.Linear layers of a torch Sequential, as float64 NumPy arrays."""
    import torch.nn as nn
    return [(m.weight.detach().double().numpy().T.copy(), m.bias.detach().double().numpy().copy())
            for m in seq if isinstance(m, nn.Linear)]


def _stack(layer_lists):
    """Stack the layers of an ensemble: [(W (M, in, out), b (M, out)), ...]."""
    return [(jnp.asarray(np.stack([L[i][0] for L in layer_lists])), jnp.asarray(np.stack([L[i][1] for L in layer_lists])))
            for i in range(len(layer_lists[0]))]


def _mlp_silu(layers, h):
    """Ensemble MLP with SiLU hidden activations; h: (M, in) -> (M, out)."""
    for i, (W, b) in enumerate(layers):
        h = jnp.einsum("mi,mio->mo", h, W) + b
        if i < len(layers) - 1:
            h = h * jax.nn.sigmoid(h)
    return h


# ====================================================================================================================
# Part 1: binary water + organic, g^E/RT = x_w x_o h(emb, x_o, T)   (s1_train.Net, kind "ge")
# ====================================================================================================================

_P1_ENSEMBLES: dict = {}


def _part1_ensemble(nets):
    """Ensemble-level Gibbs function of Part 1 (one per set of networks), with a compiled-function cache shared by all
    molecules: the molecule enters only through its subgroup embedding, passed as a constant."""
    key = tuple(id(n) for n in nets)
    if key not in _P1_ENSEMBLES:
        em = jnp.asarray(np.stack([n.em.detach().double().numpy() for n in nets]))                 # (M, 2)
        es = jnp.asarray(np.stack([n.es.detach().double().numpy() for n in nets]))
        ys1 = jnp.asarray(np.array([float(n.ys[1]) for n in nets]))
        layers = _stack([_linear_layers(n.head) for n in nets])

        def g(n, k):
            N = jnp.sum(n)
            x = n[1] / N
            z = jnp.stack([(x - em[:, 0]) / es[:, 0], (k["tn"] - em[:, 1]) / es[:, 1]], 1)        # (M, 2)
            h = _mlp_silu(layers, jnp.concatenate([k["emb"], z], 1))[:, 0] * ys1
            return ideal_mixing(n) + N * x * (1.0 - x) * jnp.mean(h)

        _P1_ENSEMBLES[key] = (g, {}, nets)            # keep the networks alive: the key uses their ids
    return _P1_ENSEMBLES[key]


def part1_gibbs(nets, counts: np.ndarray, organic_name: str) -> GibbsFunction:
    """Gibbs function of water + one organic from an ensemble of Part 1 G^E networks (mean of g^E over the members).
    ``counts``: subgroup counts of the organic in the vocabulary of the networks.  All molecules evaluated with the
    same networks share one compiled function."""
    g, cache, _ = _part1_ensemble(nets)
    c = np.asarray(counts, dtype=float)
    emb = np.stack([n.emb.weight.detach().double().numpy() @ c for n in nets])                      # (M, 16)
    return GibbsFunction(["Water", organic_name], g, consts=lambda T: {"tn": (T - 293.15) / 20.0, "emb": emb},
                         label=f"part1_ge[{organic_name}]", compiled_cache=cache)


def _f_dh(u):
    """(1+u)^2/2 - 2(1+u) + ln(1+u) + 3/2 = u^3/3 - u^4/4 + ... (the Debye-Hueckel integral of GEXNet), evaluated by its
    series for u < 0.01.  The closed form leaves u^3/3 after cancelling terms of order 1; at the infinite-dilution
    reference (I = 0, u = b sqrt(1e-30)) its round-off, multiplied by du/dI = b/(2 sqrt(I)) ~ 1e15, makes the
    reference potentials ref_i depend on the evaluation order (0.03-0.1 in the jit-compiled form; the torch float64
    evaluation happens to round to the exact limit, 0)."""
    small = u < 1.0e-2
    us = jnp.where(small, u, 0.0)
    series = sum(((-1) ** (k + 1)) * us ** k / k for k in range(3, 11))
    ul = jnp.where(small, 1.0, u)
    direct = (1 + ul) ** 2 / 2 - 2 * (1 + ul) + jnp.log1p(ul) + 1.5
    return jnp.where(small, series, direct)


# ====================================================================================================================
# Part 2: water + 14 free ions, G/RT = n_w M_w g_DH + n_T x^T H(x, m, T) x   (s2_train.GEXNet)
# ====================================================================================================================

def part2_gibbs(nets) -> GibbsFunction:
    """Gibbs function of water + the 14 free ions of the Part 2 basis from an ensemble of GEXNet networks (mean of G).

        g = sum_s n_s ln x_s - ln(M_w) sum_i n_i + G(n_w, n_i) - sum_i n_i ref_i(T),

    whose gradient is ln a_w = ln x_w + dG/dn_w and ln a_i = ln m_i + ln x_w + dG/dn_i - ref_i (molal ions, with
    ref_i = dG/dn_i at infinite dilution in water), as in ``s2_pe_gex.SurrogateElectrolyteLiquid``."""
    ni_ = len(P2_NAMES)
    layers = _stack([_linear_layers(n.f) for n in nets])
    iu = np.triu_indices(ni_ + 1)
    iu = (iu[0][1:], iu[1][1:])                                   # drop the (water, water) entry
    z2 = jnp.asarray(P2_Z ** 2)

    def G(nw, ni, k):
        m = ni / (nw * MW)
        I = 0.5 * jnp.sum(m * z2)
        s = jnp.sqrt(I + 1e-30)
        gdh = -(4 * k["A"] / k["b"] ** 3) * _f_dh(k["b"] * s)
        ntot = nw + jnp.sum(ni)
        x = jnp.concatenate([nw[None], ni]) / ntot
        inp = jnp.concatenate([jnp.log1p(m), x[:1], k["tn"][None]])
        hv = _mlp_silu(layers, jnp.broadcast_to(inp, (layers[0][0].shape[0], inp.shape[0])))        # (M, n_upper)
        # x^T H x with H = (U + U^T)/2, U = hv on the upper triangle: = sum_{i<=j} hv_ij x_i x_j (no matrix, no scatter)
        return nw * MW * gdh + ntot * jnp.mean(hv @ (x[iu[0]] * x[iu[1]]))

    dG_dni = jax.jit(jax.grad(G, argnums=1))

    def consts(T):
        A, b = debye_huckel_parameters(float(T))
        k = {"tn": jnp.asarray((T - 293.15) / 20.0), "A": jnp.asarray(A), "b": jnp.asarray(b)}
        k["ref"] = dG_dni(jnp.asarray(1.0 / MW), jnp.zeros(ni_), k)
        return k

    ions = np.ones(ni_)

    def g(n, k):
        nw, ni = n[0], n[1:]
        return ideal_mixing(n) - math.log(MW) * jnp.sum(ni * ions) + G(nw, ni, k) - jnp.dot(ni, k["ref"])

    names = ["Water"] + [AIOMFAC_KEY.get(i, i) for i in P2_NAMES]
    gf = GibbsFunction(names, g, consts=consts, label="part2_gex")
    gf.G, gf.dG = G, jax.jit(jax.grad(G, argnums=(0, 1)))
    return gf


def part2_activity_fn(gf: GibbsFunction):
    """(m (14,) free molalities per kg water, T) -> (ln a_w, ln gamma_i molal (14,)), the interface of
    s2_part2.model_fn and speciation.speciate, with the compiled derivatives of ``gf`` (from :func:`part2_gibbs`)."""
    def f(m, T):
        k = gf.constants(T)
        m = np.asarray(m, dtype=float)
        dw, di = gf.dG(1.0 / MW, m, k)
        lnxw = -math.log1p(MW * float(m.sum()))
        return lnxw + float(dw), lnxw + np.asarray(di) - np.asarray(k["ref"])
    return f


# ====================================================================================================================
# convenience constructors
# ====================================================================================================================

def part1_liquid(organic, nets, counts) -> GibbsLiquidModel:
    return GibbsLiquidModel([organic], [], part1_gibbs(nets, counts, organic.name))


def part2_liquid(ions, nets, gibbs: GibbsFunction | None = None) -> GibbsLiquidModel:
    """Electrolyte liquid of the given aiomfac_py ions (H+ and SO4-- as components; HSO4- is an explicit species)."""
    return GibbsLiquidModel([], ions, gibbs or part2_gibbs(nets))


__all__ = ["P2_NAMES", "part1_gibbs", "part2_gibbs", "part2_activity_fn", "part1_liquid", "part2_liquid", "ExplicitLiquidModel"]
