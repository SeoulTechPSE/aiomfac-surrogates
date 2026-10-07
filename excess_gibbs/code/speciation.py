"""Dissociation equilibria for a free-species activity model at fixed water (nominal composition -> free species).

AIOMFAC's ActivityModel takes nominal ion molalities and solves the bisulfate (HSO4- <-> H+ + SO4--) and, in carbonate
systems, the bicarbonate equilibria (CO2(aq) + H2O <-> H+ + HCO3-, HCO3- <-> H+ + CO3--, H2O <-> H+ + OH-) with its own
activity coefficients, including the stoichiometric CaSO4(s) pre-step of the joint system.  A free-species model
(the excess-Gibbs-energy surrogate, or AIOMFAC evaluated at a prescribed free composition) gives activity
coefficients only; this module solves the same equilibria with them:

  * for fixed activity coefficients the free molalities follow in closed form from a_H (mass action and the S, C
    balances), and the proton balance m_H + m_HSO4 + m_HCO3 + 2 m_CO2 - m_OH = H_x is monotonic in ln m_H (brentq);
  * the activity coefficients are then re-evaluated at the new composition (damped fixed-point iteration).

The system flags follow AIOMFAC's completion rules (aiomfac_py.completion): bisulfate if HSO4- is given, or H+ and
SO4--, or HCO3- and SO4--; carbonate if HCO3- is given, or H+ and CO3--, or HSO4- and CO3--.  CO2(aq) is a molal
solute with AIOMFAC's salting-out coefficient (aiomfac_py.carbonate.gamma_co2_mr); like AIOMFAC's explicit-speciation
liquid model it does not enter the activities of the other species."""
import math

import numpy as np
from scipy.optimize import brentq

from aiomfac_py.carbonate import ln_k1_hco3_at_t, ln_k2_hco3_at_t, ln_kw_at_t
from aiomfac_py.dissociation import ln_k_hso4_at_t
from aiomfac_py.params import load_subgroup_params

MW = 0.01801528
NAMES = ["Na+", "K+", "NH4+", "Ca2+", "H+", "Cl-", "NO3-", "SO4--", "HSO4-", "HCO3-", "CO3--", "IO3-", "I-", "OH-"]
SUB = [202, 203, 204, 221, 205, 242, 245, 261, 248, 250, 262, 246, 244, 247]
Z = np.array([1, 1, 1, 2, 1, -1, -1, -2, -1, -1, -2, -1, -1, -1], float)
IX = {n: i for i, n in enumerate(NAMES)}
_SG = load_subgroup_params()
LAMBDA = np.array([_SG.lambdaIN[s - 201] for s in SUB])       # CO2(aq) salting-out coefficients


def flags(m):
    p = lambda n: m[IX[n]] > 0
    bis = p("HSO4-") or (p("H+") and p("SO4--")) or (p("HCO3-") and p("SO4--"))
    carb = p("HCO3-") or (p("H+") and p("CO3--")) or (p("HSO4-") and p("CO3--"))
    return bis, carb


def speciate(model, m_nom, T, tol=1e-11, max_iter=200):
    """model(m_free (14,), T) -> (ln a_w, ln gamma_i molal (14,)).  m_nom: nominal molalities (mol per kg water).

    Returns dict(m=free molalities per kg of the final water, ln_a_w, ln_gamma, n_w (mol water per initial kg),
    m_co2, iters, converged, caso4)."""
    m = np.array(m_nom, float).copy()
    bis, carb = flags(m)
    if not (bis or carb):
        lnaw, lng = model(m, T)
        return dict(m=m, ln_a_w=lnaw, ln_gamma=lng, n_w=1.0 / MW, m_co2=0.0, iters=0, converged=True, caso4=0.0)
    # totals in mol per initial kg of water
    S = m[IX["SO4--"]] + m[IX["HSO4-"]]
    C = m[IX["HCO3-"]] + m[IX["CO3--"]]
    Hx = m[IX["H+"]] + m[IX["HSO4-"]] + m[IX["HCO3-"]] - m[IX["OH-"]]
    oh0 = m[IX["OH-"]]
    ca, caso4 = m[IX["Ca2+"]], 0.0
    if bis and carb and ca > 0 and S > 0:                      # AIOMFAC's stoichiometric CaSO4(s) pre-step
        if ca < S:
            caso4, ca = ca, 0.0
        else:
            res = min(1e-5 * S, 1e3 * np.finfo(float).eps)
            caso4, ca = S - res, ca - S + res
        S -= caso4
        m[IX["Ca2+"]] = ca
    lk_hs, lk1, lk2, lkw = ln_k_hso4_at_t(T), ln_k1_hco3_at_t(T), ln_k2_hco3_at_t(T), ln_kw_at_t(T)
    nw = 1.0 / MW
    co2 = 0.0
    # initial guess: everything dissociated, neutral water
    lng = None
    lnaw = 0.0
    damp = 1.0
    prev = None
    for it in range(max_iter):
        wk = nw * MW                                            # kg water
        if lng is None:
            lnaw, lng = model(np.maximum(m, 0.0), T)
        lco2 = 2.0 * float(LAMBDA @ np.maximum(m, 0.0))
        g = lambda n: lng[IX[n]]

        def comp(lmh):
            mh = math.exp(lmh); lah = lmh + g("H+")
            out = {"H+": mh}
            if bis:
                r1 = math.exp(lah + g("SO4--") - lk_hs - g("HSO4-"))
                s_ = S / wk
                out["SO4--"] = s_ / (1 + r1); out["HSO4-"] = s_ * r1 / (1 + r1)
            if carb:
                r2 = math.exp(lah + g("CO3--") - lk2 - g("HCO3-"))     # HCO3/CO3
                r3 = math.exp(lah + g("HCO3-") - lk1 - lnaw - lco2)     # CO2/HCO3
                c_ = C / wk
                f_co3 = 1.0 / (1 + r2 + r2 * r3)
                out["CO3--"] = c_ * f_co3; out["HCO3-"] = c_ * r2 * f_co3; out["CO2"] = c_ * r2 * r3 * f_co3
                out["OH-"] = math.exp(lkw + lnaw - lah - g("OH-"))
            return out

        def bal(lmh):
            o = comp(lmh)
            return (o["H+"] + o.get("HSO4-", 0.0) + o.get("HCO3-", 0.0) + 2 * o.get("CO2", 0.0) - o.get("OH-", 0.0)
                    - (Hx + (oh0 if not carb else 0.0)) / wk)
        lo, hi = -60.0, math.log(max(abs(Hx) / wk, 1e-30) * 10 + 10)
        if bal(lo) > 0:                                          # no protons at all (e.g. pure sulfate salt)
            lmh = lo
        else:
            lmh = brentq(bal, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
        o = comp(lmh)
        new = m.copy()
        for k, v in o.items():
            if k != "CO2":
                new[IX[k]] = v
        co2_new = o.get("CO2", 0.0)
        # water: OH- formation consumes water, CO2(aq) formation releases it (amounts per initial kg)
        nw_new = 1.0 / MW - ((new[IX["OH-"]] * wk - oh0) if carb else 0.0) + co2_new * wk
        # molalities of the unaffected species scale with the water mass
        scale = (nw * MW) / (nw_new * MW)
        for k in range(14):
            if NAMES[k] not in o:
                new[k] = m[k] * scale
        for k in o:
            if k == "CO2":
                co2_new *= wk / (nw_new * MW)
            else:
                new[IX[k]] *= wk / (nw_new * MW)
        lnaw_n, lng_n = model(np.maximum(new, 0.0), T)
        step = np.max(np.abs(np.log(np.maximum(new, 1e-300)) - np.log(np.maximum(m, 1e-300)))[new > 1e-200]) if prev is not None else 1.0
        if prev is not None and step > prev:
            damp = max(0.2, damp * 0.7)
        prev = step
        m = np.where(new > 0, (1 - damp) * m + damp * new, new) if it else new
        nw, co2 = nw_new, co2_new
        lnaw = (1 - damp) * lnaw + damp * lnaw_n if it else lnaw_n
        lng = np.where(np.isfinite(lng_n), (1 - damp) * lng + damp * lng_n, lng_n) if it else lng_n
        if step < tol:
            break
    lnaw, lng = model(np.maximum(m, 0.0), T)
    return dict(m=m, ln_a_w=lnaw, ln_gamma=lng, n_w=nw, m_co2=co2, iters=it + 1, converged=bool(step < 1e-8), caso4=caso4)
