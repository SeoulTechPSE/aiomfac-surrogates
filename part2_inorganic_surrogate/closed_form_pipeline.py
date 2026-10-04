"""
Step 4 of the inorganic-aerosol-surrogate roadmap: closed-form derivation pipeline.

Scope note (read before anything else): the design doc's canonical-form picture (Yoo, He & Amundson,
2004) assumes a MINIMAL independent-component basis (e.g. the paper's own worked example: {H2O, H+,
HSO4-, NO3-, NH4+}), with every other species -- SO4--, NH3, OH-, ... -- derived from it via the
log-linear mass-action law (Eq. 8) / precipitation law (Eq. 9). Step 1 of this project instead chose a
FIXED 14-ION BASIS matching aiomfac_py's whole Fortran-validated ion set, and Step 3's MLP predicts
ln(gamma) for all 14 directly -- so, as built, nothing is actually "derived" rather than predicted.

What *is* still a clean, Fortran-grounded closed-form test, and the one this script runs: aiomfac_py's
own dissociation.py treats {H+, SO4--} as the two independent species of the bisulfate sub-system and
HSO4- as the one dependent (mass-action-law) species -- exactly Eq. 8's shape.

Two bugs were found and fixed while building this (both self-discovered during the first run, not
pre-known):

  1. ``training_data.parquet``'s ``m_H+``/``m_HSO4-``/``m_SO4--`` columns are the NOMINAL/INPUT total
     molalities (the mass-balance constraints used to build each system), NOT the actual post-
     equilibrium molalities. The original oracle_check mistakenly compared the closed-form quadratic's
     derived (equilibrium) m_HSO4- against that nominal column -- two different physical quantities --
     which produced a spuriously large error (mean~0.15, max~2.0) even though the closed form is
     correct. Fixed by reconstructing fresh test systems directly and reading the TRUE equilibrium
     smc/sma off the ActivityModel result itself (same approach as the earlier one-point manual check
     that got ~1e-13 agreement).
  2. The original surrogate_pipeline_check fed the MLP's *own* direct ln(gamma(HSO4-)) prediction into
     the closed-form quadratic as one of its three gamma inputs. Algebraically, this makes the
     "derived" ln(gamma(HSO4-)) output IDENTICALLY EQUAL to that same input (provable from the
     quadratic's defining equation: a_HSO4 = gamma_H*m_H*gamma_SO4*m_SO4/K = gamma_HSO4*m_HSO4 holds
     exactly at the root by construction) -- confirmed numerically below (max|derived-direct|~1e-9,
     pure float-arithmetic noise). So that comparison was a tautology, not a validation. Fixed by instead
     checking the one thing the closed form *can* independently predict without circularity: the
     EQUILIBRIUM SPECIATION SPLIT (m_H+, m_HSO4-, m_SO4--) implied by feeding it predicted gammas, which
     is compared against the TRUE split from a fresh ActivityModel evaluation -- not algebraically forced
     to match.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
AIOMFAC_SRC = Path(__file__).resolve().parent  # helper modules live next to this file
sys.path.insert(0, str(AIOMFAC_SRC / "tools"))
sys.path.insert(0, str(HERE))

from aiomfac_py import ActivityModel  # noqa: E402
from aiomfac_py.io import Component  # noqa: E402
from aiomfac_py.dissociation import ln_k_hso4_at_t  # noqa: E402
from inorganic_component_basis import SOLVENT  # noqa: E402
from train_mlp_benchmark import IN_COLS, OUT_COLS, load_and_split, train  # noqa: E402

H_SUB, HSO4_SUB, SO4_SUB, NA_SUB, CL_SUB = 205, 248, 261, 202, 242


def solve_bisulfate_closed_form(m_h_max: float, m_sulf_max: float, T_K: float,
                                 gamma_h: float, gamma_hso4: float, gamma_so4: float) -> tuple[float, float, float]:
    """Eq. 8's closed form for HSO4- <-> H+ + SO4--, given FIXED activity coefficients (no SR/MR solve).

    Mass balance: m_h_max = m_H + m_HSO4, m_sulf_max = m_SO4 + m_HSO4 (both fixed, mass-balance inputs).
    Mass action : K = (gamma_H m_H)(gamma_SO4 m_SO4) / (gamma_HSO4 m_HSO4)   [Eq. 8, rearranged]
    Substituting m_H = m_h_max - x, m_SO4 = m_sulf_max - x (x = m_HSO4) gives a quadratic in x.
    Returns (m_H, m_HSO4, m_SO4).
    """
    K = math.exp(ln_k_hso4_at_t(T_K))
    if m_h_max <= 0 or m_sulf_max <= 0:
        return max(m_h_max, 0.0), 0.0, max(m_sulf_max, 0.0)
    D = gamma_h * gamma_so4 / (K * gamma_hso4)
    a_ = D
    b_ = -(D * (m_h_max + m_sulf_max) + 1.0)
    c_ = D * m_h_max * m_sulf_max
    disc = max(b_ * b_ - 4 * a_ * c_, 0.0)
    x_lo = (-b_ - math.sqrt(disc)) / (2 * a_)      # the physical root: 0 <= x <= min(m_h_max, m_sulf_max)
    m_hso4 = min(max(x_lo, 0.0), min(m_h_max, m_sulf_max))
    return m_h_max - m_hso4, m_hso4, m_sulf_max - m_hso4


def _build_and_evaluate(m_h_max: float, m_sulf_max: float, m_na: float, T_K: float):
    """Build water + a nominal (pre-dissociation) H+/HSO4-/SO4-- mix hitting total-H=m_h_max,
    total-S(VI)=m_sulf_max, plus an optional Na+/Cl- spectator salt; evaluate; return the TRUE
    equilibrium state.

    Mass-balance bookkeeping: a nominal unit of HSO4- carries both one H and one S(VI), so
    total_H = nominal(H+) + nominal(HSO4-) and total_S(VI) = nominal(SO4--) + nominal(HSO4-).
    Pick nominal(HSO4-) = 0.5*min(m_h_max, m_sulf_max) (an arbitrary interior starting split --
    the actual equilibrium split is exactly what the quadratic/solver determines), then add the
    remainder of each pool as its own charge-balanced salt: leftover free H+ paired with Cl-,
    leftover free SO4-- paired with Na+, nominal HSO4- paired with Na+ (NOT with H+ -- an earlier
    version of this script mistakenly gave the HSO4- proxy its own extra H+ subgroup, silently
    double-counting H in the mass balance and corrupting the oracle check).
    """
    water = Component(1, "Water", ((SOLVENT["subgroup"], 1),))
    comps = [water]
    fracs = []
    m_hso4_in = 0.5 * min(m_h_max, m_sulf_max)
    m_h_free = m_h_max - m_hso4_in
    m_so4_free = m_sulf_max - m_hso4_in
    if m_h_free > 1e-12:
        comps.append(Component(len(comps) + 1, "HCl-proxy", ((H_SUB, 1), (CL_SUB, 1))))
        fracs.append(m_h_free)
    if m_so4_free > 1e-12:
        comps.append(Component(len(comps) + 1, "NaSO4-proxy", ((NA_SUB, 2), (SO4_SUB, 1))))
        fracs.append(m_so4_free)
    if m_hso4_in > 1e-12:
        comps.append(Component(len(comps) + 1, "NaHSO4-proxy", ((NA_SUB, 1), (HSO4_SUB, 1))))
        fracs.append(m_hso4_in)
    if m_na > 0:
        comps.append(Component(len(comps) + 1, "NaCl", ((NA_SUB, 1), (CL_SUB, 1))))
        fracs.append(m_na)

    from aiomfac_py.params import load_subgroup_params
    sg = load_subgroup_params()
    mw = {H_SUB: sg.SMWC[H_SUB - 201], NA_SUB: sg.SMWC[NA_SUB - 201],
          SO4_SUB: sg.SMWA[SO4_SUB - 241], HSO4_SUB: sg.SMWA[HSO4_SUB - 241], CL_SUB: sg.SMWA[CL_SUB - 241]}
    mass_fracs = []
    for comp, molal in zip(comps[1:], fracs):
        salt_mw = sum(mw[sub] * q for sub, q in comp.subgroups) * 1e-3
        mass_fracs.append(molal * salt_mw)
    total_mass = 1.0 + sum(mass_fracs)
    wtf = [m / total_mass for m in mass_fracs]

    model = ActivityModel(comps)
    res = model.evaluate([0.0] + wtf, T_K, basis="mass")
    if not np.all(np.isfinite(res.ln_gamma)):
        return None
    sr = model.mixture.sr
    nn, nc = sr.n_neutral, sr.n_cation
    if H_SUB not in sr.cations or HSO4_SUB not in sr.anions or SO4_SUB not in sr.anions:
        return None
    ih = nn + sr.cations.index(H_SUB)
    ihso4 = nn + nc + sr.anions.index(HSO4_SUB)
    iso4 = nn + nc + sr.anions.index(SO4_SUB)
    lgH, lgHSO4, lgSO4 = res.ln_gamma[ih], res.ln_gamma[ihso4], res.ln_gamma[iso4]
    if -9999.9 in (lgH, lgHSO4, lgSO4):
        return None
    m_h_true = res.smc[sr.cations.index(H_SUB)]
    m_hso4_true = res.sma[sr.anions.index(HSO4_SUB)]
    m_so4_true = res.sma[sr.anions.index(SO4_SUB)]

    # build the 15-dim MLP input vector (nominal per-salt totals, matching IN_COLS /
    # generate_training_data.py's `ion_totals` convention: the amount of each ion contributed by the
    # salts actually added, not a pooled "total H" / "total S(VI)").
    row = {c: 0.0 for c in IN_COLS}
    for comp, molal in zip(comps[1:], fracs):
        for sub, q in comp.subgroups:
            name = {H_SUB: "H+", NA_SUB: "Na+", SO4_SUB: "SO4--", HSO4_SUB: "HSO4-", CL_SUB: "Cl-"}[sub]
            row[f"m_{name}"] = row.get(f"m_{name}", 0.0) + molal * q
    row["T_K"] = T_K

    return {
        "m_h_max": m_h_max, "m_sulf_max": m_sulf_max, "T_K": T_K,
        "ln_g_h_true": lgH, "ln_g_hso4_true": lgHSO4, "ln_g_so4_true": lgSO4,
        "m_h_true": m_h_true, "m_hso4_true": m_hso4_true, "m_so4_true": m_so4_true,
        "mlp_input_row": row,
    }


def make_samples(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        m_h_max = float(10 ** rng.uniform(-3, 0.5))
        m_sulf_max = float(10 ** rng.uniform(-3, 0.5))
        m_na = float(10 ** rng.uniform(-3, 0.3)) if rng.uniform() < 0.6 else 0.0
        T_K = float(rng.uniform(275.0, 325.0))
        r = _build_and_evaluate(m_h_max, m_sulf_max, m_na, T_K)
        if r is not None:
            out.append(r)
    return out


def oracle_check(samples: list[dict]):
    """4.1: feed the quadratic the TRUE converged gammas; it should reproduce the TRUE equilibrium
    m_HSO4- to near machine precision -- the canonical-form / Eq.8 sanity check."""
    errs = []
    for s in samples:
        gh, ghso4, gso4 = math.exp(s["ln_g_h_true"]), math.exp(s["ln_g_hso4_true"]), math.exp(s["ln_g_so4_true"])
        _, m_hso4_pred, _ = solve_bisulfate_closed_form(s["m_h_max"], s["m_sulf_max"], s["T_K"], gh, ghso4, gso4)
        errs.append(abs(m_hso4_pred - s["m_hso4_true"]))
    errs = np.array(errs)
    print(f"[4.1 oracle check] n={len(samples)}  max|Δm_HSO4-|={errs.max():.3e}  mean={errs.mean():.3e}")
    return errs


def surrogate_pipeline_check(model, scaler, samples: list[dict]):
    """4.2: feed the MLP's PREDICTED gammas into the closed form and compare the DERIVED equilibrium
    speciation split against the TRUE split (independent data -- not algebraically forced to match)."""
    mu, sd = scaler
    df_in = pd.DataFrame([s["mlp_input_row"] for s in samples])
    X = np.log1p(df_in[IN_COLS[:-1]].to_numpy(dtype=np.float32))
    X = np.concatenate([X, df_in[["T_K"]].to_numpy(dtype=np.float32)], axis=1)
    Xs = (X - mu) / sd
    with torch.no_grad():
        pred = model(torch.from_numpy(Xs)).numpy()
    i_h, i_hso4, i_so4 = OUT_COLS.index("ln_gamma_H+"), OUT_COLS.index("ln_gamma_HSO4-"), OUT_COLS.index("ln_gamma_SO4--")

    err_3pred, err_2pred, tautology_check, direct_hso4_err = [], [], [], []
    for k, s in enumerate(samples):
        gh_p, ghso4_p, gso4_p = math.exp(pred[k, i_h]), math.exp(pred[k, i_hso4]), math.exp(pred[k, i_so4])
        gh_t, ghso4_t, gso4_t = math.exp(s["ln_g_h_true"]), math.exp(s["ln_g_hso4_true"]), math.exp(s["ln_g_so4_true"])

        # (a) full surrogate: all 3 gammas MLP-predicted
        _, m_hso4_3pred, _ = solve_bisulfate_closed_form(s["m_h_max"], s["m_sulf_max"], s["T_K"], gh_p, ghso4_p, gso4_p)
        err_3pred.append(abs(m_hso4_3pred - s["m_hso4_true"]))

        # (b) isolate component-only error: H+/SO4-- predicted, HSO4- gamma held at TRUE value
        _, m_hso4_2pred, _ = solve_bisulfate_closed_form(s["m_h_max"], s["m_sulf_max"], s["T_K"], gh_p, ghso4_t, gso4_p)
        err_2pred.append(abs(m_hso4_2pred - s["m_hso4_true"]))

        # tautology check: "derived" ln_gamma_HSO4- from (a)'s split should equal the MLP's own direct
        # ln_gamma_HSO4- prediction EXACTLY (up to float noise) -- confirming this is not an independent signal
        if m_hso4_3pred > 1e-12:
            m_h_d, m_so4_d = s["m_h_max"] - m_hso4_3pred, s["m_sulf_max"] - m_hso4_3pred
            K = math.exp(ln_k_hso4_at_t(s["T_K"]))
            ln_g_hso4_derived = math.log(gh_p * m_h_d * gso4_p * m_so4_d / K) - math.log(m_hso4_3pred)
            tautology_check.append(abs(ln_g_hso4_derived - pred[k, i_hso4]))

        direct_hso4_err.append(abs(pred[k, i_hso4] - s["ln_g_hso4_true"]))

    print(f"[4.2 surrogate pipeline] n={len(samples)}")
    print(f"  m_HSO4- MAE, all 3 gammas MLP-predicted:                 {np.mean(err_3pred):.4e}")
    print(f"  m_HSO4- MAE, only H+/SO4-- predicted (HSO4- gamma=true): {np.mean(err_2pred):.4e}")
    print(f"  max|derived ln_g(HSO4-) - MLP's own direct prediction|:  {np.max(tautology_check):.3e}  "
          f"(tautology check -- should be ~0, confirming it's not an independent validation)")
    print(f"  (for reference) MLP direct ln_gamma_HSO4- MAE vs truth: {np.mean(direct_hso4_err):.4f}")
    return err_3pred, err_2pred


def main():
    print("=== Step 4.1: oracle closed-form check (true gammas -> Eq.8 quadratic) ===")
    samples = make_samples(300, seed=0)
    oracle_check(samples)

    print("\n=== Training (re-using Step 3's MLP setup) for Step 4.2 ===")
    data_path = HERE / "training_data.parquet"
    df_train, df_interp, df_gen = load_and_split(data_path)
    model, scaler, maes, best_val = train(df_train, df_interp, df_gen)
    print(f"(re-trained benchmark MLP; interp masked-MSE={best_val:.4f}, for use inside the pipeline check)")

    print("\n=== Step 4.2: surrogate + closed-form pipeline vs. direct MLP prediction ===")
    surrogate_pipeline_check(model, scaler, samples)


if __name__ == "__main__":
    main()
