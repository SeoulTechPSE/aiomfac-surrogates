"""S2 data: AIOMFAC activity coefficients of aqueous electrolyte solutions in the *free-species* basis.

Unlike the Part 2 training data (total ion molalities, AIOMFAC's internal bisulfate/carbonate speciation applied
before the activity coefficients), every sample here prescribes the molalities of the free species directly
(H+, HSO4-, SO4--, HCO3-, CO3--, OH- independent of each other) and evaluates AIOMFAC's LR + MR + SR terms at that
composition, exactly as the explicit-speciation liquid model of aiomfac_py.phase_equilibrium does.  A G^E surrogate
trained on these labels describes the activity model of the free species; the speciation reactions are left to the
Gibbs-energy minimization of the phase-equilibrium solver.

Species: the 14 ions of the Part 2 basis.  Outputs per sample: ln gamma_w (mole-fraction scale, AIOMFAC X basis),
ln gamma_i (molal scale, infinite dilution in water) of every ion present.  Samples are electroneutral.
Writes ../results/s2_labels.npz."""
import math
import sys
import time

import numpy as np

from aiomfac_py import ActivityModel, Component

CATIONS = [("Na+", 202, 1), ("K+", 203, 1), ("NH4+", 204, 1), ("Ca2+", 221, 2), ("H+", 205, 1)]
ANIONS = [("Cl-", 242, -1), ("NO3-", 245, -1), ("SO4--", 261, -2), ("HSO4-", 248, -1), ("HCO3-", 250, -1),
          ("CO3--", 262, -2), ("IO3-", 246, -1), ("I-", 244, -1), ("OH-", 247, -1)]
IONS = CATIONS + ANIONS
NAMES = [n for n, _, _ in IONS]
Z = np.array([z for _, _, z in IONS], float)
WATER = Component(1, "Water", ((16, 1),))


def build_model():
    comps = [WATER]
    c0, a0 = CATIONS[0], ANIONS[0]
    pairs = [(c, a0) for c in CATIONS] + [(c0, a) for a in ANIONS[1:]]
    for (cn, cs, cz), (an, as_, az) in pairs:
        g = math.gcd(cz, -az)
        comps.append(Component(len(comps) + 1, cn + an, ((cs, -az // g), (as_, cz // g))))
    return ActivityModel(comps, assume_complete=True)


MODEL = build_model()
MX = MODEL.mixture
POS = [(z > 0, MX.cat_index[s] if z > 0 else MX.an_index[s]) for _, s, z in IONS]
NGI = max(MX.ngi, 1)
NC = MX.sr.n_cation


def ln_gamma(m: np.ndarray, T: float):
    """m: molalities of the 14 species (mol per kg water) -> ln gamma_w (mole-fraction scale), ln gamma_i (molal)."""
    smc, sma = np.zeros(NGI), np.zeros(NGI)
    for (is_cat, idx), v in zip(POS, m):
        (smc if is_cat else sma)[idx] = v
    xn = np.array([1.0])
    x = MODEL._x_from_molalities(xn, smc, sma)
    lr, mr, sr = MODEL.lr_mr_sr(T, smc, sma, xn, x)
    lgw = lr.ln_gamma_neutral[0] + mr.ln_gamma_neutral[0] + sr.ln_gamma_sr[0]
    out = np.zeros(len(IONS))
    for k, (is_cat, idx) in enumerate(POS):
        if is_cat:
            out[k] = mr.ln_gamma_cation[idx] + sr.ln_gamma_sr[1 + idx] + lr.ln_gamma_cation[idx] - mr.tmolal
        else:
            out[k] = mr.ln_gamma_anion[idx] + sr.ln_gamma_sr[1 + NC + idx] + lr.ln_gamma_anion[idx] - mr.tmolal
    return float(lgw), out


def sample(rng):
    """Random electroneutral free-species composition (2-6 species) and temperature."""
    kind = rng.choice(["salts", "acid", "carbonate", "dilute", "concentrated"], p=P_KIND)
    m = np.zeros(len(IONS))
    if kind == "acid":
        sel = ["H+", "HSO4-", "SO4--"] + list(rng.choice(["Na+", "NH4+", "K+", "Cl-", "NO3-"], rng.randint(0, 3), replace=False))
    elif kind == "carbonate":
        sel = ["HCO3-", "CO3--", "OH-", "H+"] + list(rng.choice(["Na+", "K+", "Ca2+", "Cl-"], rng.randint(1, 3), replace=False))
    elif kind == "concentrated":                              # one or two salts up to MAX_CONC (supersaturated)
        nc_, na_ = rng.randint(1, 3), 1 if rng.rand() < 0.6 else 2
        sel = list(rng.choice([n for n, _, _ in CATIONS if n != "H+"], nc_, replace=False)) + \
            list(rng.choice(["Cl-", "NO3-", "SO4--", "I-"], na_, replace=False))
    else:
        nc_, na_ = rng.randint(1, 3), rng.randint(1, 3)
        sel = list(rng.choice([n for n, _, _ in CATIONS], nc_, replace=False)) + \
            list(rng.choice([n for n, _, _ in ANIONS], na_, replace=False))
    hi = 0.3 if kind == "dilute" else MAX_CONC if kind == "concentrated" else 12.0
    for n in sel:
        lo = -8 if n in ("H+", "OH-") else -4
        m[NAMES.index(n)] = 10 ** rng.uniform(lo, math.log10(hi))
    q = float(Z @ m)                                         # neutralize with the largest counter-ion present
    if abs(q) > 0:
        cands = [k for k in range(len(IONS)) if m[k] > 0 and Z[k] * q < 0]
        if not cands:
            return None
        k = max(cands, key=lambda i: m[i])
        m[k] += abs(q / Z[k])
    I = 0.5 * float(Z ** 2 @ m)
    if I > MAX_I:
        return None
    return m, float(rng.uniform(263.15, 313.15)), kind


P_KIND = [0.5, 0.2, 0.2, 0.1, 0.0]
MAX_CONC, MAX_I = 12.0, 40.0
OUTFILE = "../results/s2_labels.npz"

if __name__ == "__main__":
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    if len(sys.argv) > 2 and sys.argv[2] == "v2":            # v2: concentrated solutions included
        P_KIND[:] = [0.35, 0.15, 0.15, 0.05, 0.30]
        MAX_CONC, MAX_I = 80.0, 200.0
        OUTFILE = "../results/s2v2_labels.npz"
    rng = np.random.RandomState(0)
    M, Tl, YW, YI, K = [], [], [], [], []
    t0 = time.time()
    while len(M) < n_target:
        s = sample(rng)
        if s is None:
            continue
        m, T, kind = s
        try:
            lgw, lgi = ln_gamma(m, T)
        except Exception:
            continue
        if not (np.isfinite(lgw) and np.all(np.isfinite(lgi[m > 0]))):
            continue
        M.append(m); Tl.append(T); YW.append(lgw); YI.append(np.where(m > 0, lgi, np.nan)); K.append(kind)
        if len(M) % 5000 == 0:
            print(f"  {len(M)} samples ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(OUTFILE, m=np.array(M), T=np.array(Tl), lgw=np.array(YW), lgi=np.array(YI),
                        kind=np.array(K), names=np.array(NAMES), z=Z)
    print(f"{len(M)} samples ({time.time()-t0:.0f}s)")
