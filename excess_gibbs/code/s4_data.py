"""S4 data: AIOMFAC labels for extending the electrolyte G^E surrogate (GEXNet, 14 ions) to Li+, Mg2+ and Br-, the
three further ions of aiomfac_py's validated set (Part 2, Sect. 4.1: adding species to a fixed surrogate).

Same free-species protocol as s2_data.py (v2 settings: salts / acid / carbonate / dilute / concentrated, molalities
log-uniform, electroneutral, 263-313 K), but on the 17-ion basis (the 14 ions of Part 2 followed by Li+, Mg2+, Br-),
and every composition contains at least one of the three new ions.  Compositions that contain both Mg2+ and Br- form
a held-out set (an ion pair never seen in training), generated separately.
Writes ../results/s4_labels.npz (m, T, lgw, lgi, kind, heldout, names, z)."""
import math
import sys
import time

import numpy as np

from aiomfac_py import ActivityModel, Component

CATIONS = [("Na+", 202, 1), ("K+", 203, 1), ("NH4+", 204, 1), ("Ca2+", 221, 2), ("H+", 205, 1)]
ANIONS = [("Cl-", 242, -1), ("NO3-", 245, -1), ("SO4--", 261, -2), ("HSO4-", 248, -1), ("HCO3-", 250, -1),
          ("CO3--", 262, -2), ("IO3-", 246, -1), ("I-", 244, -1), ("OH-", 247, -1)]
NEW = [("Li+", 201, 1), ("Mg2+", 223, 2), ("Br-", 243, -1)]
IONS = CATIONS + ANIONS + NEW                       # the 14 Part 2 ions first, new ions appended
NAMES = [n for n, _, _ in IONS]
Z = np.array([z for _, _, z in IONS], float)
CAT_ALL = [i for i in IONS if i[2] > 0]
AN_ALL = [i for i in IONS if i[2] < 0]
NEW_NAMES = [n for n, _, _ in NEW]
WATER = Component(1, "Water", ((16, 1),))


def build_model():
    comps = [WATER]
    c0, a0 = CAT_ALL[0], AN_ALL[0]
    pairs = [(c, a0) for c in CAT_ALL] + [(c0, a) for a in AN_ALL[1:]]
    for (cn, cs, cz), (an, as_, az) in pairs:
        g = math.gcd(cz, -az)
        comps.append(Component(len(comps) + 1, cn + an, ((cs, -az // g), (as_, cz // g))))
    return ActivityModel(comps, assume_complete=True)


MODEL = build_model()
MX = MODEL.mixture
POS = [(z > 0, MX.cat_index[s] if z > 0 else MX.an_index[s]) for _, s, z in IONS]
NGI = max(MX.ngi, 1)
NC = MX.sr.n_cation


def ln_gamma(m, T):
    """m: molalities of the 17 species -> ln gamma_w (mole-fraction scale), ln gamma_i (molal, infinite dilution)."""
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


P_KIND = [0.35, 0.15, 0.15, 0.05, 0.30]
MAX_CONC, MAX_I = 80.0, 200.0


def sample(rng):
    kind = rng.choice(["salts", "acid", "carbonate", "dilute", "concentrated"], p=P_KIND)
    m = np.zeros(len(IONS))
    cat_names = [n for n, _, _ in CAT_ALL]
    an_names = [n for n, _, _ in AN_ALL]
    if kind == "acid":
        sel = ["H+", "HSO4-", "SO4--"] + list(rng.choice(["Na+", "NH4+", "K+", "Cl-", "NO3-", "Li+", "Mg2+", "Br-"],
                                                         rng.randint(0, 3), replace=False))
    elif kind == "carbonate":
        sel = ["HCO3-", "CO3--", "OH-", "H+"] + list(rng.choice(["Na+", "K+", "Ca2+", "Cl-", "Li+", "Mg2+", "Br-"],
                                                                rng.randint(1, 3), replace=False))
    elif kind == "concentrated":
        nc_, na_ = rng.randint(1, 3), 1 if rng.rand() < 0.6 else 2
        sel = list(rng.choice([n for n in cat_names if n != "H+"], nc_, replace=False)) + \
            list(rng.choice(["Cl-", "NO3-", "SO4--", "I-", "Br-"], na_, replace=False))
    else:
        nc_, na_ = rng.randint(1, 3), rng.randint(1, 3)
        sel = list(rng.choice(cat_names, nc_, replace=False)) + list(rng.choice(an_names, na_, replace=False))
    if not any(n in sel for n in NEW_NAMES):                 # every composition contains a new ion
        sel.append(str(rng.choice(NEW_NAMES)))
    hi = 0.3 if kind == "dilute" else MAX_CONC if kind == "concentrated" else 12.0
    for n in set(sel):
        lo = -8 if n in ("H+", "OH-") else -4
        m[NAMES.index(n)] = 10 ** rng.uniform(lo, math.log10(hi))
    q = float(Z @ m)
    if abs(q) > 0:
        cands = [k for k in range(len(IONS)) if m[k] > 0 and Z[k] * q < 0]
        if not cands:
            return None
        k = max(cands, key=lambda i: m[i])
        m[k] += abs(q / Z[k])
    if 0.5 * float(Z ** 2 @ m) > MAX_I:
        return None
    return m, float(rng.uniform(263.15, 313.15)), kind


def generate(n_train, n_held, seed=0):
    rng = np.random.RandomState(seed)
    iMg, iBr = NAMES.index("Mg2+"), NAMES.index("Br-")
    out = {k: [] for k in ("m", "T", "lgw", "lgi", "kind", "heldout")}
    nt = nh = 0
    t0 = time.time()
    while nt < n_train or nh < n_held:
        s = sample(rng)
        if s is None:
            continue
        m, T, kind = s
        held = m[iMg] > 0 and m[iBr] > 0
        if (held and nh >= n_held) or (not held and nt >= n_train):
            continue
        try:
            lgw, lgi = ln_gamma(m, T)
        except Exception:                                    # noqa: BLE001
            continue
        if not (np.isfinite(lgw) and np.all(np.isfinite(lgi[m > 0]))):
            continue
        for k, v in zip(out, (m, T, lgw, np.where(m > 0, lgi, np.nan), kind, held)):
            out[k].append(v)
        nh += held; nt += not held
        if (nt + nh) % 5000 == 0:
            print(f"  {nt} + {nh} held-out ({time.time()-t0:.0f}s)", flush=True)
    return {k: np.array(v) for k, v in out.items()}


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
    n_held = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    d = generate(n_train, n_held)
    np.savez_compressed("../results/s4_labels.npz", **d, names=np.array(NAMES), z=Z)
    print(len(d["m"]), "samples;", int(d["heldout"].sum()), "with the held-out pair Mg2+/Br-")
