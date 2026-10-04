"""
Step 1 of the inorganic-aerosol-surrogate roadmap: finalize the component basis
and vector spec, and smoke-test that every component in it actually produces
output through aiomfac_py's ActivityModel.

Component basis = exactly the ions aiomfac_py's README lists as Fortran-validated
(see viscosity.py's CATION_SUBGROUP / ANION_SUBGROUP maps, restricted to the
subset completion.py/dissociation.py/carbonate.py know how to auto-complete and
solve dissociation/precipitation equilibria for), plus the solvent H2O and
temperature T.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aiomfac_py import ActivityModel
from aiomfac_py.io import Component

# ---------------------------------------------------------------------------
# 1. Fixed component-basis vector spec (order, subgroup id, charge, unit)
# ---------------------------------------------------------------------------

SOLVENT = {"name": "H2O", "subgroup": 16, "unit": "kg (reference mass)"}

CATIONS = [
    # (name, subgroup id, charge)
    ("Na+",  202, 1),
    ("K+",   203, 1),
    ("NH4+", 204, 1),
    ("Ca2+", 221, 2),
    ("H+",   205, 1),
]

ANIONS = [
    ("Cl-",   242, -1),
    ("NO3-",  245, -1),
    ("SO4--", 261, -2),
    ("HSO4-", 248, -1),
    ("HCO3-", 250, -1),
    ("CO3--", 262, -2),
    ("IO3-",  246, -1),
    ("I-",    244, -1),
    ("OH-",   247, -1),
]

# Fixed input-vector order: (m_Na, m_K, m_NH4, m_Ca, m_H,            <- cation total molalities
#                            m_Cl, m_NO3, m_SO4, m_HSO4, m_HCO3,     <- anion total molalities
#                            m_CO3, m_IO3, m_I, m_OH,
#                            T)
INPUT_VECTOR_SPEC = (
    [f"m_{name}" for name, _, _ in CATIONS]
    + [f"m_{name}" for name, _, _ in ANIONS]
    + ["T_K"]
)
INPUT_UNITS = {**{f"m_{n}": "mol/kg H2O" for n, _, _ in CATIONS},
               **{f"m_{n}": "mol/kg H2O" for n, _, _ in ANIONS},
               "T_K": "K"}

# Output vector spec: ln(gamma) of the *component* species only (closed-form
# propagation to non-components/solids is Step 4, not here).
OUTPUT_VECTOR_SPEC = ["ln_gamma_H2O"] + [f"ln_gamma_{n}" for n, _, _ in CATIONS] + \
                     [f"ln_gamma_{n}" for n, _, _ in ANIONS]

print("=== Finalized input vector spec (len=%d) ===" % len(INPUT_VECTOR_SPEC))
for k in INPUT_VECTOR_SPEC:
    print(f"  {k:10s} [{INPUT_UNITS[k]}]")
print()
print("=== Output vector spec (component ln(gamma), len=%d) ===" % len(OUTPUT_VECTOR_SPEC))
for k in OUTPUT_VECTOR_SPEC:
    print(f"  {k}")
print()


def _salt_stoichiometry(z_cat: int, z_an: int) -> tuple[int, int]:
    """Smallest (cation_qty, anion_qty) that charge-balances a single cation/anion pair."""
    g = math.gcd(abs(z_cat), abs(z_an))
    return abs(z_an) // g, abs(z_cat) // g


@dataclass
class SmokeResult:
    label: str
    ok: bool
    detail: str


def _run_binary_salt(cat_name, cat_sub, z_cat, an_name, an_sub, z_an) -> SmokeResult:
    qc, qa = _salt_stoichiometry(z_cat, z_an)
    water = Component(1, "Water", ((SOLVENT["subgroup"], 1),))
    salt = Component(2, f"{cat_name}{an_name}", ((cat_sub, qc), (an_sub, qa)))
    label = f"H2O + {cat_name}/{an_name} (stoich {qc}:{qa})"
    try:
        model = ActivityModel([water, salt])
        # composition[] is indexed by ORIGINAL component position (0=water, 1=salt), one entry per
        # component; index 0 is a placeholder ActivityModel always overwrites with 1-sum(others) --
        # the real control is composition[1], the salt's own mass fraction (5% here).
        res = model.evaluate([0.0, 0.05], 298.15, basis="mass")
        ok = bool(np.all(np.isfinite(res.ln_gamma)))
        # a non-trivial check, not just "finite": the salt's own ions must actually be present
        # (not the -9999.9 "~0 mole fraction" sentinel that a water-only evaluation would also pass)
        non_trivial = bool(np.all(res.ln_gamma != -9999.9))
        detail = f"ln_gamma finite={ok}, non_sentinel={non_trivial}, n_species={res.ln_gamma.size}"
        return SmokeResult(label, ok and non_trivial, detail)
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(label, False, f"EXCEPTION: {exc!r}")


def main():
    results: list[SmokeResult] = []

    # 1. Water alone (solvent-only sanity check)
    try:
        water = Component(1, "Water", ((SOLVENT["subgroup"], 1),))
        model = ActivityModel([water])
        res = model.evaluate([1.0], 298.15, basis="mass")
        results.append(SmokeResult("H2O only", bool(np.all(np.isfinite(res.ln_gamma))),
                                    f"n_species={res.ln_gamma.size}"))
    except Exception as exc:  # noqa: BLE001
        results.append(SmokeResult("H2O only", False, f"EXCEPTION: {exc!r}"))

    # 2. Every cation paired with every anion -> each of the 14 ions appears in
    #    at least 9 (cations) / 5 (anions) smoke tests; charge-balanced binary salts.
    for cat_name, cat_sub, z_cat in CATIONS:
        for an_name, an_sub, z_an in ANIONS:
            results.append(_run_binary_salt(cat_name, cat_sub, z_cat, an_name, an_sub, z_an))

    n_ok = sum(1 for r in results if r.ok)
    print(f"=== Smoke-test results: {n_ok}/{len(results)} passed ===")
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        print(f"[{status}] {r.label:30s} {r.detail}")

    # Per-ion coverage check: does every one of the 14 ions appear in at least one
    # successful (water+salt) evaluation?
    covered = {name: False for name, _, _ in CATIONS + ANIONS}
    idx = 1  # results[0] is "H2O only"
    for cat_name, _, _ in CATIONS:
        for an_name, _, _ in ANIONS:
            r = results[idx]; idx += 1
            if r.ok:
                covered[cat_name] = True
                covered[an_name] = True
    print()
    print("=== Per-ion coverage (appears in >=1 successful evaluate() call) ===")
    all_covered = True
    for name, _, _ in CATIONS + ANIONS:
        ok = covered[name]
        all_covered &= ok
        print(f"  {name:6s} {'OK' if ok else 'MISSING'}")

    print()
    print("ALL IONS COVERED:", all_covered, " | ALL SMOKE TESTS PASSED:", n_ok == len(results))
    return all_covered and n_ok == len(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
