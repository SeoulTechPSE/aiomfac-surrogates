"""
Step 2 of the inorganic-aerosol-surrogate roadmap: ML training-data generator.

Extends tools/make_cases.py (which writes Fortran-AIOMFAC-web input files for
cross-validation) with a generator that calls ``ActivityModel`` directly --
no Fortran input files involved -- and samples across the four new axes from
the project design doc's "데이터 생성/샘플링 전략" section:

  (a) component-combination random sampling   -- random subset of the 14 fixed
      ions, paired into charge-balanced binary salts, random relative amounts
  (b) log-scale ionic-strength sampling        -- single-salt systems swept
      log-uniformly from dilute to near-saturation
  (c) continuous 275-325 K temperature sampling
  (d) targeted CaSO4(s)-precipitation sampling -- Ca2+ and SO4-- forced to
      co-occur at concentrations high enough to trigger the
      carbonate.py::_precipitate_ca_sulfate step

Each row stores:
  - input vector : total molality [mol/kg H2O] of each of the 14 fixed-basis
    ions (0 for ions absent from that sample) + T [K]  (== INPUT_VECTOR_SPEC
    in inorganic_component_basis.py)
  - output vector: ln(gamma) of water + the 14 component ions, as actually
    returned by ActivityModel.evaluate() (component order matches
    OUTPUT_VECTOR_SPEC) -- NaN where that ion is absent from the system
  - metadata     : sampling axis, number of salts, charge-neutrality residual
    of the INPUT target (should be ~0 by construction), ionic strength

Usage: python tools/generate_training_data.py <output.parquet_or_csv> [n_per_axis]
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from inorganic_component_basis import CATIONS, ANIONS, SOLVENT  # noqa: E402

from aiomfac_py import ActivityModel
from aiomfac_py.io import Component
from aiomfac_py.params import load_subgroup_params

SG = load_subgroup_params()
WATER_MW = float(SG.GroupMW[16 - 1]) * 1e-3          # kg/mol
ION_MW = {name: float(SG.SMWC[sub - 201]) * 1e-3 for name, sub, _ in CATIONS}
ION_MW.update({name: float(SG.SMWA[sub - 241]) * 1e-3 for name, sub, _ in ANIONS})
ION_Z = {name: z for name, _, z in CATIONS}
ION_Z.update({name: z for name, _, z in ANIONS})
ION_SUB = {name: sub for name, sub, _ in CATIONS}
ION_SUB.update({name: sub for name, sub, _ in ANIONS})

CATION_NAMES = [n for n, _, _ in CATIONS]
ANION_NAMES = [n for n, _, _ in ANIONS]
ALL_ION_NAMES = CATION_NAMES + ANION_NAMES

INPUT_COLS = [f"m_{n}" for n in ALL_ION_NAMES] + ["T_K"]
OUTPUT_COLS = ["ln_gamma_H2O"] + [f"ln_gamma_{n}" for n in ALL_ION_NAMES]


def _salt_stoich(z_cat: int, z_an: int) -> tuple[int, int]:
    g = math.gcd(abs(z_cat), abs(z_an))
    return abs(z_an) // g, abs(z_cat) // g


def _build_system(salt_molality: dict[tuple[str, str], float], T_K: float):
    """salt_molality: {(cation_name, anion_name): molality [mol/kg H2O]} -> (components, mass_fracs, ion_totals)."""
    water = Component(1, "Water", ((SOLVENT["subgroup"], 1),))
    comps = [water]
    mass_fracs = []
    ion_totals = {n: 0.0 for n in ALL_ION_NAMES}
    for k, ((cat, an), molal) in enumerate(salt_molality.items(), start=2):
        if molal <= 0:
            continue
        qc, qa = _salt_stoich(ION_Z[cat], ION_Z[an])
        mw_salt = qc * ION_MW[cat] + qa * ION_MW[an]          # kg/mol formula unit
        mass_kg_per_kg_water = molal * mw_salt
        comps.append(Component(k, f"{cat}{an}", ((ION_SUB[cat], qc), (ION_SUB[an], qa))))
        mass_fracs.append(mass_kg_per_kg_water)
        ion_totals[cat] += molal * qc
        ion_totals[an] += molal * qa
    total_salt_mass = sum(mass_fracs)                        # kg salt per kg water
    total_mass = 1.0 + total_salt_mass                        # kg water fixed at 1
    fracs = [m / total_mass for m in mass_fracs]              # mass fraction of components 2..N
    return comps, fracs, ion_totals, total_mass


def _charge_residual(ion_totals: dict[str, float]) -> float:
    return sum(ion_totals[n] * ION_Z[n] for n in ALL_ION_NAMES)


_SENTINEL = -9999.9  # model.py's own flag for "this species' mole fraction is ~0" (matches Fortran convention)


def _evaluate(comps, fracs, T_K) -> tuple[dict[str, float], list[str]] | None:
    try:
        model = ActivityModel(comps)
        # ActivityModel.evaluate()'s composition[] is indexed by ORIGINAL component position (0=water,
        # 1..=salts), one entry per component; index 0 is a placeholder it always overwrites internally
        # with 1-sum(others) (composition.py::input_to_mass_frac), so `fracs` (the salts' own mass
        # fractions, as built by _build_system) must be prepended with a dummy water entry here -- a bare
        # `fracs` of length len(comps)-1 would silently assign the FIRST salt's fraction to water's slot
        # instead (leaving every real salt at its zero-initialized default -- a dilute-limit no-op, not
        # the intended composition).
        res = model.evaluate([0.0] + list(fracs), T_K, basis="mass")
    except Exception:
        return None
    if not np.all(np.isfinite(res.ln_gamma)):
        return None
    sr = model.mixture.sr
    nn, nc = sr.n_neutral, sr.n_cation
    out = {"ln_gamma_H2O": float(res.ln_gamma[0])}
    sentinel_ions = []
    for n in ALL_ION_NAMES:
        sub = ION_SUB[n]
        if sub in sr.cations:
            idx = nn + sr.cations.index(sub)
        elif sub in sr.anions:
            idx = nn + nc + sr.anions.index(sub)
        else:
            out[f"ln_gamma_{n}"] = np.nan
            continue
        val = float(res.ln_gamma[idx])
        if val == _SENTINEL:
            # species present in the basis but driven to ~0 mole fraction -- either because the system is
            # at the extreme-dilute end of the sampled range, or because an equilibrium/precipitation step
            # (CaSO4(s), carbonate speciation, ...) consumed nearly all of it. Either way this is not a
            # regression target: store NaN + flag it in metadata instead of letting a -9999.9 sentinel
            # leak into the training labels.
            out[f"ln_gamma_{n}"] = np.nan
            sentinel_ions.append(n)
        else:
            out[f"ln_gamma_{n}"] = val
    return out, sentinel_ions


# ---------------------------------------------------------------------------
# Sampling axes
# ---------------------------------------------------------------------------

def sample_combo(rng: np.random.Generator):
    n_cat = rng.integers(1, 4)
    n_an = rng.integers(1, 4)
    cats = rng.choice(CATION_NAMES, size=n_cat, replace=False)
    ans = rng.choice(ANION_NAMES, size=n_an, replace=False)
    salt_molality = {}
    for cat, an in itertools.product(cats, ans):
        salt_molality[(cat, an)] = float(10 ** rng.uniform(-4, math.log10(2.0)))
    T_K = float(rng.uniform(275.0, 325.0))
    return salt_molality, T_K


def sample_log_ionic_strength(rng: np.random.Generator):
    cat = rng.choice(CATION_NAMES)
    an = rng.choice(ANION_NAMES)
    molal = float(10 ** rng.uniform(-5, math.log10(6.0)))     # ~dilute to near-saturation
    T_K = float(rng.uniform(275.0, 325.0))
    return {(cat, an): molal}, T_K


def sample_temperature_sweep(rng: np.random.Generator):
    cat = rng.choice(CATION_NAMES)
    an = rng.choice(ANION_NAMES)
    molal = float(10 ** rng.uniform(-3, math.log10(2.0)))
    T_K = float(rng.uniform(275.0, 325.0))                    # continuous coverage, resampled per point
    return {(cat, an): molal}, T_K


def sample_caso4_precip(rng: np.random.Generator):
    # Force Ca2+/SO4-- co-occurrence high enough to trigger the precipitation step, plus a spectator
    # salt (Na+/Cl-) at a moderate level so the system isn't a bare binary electrolyte every time.
    m_caso4 = float(10 ** rng.uniform(-2.0, math.log10(1.5)))   # up to well past CaSO4's real solubility
    salt_molality = {("Ca2+", "SO4--"): m_caso4}
    if rng.uniform() < 0.7:
        salt_molality[("Na+", "Cl-")] = float(10 ** rng.uniform(-3, math.log10(1.0)))
    T_K = float(rng.uniform(275.0, 325.0))
    return salt_molality, T_K


AXES = {
    "combo": sample_combo,
    "log_ionic_strength": sample_log_ionic_strength,
    "temperature_sweep": sample_temperature_sweep,
    "caso4_precip": sample_caso4_precip,
}


def generate(n_per_axis: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n_attempt = {k: 0 for k in AXES}
    n_fail = {k: 0 for k in AXES}
    for axis, sampler in AXES.items():
        made = 0
        while made < n_per_axis:
            n_attempt[axis] += 1
            salt_molality, T_K = sampler(rng)
            comps, fracs, ion_totals, total_mass = _build_system(salt_molality, T_K)
            resid = _charge_residual(ion_totals)
            if abs(resid) > 1e-9 * max(1.0, sum(abs(ion_totals[n]) for n in ALL_ION_NAMES)):
                n_fail[axis] += 1
                continue
            evaluated = _evaluate(comps, fracs, T_K)
            if evaluated is None:
                n_fail[axis] += 1
                continue
            out, sentinel_ions = evaluated
            row = {f"m_{n}": ion_totals[n] for n in ALL_ION_NAMES}
            row["T_K"] = T_K
            row.update(out)
            row["axis"] = axis
            row["n_salts"] = len(salt_molality)
            row["ionic_strength"] = 0.5 * sum(ion_totals[n] * ION_Z[n] ** 2 for n in ALL_ION_NAMES)
            row["charge_residual"] = resid
            row["sentinel_ions"] = ",".join(sentinel_ions)  # ions whose ln_gamma was the -9999.9 "~0 mole
                                                              # fraction" flag (dilution underflow or
                                                              # precipitation/equilibrium depletion)
            rows.append(row)
            made += 1
    df = pd.DataFrame(rows)
    print("=== generation summary ===")
    for axis in AXES:
        print(f"  {axis:20s} kept={n_per_axis:5d}  attempted={n_attempt[axis]:5d}  failed={n_fail[axis]:4d}")
    return df


def validate(df: pd.DataFrame) -> bool:
    ok = True
    max_resid = df["charge_residual"].abs().max()
    print(f"max |charge residual| = {max_resid:.3e}  (should be ~0)")
    ok &= bool(max_resid < 1e-8)
    for col in INPUT_COLS:
        if col == "T_K":
            in_range = df[col].between(275.0, 325.0).all()
        else:
            in_range = (df[col] >= 0).all()
        print(f"range check {col:10s}: {'OK' if in_range else 'FAIL'}")
        ok &= bool(in_range)
    n_finite_out = df[OUTPUT_COLS].apply(lambda c: np.isfinite(c.dropna()).all()).all()
    print(f"all stored outputs finite: {bool(n_finite_out)}")
    ok &= bool(n_finite_out)
    n_precip_axis = (df["axis"] == "caso4_precip").sum()
    print(f"caso4_precip rows generated: {n_precip_axis}")
    return ok


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("training_data.parquet")
    n_per_axis = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    df = generate(n_per_axis)
    ok = validate(df)
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)
    print(f"\nwrote {len(df)} rows ({len(AXES)} axes x {n_per_axis}) -> {out_path}")
    print("VALIDATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
