"""Fortran AIOMFAC (andizuend/AIOMFAC, AIOMFAC-web v3.14) versus the pure-Python port aiomfac_py, same machine, same
session, one CPU thread, same cases as the cost benchmarks of both papers:

  p1_bench   the six molecules of part1_reruns/ge_cost.py (binary water + organic, w = 0.5, 298.15 K):
             model construction and one evaluation
  p1_multi   water + 1 ... 40 MCM organics (equal mass fractions, 298.15 K), as cloud_timings.py
  p2_fixed   NaCl at the composition of s2_part2.py cost (model built once)
  p2_vary    the 300 ion combinations of s2_part2.py cost (model built and evaluated for every point)

Fortran: a timing driver (fortran_bench/bench_main.f90, built with fortran_bench/build.sh) around AIOMFAC_inout, built twice -- with the flags of the
repository's build_command_line.txt (-O3 -fbounds-check ...) and with -O3 -march=native without bounds checking.
AIOMFAC_inout is timed with the viscosity branch off (activities only, as aiomfac_py's ActivityModel.evaluate) and on
(AIOMFAC-web default, which adds finite-difference activity perturbations for AIOMFAC-VISC; not for the
multicomponent organic systems, where the viscosity branch stops on an array bound in ModPureViscosPar).
Water activities of both codes are compared for every system.  Writes ../results/fortran_bench.json.

usage: python fortran_bench.py <fbench_dir> [--reuse]   (--reuse: Fortran timings from the raw cache)"""
import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components

sys.path.insert(0, "../../part2_inorganic_surrogate")
import generate_training_data as gtd  # noqa: E402

FB = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/fbench")
WATER = Component(1, "Water", ((16, 1),))
BENCH = ["CC(=O)C1CC(CC(=O)O)C1(C)C", "OC(=O)CC1CC(C(=O)O)C1(C)C", "OCC(O)CO", "CC(=O)O", "OCC(O)C(O)C(O)C(O)CO",
         "OC(=O)c1ccccc1"]


def timeit(fn, n, warm=3):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


def cases():
    out = []                                  # (group, label, comps, [(T, w_2..w_n)])
    for smi in BENCH:
        out.append(("p1_bench", smi, smiles_to_components([smi]).components, [(298.15, [0.5])]))
    pool = json.load(open("../results/s1_pool.json"))["pool"]
    mcm = [o["subgroups"] for o in pool if o["src"] == "mcm"][:400]
    rng = np.random.RandomState(1)
    for n_org in (1, 2, 5, 10, 20, 40):
        picks = [mcm[i] for i in rng.choice(len(mcm), n_org, replace=False)]
        comps = [WATER] + [Component(k + 2, f"o{k}", tuple(map(tuple, p))) for k, p in enumerate(picks)]
        out.append(("p1_multi", n_org, comps, [(298.15, [1.0 / (n_org + 1)] * n_org)]))
    comps, fr, _, _ = gtd._build_system({("Na+", "Cl-"): 1.0}, 298.15)
    out.append(("p2_fixed", "NaCl", comps, [(298.15, list(fr))]))
    rng = np.random.default_rng(1)
    for k in range(300):
        sm, T = gtd.sample_combo(rng)
        comps, fr, _, _ = gtd._build_system(sm, T)
        out.append(("p2_vary", k, comps, [(T, list(fr))]))
    return out


def write_input(cs, path, nrb, nre):
    lines = [str(len(cs))]
    for i, (g, _, comps, pts) in enumerate(cs):
        b, e = nrb[g], nre[g]
        lines.append(f"{i} {len(comps)} {len(pts)} {b} {e} {0 if g == 'p1_multi' else 1}")
        for c in comps:
            subs = list(c.subgroups)
            lines.append(f"{len(subs)} " + " ".join(f"{s} {q}" for s, q in subs))
        for T, w in pts:
            lines.append(f"{T:.6f} " + " ".join(f"{v:.15e}" for v in w))
    open(path, "w").write("\n".join(lines) + "\n")


def run_fortran(exe, inp):
    """one process per system: the Fortran keeps saved state of the dissociation solver between calls (sized for the
    first system), so systems are not mixed within one run"""
    blocks, lines = [], open(inp).read().split("\n")[1:]
    i = 0
    while i < len(lines) and lines[i].strip():
        h = lines[i].split(); ncp, npts = int(h[1]), int(h[2])
        blocks.append(lines[i:i + 1 + ncp + npts]); i += 1 + ncp + npts
    rows = {}
    for b in blocks:
        f1 = inp + ".one"
        open(f1, "w").write("1\n" + "\n".join(b) + "\n")
        r = subprocess.run([exe, f1], cwd=FB, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            p = line.split()
            if len(p) == 7 and p[0].isdigit():
                rows[int(p[0])] = {"build_us": float(p[2]), "eval_us": float(p[3]), "eval_visc_us": float(p[4]),
                                   "err": int(p[5]), "a_w": float(p[6])}
        if r.returncode != 0:
            rows[int(b[0].split()[0])] = None
    return rows


def python_times(cs, npy):
    rows = {}
    for i, (g, _, comps, pts) in enumerate(cs):
        T, w = pts[0]
        x = [0.0] + list(w)
        m = ActivityModel(comps)
        res = m.evaluate(x, T, basis="mass")
        aw = float(res.activity[0]) if hasattr(res, "activity") else float("nan")
        rows[i] = {"build_us": 1e6 * timeit(lambda: ActivityModel(comps), npy[g][0], warm=1),
                   "eval_us": 1e6 * timeit(lambda: m.evaluate(x, T, basis="mass"), npy[g][1], warm=1),
                   "a_w": aw}
    return rows


if __name__ == "__main__":
    cs = cases()
    nrb = {"p1_bench": 2000, "p1_multi": 200, "p2_fixed": 2000, "p2_vary": 200}
    nre = {"p1_bench": 20000, "p1_multi": 2000, "p2_fixed": 20000, "p2_vary": 2000}
    inp = os.path.join(FB, "bench_input.txt")
    write_input(cs, inp, nrb, nre)
    cache = "../results/fortran_bench_fortran_raw.json"
    if os.path.exists(cache) and "--reuse" in sys.argv:
        F = {k: {int(i): v for i, v in d.items()} for k, d in json.load(open(cache)).items()}
    else:
        F = {name: run_fortran(os.path.join(FB, exe), inp) for name, exe in
             (("official", "bench_official.out"), ("O3_nobc", "bench_O3.out"))}
        json.dump(F, open(cache, "w"))
    npy = {"p1_bench": (100, 500), "p1_multi": (20, 200), "p2_fixed": (100, 500), "p2_vary": (3, 10)}
    P = python_times(cs, npy)
    out = {"machine": {"cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                                    if l.startswith("model name")), ""),
                       "platform": platform.platform(),
                       "gfortran": subprocess.run(["gfortran", "--version"], capture_output=True,
                                                  text=True).stdout.splitlines()[0],
                       "python": platform.python_version(), "threads": 1},
           "cases": []}
    for i, (g, lab, comps, pts) in enumerate(cs):
        out["cases"].append({"group": g, "label": lab, "ncomp": len(comps), "python": P[i],
                             **{f"fortran_{k}": F[k].get(i) for k in F}})
    json.dump(out, open("../results/fortran_bench.json", "w"), indent=1)

    def med(g, key, src):
        v = [c[src][key] for c in out["cases"] if c["group"] == g and c[src] and c[src]["err"] == 0] \
            if src != "python" else [c[src][key] for c in out["cases"] if c["group"] == g]
        return float(np.median(v)) if v else float("nan")
    for g in ("p1_bench", "p2_fixed", "p2_vary"):
        print(g, {src: {k: round(med(g, k, src), 2) for k in ("build_us", "eval_us")}
                  for src in ("python", "fortran_official", "fortran_O3_nobc")},
              "visc", round(med(g, "eval_visc_us", "fortran_official"), 1))
    for c in out["cases"]:
        if c["group"] == "p1_multi":
            print("multi", c["label"], round(c["python"]["eval_us"], 1), round(c["fortran_official"]["eval_us"], 2),
                  round(c["fortran_O3_nobc"]["eval_us"], 2))
    d = [abs(c["python"]["a_w"] - c["fortran_official"]["a_w"]) for c in out["cases"]
         if c["fortran_official"] and c["fortran_official"]["err"] == 0 and np.isfinite(c["python"]["a_w"])]
    nerr = sum(1 for c in out["cases"] if not c["fortran_official"] or c["fortran_official"]["err"] != 0)
    print("max |a_w(Fortran) - a_w(Python)|:", max(d) if d else None, "  systems with Fortran error flags:", nerr)
