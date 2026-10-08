"""S2 derivative labels for a Sobolev-type training of the electrolyte G^E surrogate.

For every free-species composition of ``results/s2v2_labels.npz``, the derivatives of AIOMFAC's

    ln a_w (= ln x_w + ln gamma_w, x on the ion basis)  and  ln gamma_i (molal, ions present)

with respect to ln m_j of every species j present, by central differences of the same AIOMFAC function that produced
the labels (s2_data.ln_gamma; step 1e-4 in ln m, float64).  The scaled form d/d ln m_j = m_j d/dm_j is dimensionless,
like the scaled derivatives x(1 - x) d/dx of Part 1.  Entries of absent species are NaN.

Writes ../results/s2v2_dlabels.npz with D (n_samples, 15, 14): row 0 = ln a_w, rows 1.. = ln gamma of the 14 ions.
Usage: python s2_dlabels.py [n_workers]"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

H = 1.0e-4
MW = 0.01801528


def _one(args):
    from s2_data import ln_gamma
    m, T = args
    out = np.full((15, 14), np.nan)
    pres = np.flatnonzero(m > 0)

    def f(mm):
        lgw, lgi = ln_gamma(mm, T)
        return np.concatenate([[np.log(1.0 / (1.0 + MW * mm.sum())) + lgw], lgi])

    try:
        for j in pres:
            mp, mn = m.copy(), m.copy()
            mp[j] *= np.exp(H); mn[j] *= np.exp(-H)
            d = (f(mp) - f(mn)) / (2 * H)
            out[0, j] = d[0]
            out[1 + pres, j] = d[1 + pres]
    except Exception:                                            # noqa: BLE001 - leave the sample unlabelled
        pass
    return out


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    D = np.load("../results/s2v2_labels.npz", allow_pickle=True)
    M, T = D["m"], D["T"]
    t0 = time.time()
    with Pool(workers) as pool:
        res = []
        for k, r in enumerate(pool.imap(_one, zip(M, T), chunksize=200)):
            res.append(r)
            if (k + 1) % 10000 == 0:
                print(f"  {k + 1} samples ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed("../results/s2v2_dlabels.npz", D=np.array(res, dtype=np.float32))
    print(f"{len(res)} samples ({time.time() - t0:.0f}s)")
