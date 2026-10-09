"""Summary of the Li+/Mg2+/Br- extension runs (s4_extend.py): test errors per mode, per new ion and by ionic strength,
the starting point (new weights zero: Debye-Hueckel only for the new ions' interactions) and the data-efficiency curve.
Writes ../results/s4_summary.json."""
import json
import os

os.environ.setdefault("S2_TAG", "s2v2")
import numpy as np
import torch

import s4_extend as X


def load(name, mode, seed):
    net = X.make(mode, seed)
    net.load_state_dict(torch.load(f"../results/s4_models/{name}.pt"))
    return net.eval()


def per_ion(net, idx):
    m, tn, A, b, aw, yi, pr = X.tens(idx)
    _, pi = net(m, tn, A, b, create_graph=False)
    e = np.abs(pi.detach().numpy() - yi.numpy()); prn = pr.numpy()
    out = {n: float(e[prn[:, X.NAMES17.index(n)], X.NAMES17.index(n)].mean()) for n in X.NEW}
    I = 0.5 * (X.MA[idx] * X.Z17 ** 2).sum(1)
    for lo, hi in ((0, 1), (1, 6), (6, 20), (20, 1e9)):
        sel = (I >= lo) & (I < hi)
        if sel.any():
            out[f"I_{lo}_{int(min(hi, 999))}"] = {"n": int(sel.sum()), "mae_lng_ion": float(e[sel][prn[sel]].mean())}
    return out


if __name__ == "__main__":
    runs = [json.loads(l) for l in open("../results/s4_runs.jsonl")]
    res = {"runs": runs, "n": {"new_train": int(len(X.NEW_TR)), "new_test": int(len(X.NEW_TE)),
                               "new_heldout_MgBr": int(len(X.NEW_HELD)), "old_test": int(len(X.OLD_TE))}}
    # starting point: base network with zero new weights
    res["start"] = {"new_test": X.evaluate(X.make("frozen", 0), X.NEW_TE),
                    "new_heldout_MgBr": X.evaluate(X.make("frozen", 0), X.NEW_HELD)}
    res["per_ion"] = {}
    for r in runs:
        if "release" in r["name"] or "new_test" not in r:
            continue
        net = load(r["name"], r["mode"], r["seed"])
        res["per_ion"][r["name"]] = {"new_test": per_ion(net, X.NEW_TE), "heldout": per_ion(net, X.NEW_HELD)}
    I = 0.5 * (X.MA * X.Z17 ** 2).sum(1)
    res["ionic_strength_median"] = {"new_test": float(np.median(I[X.NEW_TE])),
                                    "heldout": float(np.median(I[X.NEW_HELD])), "old_test": float(np.median(I[X.OLD_TE]))}
    json.dump(res, open("../results/s4_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1)[:6000])
