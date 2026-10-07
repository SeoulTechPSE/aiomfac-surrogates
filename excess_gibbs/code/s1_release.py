"""S1 release ensemble: ge_silu_sob trained on all BIMOG + MCM molecules with the protocol of the Part 1 release
checkpoint (seeded 10 % validation, <= 300 epochs, patience 30; 3 seeds).  Writes ../results/s1_models/
ge_silu_sob_release_seed{k}.pt and ../results/s1_release_meta.json."""
import json
import time

import numpy as np
import torch

import s1_train as S

allm = np.isin(S.SRC, ["bimog", "mcm"])
pts = np.where(allm[S.PM])[0]
meta = {"model": "ge_silu_sob", "n_molecules": int(allm.sum()), "n_points": int(len(pts)), "members": []}
for seed in range(3):
    t0 = time.time()
    tr = pts.copy(); np.random.RandomState(seed).shuffle(tr)
    # S.train applies the fixed interpolation split of Table 1; for the release protocol pass the points so that its
    # 20 % interpolation hold-out is not used: call with all points and an empty hold-out instead
    net, ep, best = S.train("ge_silu_sob", tr, seed, max_epochs=300, holdout=0.0)
    torch.save(net.state_dict(), f"../results/s1_models/ge_silu_sob_release_seed{seed}.pt")
    meta["members"].append({"seed": seed, "epochs": ep, "val": best, "train_s": time.time() - t0})
    print(f"seed {seed}: {ep} epochs, val {best:.5f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(meta, open("../results/s1_release_meta.json", "w"), indent=1)
print("RELEASE DONE")
