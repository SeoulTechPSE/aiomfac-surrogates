Trained surrogate model: hand-crafted-feature MLP, full BIMOG pool (368 molecules),
DENSE FIXED composition grid (200 points/molecule, no per-molecule jitter)
============================================================================

Produced by ../train_dense_surrogate.py. Identical architecture, features, and molecule
split to ../trained_model_delivered/ (the main checkpoint used throughout the paper) --
the only change is composition-axis sampling during training: 200 fixed, shared
water-mass-fraction points per molecule instead of 20 randomly-jittered-per-molecule ones.

Why this checkpoint exists: the main (sparse-grid) checkpoint's predicted
ln(gamma_organic) shows small non-monotonic "wiggles" between training points for some
molecules (most visibly 1,2,4-butanetriol -- 7 slope-sign-changes on a fine 400-point
grid, versus 2 for the true AIOMFAC curve), traced to composition-axis training-data
sparsity. This checkpoint fixes that by densifying the composition grid rather than by
adding an explicit smoothness penalty (see the main notebook's Sec. 8 / paper Sec. 3.11
for the full diagnosis and the reasoning for that choice).

**This checkpoint is used ONLY for the worked-example validation in Sec. 8 of the main
notebook (paper Sec. 3.11, Table 6 / Figure 7), and by Part 2's organic-inorganic
combination notebook (paper Sec. 4).** Every other result in Part 1 -- Tables 1-3, the
GNN comparison, the viscosity pipeline, and the Sobol GSA -- uses
../trained_model_delivered/ as trained in the main notebook's own Part I cells.

Files:
  - surrogate_weights.pt   PyTorch state_dict for the feed-forward MLP
  - surrogate_norm.npz     input/output normalization (mean/std) used at train time
  - surrogate_meta.json    feature names/order, main-group list, and the sanity-check
                            interpolation/generalization MAE from this training run
