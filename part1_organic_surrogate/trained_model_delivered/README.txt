Trained surrogate model: hand-crafted-feature MLP, full BIMOG pool (368 molecules)
====================================================================================

This is the ONE surrogate checkpoint that was actually saved to disk during this
project's exploratory runs (torch.save was only called in the GSA training script).
It corresponds to:
  - Paper Table 1, row "B: full BIMOG" (n=368)
  - Paper Table 2, row "A. Hand-crafted-feature MLP" (the best model overall:
    Gen. MAE water = 0.025, Gen. MAE organic = 0.298 -- matches surrogate_meta.json's
    sanity_gen_mae exactly)
  - The forward model used for the Sobol GSA (Table/Figure 4) and for Part III's
    viscosity pipeline's AIOMFAC-replacement step

Files:
  - surrogate_weights.pt   PyTorch state_dict for the feed-forward MLP
  - surrogate_norm.npz     input/output normalization (mean/std) used at train time
  - surrogate_meta.json    feature names/order, main-group list, and the exact
                            generalization-MAE numbers reported in the paper

All OTHER rows in Tables 1-3 (pools A and C; GNN Models B-G; the three viscosity-
pipeline variants) were trained during this session's exploratory scripts but their
weights were only ever kept in memory, not checkpointed -- so there is nothing else
to hand over as-is. The accompanying notebook, aiomfac_surrogate_paper_notebook.ipynb,
reproduces every one of those training runs end to end and includes a "Save every
trained model from this run" cell near the end that automatically checkpoints every
model object left in memory to ./saved_models/ -- run the notebook once in Colab to
obtain the full set of trained models used in Tables 1-3.
