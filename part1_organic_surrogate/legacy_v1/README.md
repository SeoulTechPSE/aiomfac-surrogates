# Legacy (v1) material for Part 1

`01_organic_surrogate_notebook_v1.ipynb` is the reproduction notebook of the last v1 release (v1.1.0), kept for
traceability. It cannot be run from this folder; the complete v1 state, including the descriptor-MLP checkpoints
`trained_model_delivered/` and `trained_model_dense/`, is available under the git tag `v1.1.0`.

Its GNN and hybrid models (Part II) ran at reduced scale with one seed and at most 25 epochs; Table 3 of the revised
paper is reproduced at full scale by `../08_gnn_hybrids_multiseed.ipynb`.

The v1 Part V speed benchmark, executed on a Google Colab NVIDIA T4 runtime, produced Run 2 of Appendix A
(`timing_results_gpu_run.json`; `timing_results.json` is the original CPU run).
