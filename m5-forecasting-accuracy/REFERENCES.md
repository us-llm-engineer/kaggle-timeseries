# References

Used for the method:

1. S. Makridakis, E. Spiliotis, V. Assimakopoulos, Z. Chen, A. Gaba, I. Tsetlin, R. L. Winkler. *The M5 accuracy competition: Results, findings, and conclusions.* International Journal of Forecasting 38(4), 2022. The task, the 12 aggregation levels and the WRMSSE definition implemented in `src/m5lib.py`.
2. M. Anderer, F. Li. *Hierarchical forecasting with a top-down alignment of independent level forecasts.* arXiv:2103.08250, 2021. V4 follows its bottom-level design (a LightGBM per store without rolling or lagged demand features) and its idea of aligning the bottom-level forecasts to an independent top-level forecast. The asymmetric-gradient multiplier of the paper is replaced here by a post-hoc ratio (`align` in `src/m5lib.py`), and the chosen strength was 0.
3. Z. Islam, A. Mohammed. *Global tree forecasters collapse at the hierarchical aggregate: a five-panel failure characterization.* arXiv:2609.27912. The reason V3 uses a separate aggregate-level model with each series divided by one plus its trailing mean instead of one tree for all levels; its numbers are for monthly data and are not used here.
4. G. Ke, Q. Meng, T. Finley, T. Wang et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.

Read and not used in the published variants:

5. O. Sprangers, W. Wadman, S. Schelter, M. de Rijke. *Hierarchical Forecasting at Scale.* arXiv:2310.12809. Sparse hierarchical loss for a global model; it needs a loss over all 12 levels, which the tree models here do not optimise.
6. R. Musat, A. Cabuz. *Switch-Hurdle: A MoE Encoder with AR Hurdle Decoder for Intermittent Demand Forecasting.* arXiv:2602.22685. The paper does not state which WRMSSE levels, window or scaling it uses, so its M5 number cannot be compared with ours.
