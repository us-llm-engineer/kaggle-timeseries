# References

Used for the method:

1. Q. Chen, C.-Y. Robert. *Multivariate Realized Volatility Forecasting with Graph Neural Network.* arXiv:2112.09015, 2021. The RMSPE training loss, the indicator-by-aggregator window features with cumulative sub-windows, the per-second reading of missing quote and trade seconds, and the comparison of tree and neural baselines.
2. A. Lensky, M. Hao. *Learning to Predict Short-Term Volatility with Order Flow Image Representation.* arXiv:2304.02472, 2023. Order-book and trade tensor as input to a convolutional network next to aggregated features.
3. Z. M. Li, R. J. A. Laeven, M. H. Vellekoop. *Dependent Microstructure Noise and Integrated Volatility Estimation from High-Frequency Data.* arXiv:1704.08964. Pre-averaged returns, noise-moment estimators and the finite-sample bias correction.
4. J. Lee, Y. Lee, J. Kim, A. Kosiorek, S. Choi, Y. W. Teh. *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* ICML 2019, arXiv:1810.00825. V4 uses a plain permutation-equivariant transformer encoder over the set of stocks (no inducing points; the sets have at most 112 elements).
5. G. Ke, Q. Meng, T. Finley, T. Wang et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.

Read and not used:

6. F. Moreno-Pino, S. Zohren. *DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions.* arXiv:2210.04797. Needs the previous days of intraday data of the same asset; window ids here have no order.
7. M. Rosenbaum, J. Zhang. *On the universality of the volatility formation process: when machine learning and rough volatility agree.* arXiv:2206.14114. Needs an ordered 22-day history of daily realised measures.
8. L. Grinsztajn, E. Oyallon, G. Varoquaux. *Why do tree-based models still outperform deep learning on tabular data?* arXiv:2207.08815. Benchmarks use training sets of at most 10,000 to 50,000 rows, a smaller regime than the 303,388 fit rows here.
9. A. De Myttenaere, B. Golden, B. Le Grand, F. Rossi. *Mean Absolute Percentage Error for regression models.* arXiv:1605.02541. Percentage-error loss as weighted absolute error; the squared-error analogue used here (scalar `sum(p/y)/sum((p/y)^2)`) is derived, not taken from the paper.
10. M. B. Deaconu, I. D. Pop. *HAN-Mamba: Hierarchical Selective State Space Networks for Multi-Scale Financial Volatility Forecasting.* arXiv:2610.10323. Uses 60 to 240 previous windows plus daily and weekly histories, so it does not apply to unordered windows.
