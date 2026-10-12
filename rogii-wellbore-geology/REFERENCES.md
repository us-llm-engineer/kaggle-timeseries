# References

Every paper below was checked against its arXiv record (title and authors at the id). The *Role* column says how it was used:
**used** = implemented in this repository's code; **tried** = implemented and evaluated, not kept; **read** = studied while designing, not implemented.

| # | Paper | Role | What it contributed |
| --- | --- | --- | --- |
| 1 | Niklas Dieckow, Katharina Ostaszewski, Philip Heinisch, Henriette Struckmann et al. *Real-time rail vehicle localisation using spatially resolved magnetic field measurements*. arXiv:[2507.19327](https://arxiv.org/abs/2507.19327), 2025. | used | Heavy-tailed match score (Eq. 4) as a U-Net input channel; grid filter in the tracker baseline (`src/tracker.py`). |
| 2 | Jeffrey W. Miller, David B. Dunson. *Robust Bayesian inference via coarsening*. arXiv:[1506.06101](https://arxiv.org/abs/1506.06101), 2015. | used | Coarsened likelihood for the tracker baseline: power ζ = α/(α+n) on the emission. |
| 3 | Olaf Ronneberger, Philipp Fischer, Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. arXiv:[1505.04597](https://arxiv.org/abs/1505.04597), 2015. | used | Encoder-decoder with skip connections; the backbone of `src/grid_unet.py`. |
| 4 | Saurav Sengupta, Scott Kilianski, Suchetha Sharma, Sakina Lashkeri et al. *Combining Residual U-Net and Data Augmentation for Dense Temporal Segmentation of Spike Wave Discharges in Single-Channel EEG*. arXiv:[2601.00459](https://arxiv.org/abs/2601.00459), 2026. | tried | Residual 1-D U-Net for dense labelling; our 1-D variant scored 15.71 ft on the choose fold (`results/baselines/unet1d_choose.jsonl`). |
| 5 | Gangwei Xu, Shujun Chen, Hao Jia, Miaojie Feng et al. *Memory-Efficient Optical Flow via Radius-Distribution Orthogonal Cost Volume*. arXiv:[2312.03790](https://arxiv.org/abs/2312.03790), 2023. | tried | Local cost volume, used iteratively inside the 1-D U-Net variant. |
| 6 | Sergey Alyaev, Ahmed H. Elsheikh. *Direct multi-modal inversion of geophysical logs using deep learning*. arXiv:[2201.01871](https://arxiv.org/abs/2201.01871), 2021. | tried | Multi-hypothesis (MTP) inversion network; chained with a beam it scored 21.11 ft (`results/baselines/mtp_net_choose.jsonl`). |
| 7 | Bjorn Engquist, Yunan Yang. *Optimal Transport Based Seismic Inversion: Beyond Cycle Skipping*. arXiv:[2002.00031](https://arxiv.org/abs/2002.00031), 2020. | tried | Optimal-transport (W2) misfit on GR windows; it scored 46.33 ft (`results/baselines/ot_misfit_choose.jsonl`). |
| 8 | Ming Xu, Sourav Garg, Michael Milford, Stephen Gould. *Deep Declarative Dynamic Time Warping for End-to-End Learning of Alignment Paths*. arXiv:[2303.10778](https://arxiv.org/abs/2303.10778), 2023. | read | Learning alignment paths end to end through dynamic time warping. |
| 9 | Marco Cuturi, Mathieu Blondel. *Soft-DTW: a Differentiable Loss Function for Time-Series*. arXiv:[1703.01541](https://arxiv.org/abs/1703.01541), 2017. | read | Differentiable soft-DTW alignment loss. |
| 10 | Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag et al. *Recurrent Neural Networks for Multivariate Time Series with Missing Values*. arXiv:[1606.01865](https://arxiv.org/abs/1606.01865), 2016. | read | Modelling informative missingness, relevant to the gaps in GR. |
| 11 | Michael Zhang, Khaled K. Saab, Michael Poli, Tri Dao et al. *Effectively Modeling Time Series with Simple Discrete State Spaces*. arXiv:[2303.09489](https://arxiv.org/abs/2303.09489), 2023. | read | State-space sequence models for long series. |
| 12 | Roberto Henry Herrera, Mirko van der Baan. *Automated seismic-to-well ties?*. arXiv:[1209.0201](https://arxiv.org/abs/1209.0201), 2012. | read | Automated log-to-reference ties by warping. |
| 13 | Fengfeng Li, Zhou Feng, Hongliang Wu, Hao Zhang et al. *Depth Matching Method Based on ShapeDTW for Oil-Based Mud Imager*. arXiv:[2512.01611](https://arxiv.org/abs/2512.01611), 2025. | read | Shape-aware DTW for depth matching of well logs. |
| 14 | Yining Yang, Ruogu Chen, Jie Han. *CODA: Cascaded Online Discontinuity-Aware Alignment for Real-Time Image-Based Score Following*. arXiv:[2607.21899](https://arxiv.org/abs/2607.21899), 2026. | read | Online alignment that tolerates discontinuities, an analogue of faults. |

## Competition

- ROGII. *ROGII Wellbore Geology Prediction*. Kaggle competition, 2026. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

## BibTeX

```bibtex
@article{dieckow2025_2507_19327,
  title   = {Real-time rail vehicle localisation using spatially resolved magnetic field measurements},
  author  = {Niklas Dieckow and Katharina Ostaszewski and Philip Heinisch and Henriette Struckmann and Hendrik Ranocha},
  journal = {arXiv preprint arXiv:2507.19327},
  year    = {2025}
}

@article{miller2015_1506_06101,
  title   = {Robust Bayesian inference via coarsening},
  author  = {Jeffrey W. Miller and David B. Dunson},
  journal = {arXiv preprint arXiv:1506.06101},
  year    = {2015}
}

@article{ronneberger2015_1505_04597,
  title   = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author  = {Olaf Ronneberger and Philipp Fischer and Thomas Brox},
  journal = {arXiv preprint arXiv:1505.04597},
  year    = {2015}
}

@article{sengupta2026_2601_00459,
  title   = {Combining Residual U-Net and Data Augmentation for Dense Temporal Segmentation of Spike Wave Discharges in Single-Channel EEG},
  author  = {Saurav Sengupta and Scott Kilianski and Suchetha Sharma and Sakina Lashkeri and Ashley McHugh and Mark Beenhakker and Donald E. Brown},
  journal = {arXiv preprint arXiv:2601.00459},
  year    = {2026}
}

@article{xu2023_2312_03790,
  title   = {Memory-Efficient Optical Flow via Radius-Distribution Orthogonal Cost Volume},
  author  = {Gangwei Xu and Shujun Chen and Hao Jia and Miaojie Feng and Xin Yang},
  journal = {arXiv preprint arXiv:2312.03790},
  year    = {2023}
}

@article{alyaev2021_2201_01871,
  title   = {Direct multi-modal inversion of geophysical logs using deep learning},
  author  = {Sergey Alyaev and Ahmed H. Elsheikh},
  journal = {arXiv preprint arXiv:2201.01871},
  year    = {2021}
}

@article{engquist2020_2002_00031,
  title   = {Optimal Transport Based Seismic Inversion: Beyond Cycle Skipping},
  author  = {Bjorn Engquist and Yunan Yang},
  journal = {arXiv preprint arXiv:2002.00031},
  year    = {2020}
}

@article{xu2023_2303_10778,
  title   = {Deep Declarative Dynamic Time Warping for End-to-End Learning of Alignment Paths},
  author  = {Ming Xu and Sourav Garg and Michael Milford and Stephen Gould},
  journal = {arXiv preprint arXiv:2303.10778},
  year    = {2023}
}

@article{cuturi2017_1703_01541,
  title   = {Soft-DTW: a Differentiable Loss Function for Time-Series},
  author  = {Marco Cuturi and Mathieu Blondel},
  journal = {arXiv preprint arXiv:1703.01541},
  year    = {2017}
}

@article{che2016_1606_01865,
  title   = {Recurrent Neural Networks for Multivariate Time Series with Missing Values},
  author  = {Zhengping Che and Sanjay Purushotham and Kyunghyun Cho and David Sontag and Yan Liu},
  journal = {arXiv preprint arXiv:1606.01865},
  year    = {2016}
}

@article{zhang2023_2303_09489,
  title   = {Effectively Modeling Time Series with Simple Discrete State Spaces},
  author  = {Michael Zhang and Khaled K. Saab and Michael Poli and Tri Dao and Karan Goel and Christopher Ré},
  journal = {arXiv preprint arXiv:2303.09489},
  year    = {2023}
}

@article{herrera2012_1209_0201,
  title   = {Automated seismic-to-well ties?},
  author  = {Roberto Henry Herrera and Mirko van der Baan},
  journal = {arXiv preprint arXiv:1209.0201},
  year    = {2012}
}

@article{li2025_2512_01611,
  title   = {Depth Matching Method Based on ShapeDTW for Oil-Based Mud Imager},
  author  = {Fengfeng Li and Zhou Feng and Hongliang Wu and Hao Zhang and Han Tian and Peng Liu and Lixin Yuan},
  journal = {arXiv preprint arXiv:2512.01611},
  year    = {2025}
}

@article{yang2026_2607_21899,
  title   = {CODA: Cascaded Online Discontinuity-Aware Alignment for Real-Time Image-Based Score Following},
  author  = {Yining Yang and Ruogu Chen and Jie Han},
  journal = {arXiv preprint arXiv:2607.21899},
  year    = {2026}
}
```
