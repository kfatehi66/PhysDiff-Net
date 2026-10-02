# PhysDiff-Net: Physics-Guided Diffusion for Synthetic Wireless Traffic Generation

This repository hosts the code release for the paper *PhysDiff-Net: Physics-Guided Diffusion for Synthetic Wireless Traffic Generation*, accepted for publication in the IEEE Transactions on Machine Learning in Communications and Networking.

> **Status:** the code, trained model, and evaluation scripts are being prepared for release. This page will be updated when they are available.

## Overview

PhysDiff-Net is a conditional denoising diffusion model that generates synthetic access-point-level WiFi telemetry conditioned on the hour of day. The denoising network is a 6-layer Transformer encoder (5.62M parameters) in which each access point's feature vector forms one token. The model was trained on measurements collected over about five weeks at 10-second resolution from 11 access points in a municipal WiFi deployment in Blackpool, UK, represented by 29 engineered features per access point (load, traffic, signal quality, mobility, temporal, and spatial descriptors).

The framework has three components:

1. **Variational hour conditioning.** The hour label is mapped to a Gaussian embedding and sampled with the reparameterisation trick. An auxiliary hour-classification loss and a supervised contrastive loss strengthen the temporal signal, and classifier-free guidance steers generation toward the traffic profile of a chosen hour.
2. **Physics-guided training loss.** A differentiable hinge penalty on the batch-level Pearson correlation between user density and SNR, computed on the Tweedie estimate of the clean sample, enforces the anti-correlation expected from co-channel interference (ρ < −0.3). In the Blackpool measurements, SNR and user density are weakly positively correlated, so the loss enforces the interference relationship as a domain prior for the high-load regimes that what-if simulation targets.
3. **Variance-Guided Diffusion Sampling (VGDS).** Strong classifier-free guidance inflates output variance approximately as 1 + ηw², where w is the guidance scale. VGDS rescales each feature of the generated batch toward target statistics after denoising, with the correction factor clipped to [0.65, 1.35], and requires no retraining.

## Release contents

The release will include:

- the PyTorch implementation of PhysDiff-Net, covering training, sampling, and VGDS;
- the trained model checkpoint used for the results reported in the paper;
- the evaluation scripts for the six reported metrics;
- the MATLAB scripts used for distribution analysis, dependency estimation, and spatio-temporal fingerprint extraction.

## Citation

If you use the code or dataset in this repository in your research, please cite the following paper:

K. Fatehi, M. Rahmani Ghourtani, D. Grace, M. Arvaneh, K. K. Leung, and H. Ahmadi, “PhysDiff-Net: Physics-Guided Diffusion for Synthetic Wireless Traffic Generation,” *IEEE Transactions on Machine Learning in Communications and Networking*, 2026.

```bibtex
@article{fatehi2026physdiffnet,
  author  = {Fatehi, Kavan and Rahmani Ghourtani, Mostafa and Grace, David and Arvaneh, Mahnaz and Leung, Kin K. and Ahmadi, Hamed},
  title   = {{PhysDiff-Net}: Physics-Guided Diffusion for Synthetic Wireless Traffic Generation},
  journal = {IEEE Transactions on Machine Learning in Communications and Networking},
  year    = {2026}
}
```

## Acknowledgement

This work was supported in part by the UK Engineering and Physical Sciences Research Council (EPSRC), Hub for All Spectrum Communications, O-RAN intelligent adaptive load balancing and efficiency in highly dense deployments (ORLANDO) project, under Grant EP/X040569/1.
