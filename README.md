# DS-ROM: Data regularization and deconvolution for transport-dominated ROMs

![Tests](https://github.com/flabowski/POD-ROM-and-gaussian-convolution/actions/workflows/tests.yml/badge.svg)

This repository contains the code and notebooks accompanying the article  
**"Data regularization and deconvolution for non-intrusive reduced-order modeling of transport-dominated problems"**  
by Florian Arbes, Kent-Andre Mardal, and Øyvind Jensen.

**Published article:**  
https://www.sciencedirect.com/science/article/pii/S2590037426000932

DS-ROM is a non-intrusive reduced-order modeling framework for transport-dominated problems. Gaussian smoothing is applied to snapshot data to increase feature overlap and improve linear compressibility. After reduced-order prediction, iterative deconvolution aims to invert an estimated blur rather than generically sharpen all gradients.

The paper investigates:
- the effect of smoothing on POD modes and singular-value decay,
- the smoothing-bias versus approximation-error trade-off,
- rank-dependent selection of the smoothing width,
- parameter-adaptive desmoothing,
- stabilized Richardson–Lucy deconvolution,
- one-dimensional linear transport and two-dimensional viscous Burgers test cases.

## Repository structure

- [Notebooks](https://github.com/flabowski/POD-ROM-and-gaussian-convolution/tree/main/notebooks) reproduce the numerical experiments and figures from the article.
- [Source code](https://github.com/flabowski/POD-ROM-and-gaussian-convolution/tree/main/src/smooth_POD_ROM) contains the implementation.

## Installation

```bash
pip install -e .
```

Run the tests with:

```bash
pytest
```

## Data and reproducibility

The archived code and figure notebooks associated with the article are available on Zenodo:

https://doi.org/10.5281/zenodo.20811949

## Citation

If you use this code or DS-ROM in your work, please cite our article [`CITATION.bib`](CITATION.bib).
