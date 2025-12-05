# torch_pysis

## Overview

This repository is a CUDA-enabled fork of [eljost/pysisyphus](https://github.com/eljost/pysisyphus). It accelerates intrinsic reaction coordinate (IRC) calculations by adding GPU support to the RFOptimizer, RSIRFO, and EulerPC pathways. The goal is to deliver faster quantum chemistry workflows on capable hardware.

## Intended Use

The CUDA-enabled optimizers are designed to be used as building blocks within:

- [t-0hmura/pdb2reaction](https://github.com/t-0hmura/pdb2reaction)
- [t-0hmura/mlmm_toolkit](https://github.com/t-0hmura/mlmm_toolkit)

Feel free to integrate torch_pysis into other projects that can benefit from accelerated IRC calculations.
