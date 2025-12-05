# torch_pysis

## Overview

This repository is a partially CUDA-enabled fork of [eljost/pysisyphus](https://github.com/eljost/pysisyphus). It accelerates the RFOptimizer, RS-I-RFO TS optimizer, and EulerPC intrinsic reaction coordinate (IRC) calculations by adding GPU support. The CUDA-enabled optimizers are designed to be used as building blocks within:

- [t-0hmura/pdb2reaction](https://github.com/t-0hmura/pdb2reaction)
- [t-0hmura/mlmm_toolkit](https://github.com/t-0hmura/mlmm_toolkit)

