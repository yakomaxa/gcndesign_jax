# GCNdesign_jax

GCNdesign in **JAX** (WIP).

The original **GCNdesign** was written in **PyTorch**.  
This repository is a re-implementation of the algorithm in **JAX**, with trained parameters ported from Torch into JAX.

Porting GCNdesign into JAX makes it possible to train **conjugate models written in JAX**, such as **AlphaFold** and **ProteinMPNN in JAX**, enabling tighter integration and end-to-end optimization.

## Status

This project is a work in progress. APIs, directory structure, and training scripts may change without notice.

## Scope

- JAX re-implementation of the original GCNdesign model  
- Ported pretrained parameters from the PyTorch version  
- Focus on compatibility with other JAX-based protein models

This repository does **not** currently include:
- A cleaned or finalized training pipeline
- Full documentation or benchmarks

## Reproducibility

Numerical differences between the PyTorch and JAX implementations may exist due to framework-specific behavior and precision differences.

## Original PyTorch version

https://github.com/ShintaroMinami/GCNdesign

## Repository Contents

The root directory currently contains a mixture of scripts used during the porting process and is not yet well organized. This will be cleaned up as the JAX implementation stabilizes.

## License

MIT License, following the original version. See the `LICENSE` files in this repository for details.

## TODO

- [ ] Layer-wise checks against the PyTorch implementation  
- [ ] Clean up repository structure  
- [ ] Add training scripts in JAX  
- [ ] Validate outputs against the PyTorch implementation  
- [ ] Document integration with other JAX protein models


## Logs

Made public: 2026 Mar. 14th

## Intended Audience

This repository is intended for researchers and developers familiar with protein design models and the JAX ecosystem.

Note that the developer (K.S.) is not a native JAX user, so some implementations may not follow best JAX practices (or even good ones). Comments, suggestions, and improvements are very welcome.
