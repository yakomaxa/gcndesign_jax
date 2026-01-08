# GCNdesign_jax

GCNdesign in **JAX** (WIP).

The original **GCNdesign** was written in **PyTorch**.  
This repository is a re-implementation of the algorithm in **JAX**, with trained parameters ported from Torch into JAX.

Porting GCNdesign into JAX makes it possible to train **conjugate models written in JAX**, such as **AlphaFold** and **ProteinMPNN in JAX**, enabling tighter integration and end-to-end optimization.

## Original PyTorch version
https://github.com/ShintaroMinami/GCNdesign

## Contents
The root directory currently contains a mixture of scripts used during the porting process and is not yet well organized. This will be cleaned up as the JAX implementation stabilizes.

## License
MIT License, following the original version.  
See the `LICENSE` files in this repository for details.
