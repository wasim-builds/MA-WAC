# Multi-Agent World-Action Cloning (MA-WAC)

This repository contains the official PyTorch implementation of **MA-WAC: Stabilizing Imitation Learning via Shared Latent Proxies**.

## Overview
Standard Behavioral Cloning in multi-agent environments suffers from compounding errors and catastrophic covariate shifts. MA-WAC solves this by introducing a **Shared Latent World Proxy (SLWP)** that natively models macroscopic intents from multiple interacting agents simultaneously. Before executing actions in the true environment, an internal **Roll-back Auditor** evaluates the SLWP's output and penalizes joint macroscopic actions that lead to latent collisions.

## Repository Structure
- `data/` : Contains the authentic ETH/UCY Pedestrian Trajectory Datasets (`eth_obsmat.txt`).
- `src/` : PyTorch source code.
  - `model.py` : Defines the Perception Encoder, Macroscopic Actor, and SLWP.
  - `environment.py` : Environment wrappers.
  - `dataset_loader.py` : Parses the real ETH coordinates into tensors.
  - `simulation.py` : The main training loop with the Roll-back Auditor.
  - `plot_results.py` : Generates the empirical convergence graphs.
- `paper/` : The LaTeX source code for the manuscript (IEEEtran format).

## Quickstart

### 1. Requirements
Ensure you have Python 3.8+ and PyTorch installed.
```bash
pip install torch numpy matplotlib
```

### 2. Training the Model
To execute the end-to-end differentiable auditing loop on the ETH dataset:
```bash
python src/simulation.py
```
This will train the Macroscopic Actor and the SLWP, outputting the metrics to `data/results_variance.json`.

### 3. Generating Figures
To plot the convergence of the Roll-back rate against the Actor Loss:
```bash
python src/plot_results.py
```
The high-quality `.eps` graphs will be saved to the `paper/` directory for LaTeX compilation.

## Citation
If you use this codebase, please cite our paper:
```bibtex
@inproceedings{khan2026mawac,
  title={Multi-Agent World-Action Cloning: Stabilizing Imitation Learning via Shared Latent Proxies},
  author={Khan, Mohammed Wasim and Chenam, Venkata Bhikshapathi},
  year={2026}
}
```
