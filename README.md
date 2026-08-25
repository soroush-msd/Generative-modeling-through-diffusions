# Generative Modeling Through Diffusions

> **Master's Thesis** :
Master's Degree in Statistics, Universidad Carlos III de Madrid (UC3M), 2024–2025

**Author:** Seyed Amirhossein Mosaddad  
**Supervisor:** Eduardo García Portugués  

[![Thesis](https://img.shields.io/badge/Thesis-UC3M_e--Archivo-8A2BE2?style=for-the-badge)](https://hdl.handle.net/10016/48727)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

## Overview

This repository contains all the code, experiments, and animations accompanying the master's thesis *"Generative Modeling Through Diffusions"*. The thesis provides a step-by-step introduction to score-based diffusion generative models, covering the theory and practical implementation of:

- **Score functions** and **score matching** (explicit and denoising)
- **Langevin dynamics** for sampling
- **Neural networks** (MLPs and U-Nets) for learning score functions
- **Noise-conditional score networks (NCSN)**
- **Stochastic differential equations (SDEs)** for generative modeling
- **Real data generation** from the Quick, Draw! dataset

<p align="center">
  <img src="./assets/slides_page_0004.jpg" alt="Overview" width="800">
</p>

<p align="center">
  <em>Figure 1. Overview of the theoretical and methodological framework of the thesis.</em>
</p>

## Repository Structure

```
├── Example 1 - Fisher div 1/         # Score matching for N(0, θ), analytical & empirical
├── Example 2 - Fisher div 2/         # Score matching for N(μ, σ²), 2D surface optimization
├── Example 3 - Langevin diffusion/   # Langevin dynamics on a 2D Gaussian mixture
├── Example 4 - Naive model/          # Naive score matching on an oval distribution (MLP)
├── Example 5 - NCSN/                 # Denoising score matching with noise-conditional score network
├── Experiment - QuickDraw/           # U-Net + VE-SDE on Quick, Draw! images (tree, car, octopus)
├── slides/                           # Thesis defense presentation slides
├── assets/                           # Assets
└── README.md
```

## Examples & Experiments

### Example 1: Fisher Divergence (1D Variance)

Demonstrates that the Fisher divergence $L(\theta)$ and the score matching objective $J(\theta)$ differ only by a constant, confirming Hyvärinen (2005). The empirical estimator $\hat{J}(\theta)$ converges to the true variance as sample size grows.

<div align="center">
  <img src="./assets/example1/j_hat_sampling_variability_N10.jpg" width="300">
  <img src="./assets/example1/j_hat_sampling_variability_N50.jpg" width="300">
  <br>
  <img src="./assets/example1/j_hat_sampling_variability_N100.jpg" width="300">
  <img src="./assets/example1/j_hat_sampling_variability_N500.jpg" width="300">
</div>

<p align="center">
  <em>Figure 2. Behavior of the objective functions and their sample-based estimates across different sample sizes.</em>
</p>


**Thesis reference:** Example 1, Figure 2.1

---

### Example 2: Fisher Divergence (Mean and Variance)

Extends Example 1 to jointly estimate both the mean and variance of a normal distribution. 3D surface plots of $\hat{J}(\mu, \sigma^2)$ show convergence to the true parameters.

<div align="center">
  <img src="./assets/example2/loss_surface_Jhat_N10.jpg" width="300">
  <img src="./assets/example2/loss_surface_Jhat_N50.jpg" width="300">
  <br>
  <img src="./assets/example2/loss_surface_Jhat_N100.jpg" width="300">
  <img src="./assets/example2/loss_surface_Jhat_N500.jpg" width="300">
</div>

<p align="center">
  <em>Figure 3. Sample-based estimator surfaces for joint estimation of the mean and variance across increasing sample sizes.</em>
</p>

**Thesis reference:** Example 2, Figure 2.2

---

### Example 3: Langevin Dynamics

Visualizes how Langevin dynamics generates samples from a 2D mixture of three Gaussians using the exact score function. 500 uniformly initialized points converge to the mixture modes.


<p align="center">
  <img src="./assets/example3/langevin_mixture.gif" width="400" alt="Langevin dynamics sampling from a mixture distribution">
</p>

<p align="center">
  <em>Figure 4. Langevin dynamics sampling from a mixture distribution.</em>
</p>

**Thesis reference:** Example 3, Figure 2.3

---

### Example 4: Naive Score Model

Trains an MLP (6 hidden layers, 1024 units, softplus activation) to learn the score function of an oval-shaped distribution via **explicit score matching**. Demonstrates that score estimates are accurate in high-density regions but unreliable in low-density areas.

<div align="center">
  <img src="./assets/example4/angular_error_heatmap.jpg" width="327">
  <img src="./assets/example4/error_magnitude_heatmap.jpg" width="314">
</div>

<p align="center">
  <em>
    Figure 5. Error analysis of the learned score field: angular error (left) and magnitude error (right), with contours indicating the underlying data density.
  </em>
</p>

<div align="center">
  <img src="./assets/example4/langevin_high_density_ring.gif" width="275">
  <img src="./assets/example4/langevin_low_density_center.gif" width="275">
  <img src="./assets/example4/langevin_outer_low_density.gif" width="275">
</div>

<p align="center">
  <em>
    Figure 6. Langevin Markov chain trajectories from different initializations over 200 steps, indicating stable dynamics in high-density areas and unstable behavior in low-density regions.
  </em>
</p>

**Thesis reference:** Example 4, Figures 3.1–3.4

---

### Example 5: Noise-Conditional Score Network (NCSN)

Uses **denoising score matching** with multiple noise levels to address the limitations of the naive approach. Trains the same MLP architecture with a geometric noise schedule ($L = 20$ levels). Generates samples via **annealed Langevin dynamics**.

<div align="center">
  <img src="./assets/example5/ncsn_error_magnitude_high_noise.jpg" width="287">
  <img src="./assets/example5/ncsn_error_magnitude_medium_noise.jpg" width="275">
  <img src="./assets/example5/ncsn_error_magnitude_low_noise.jpg" width="281">
  <br>
  <img src="./assets/example5/ncsn_angular_error_high_noise.jpg" width="282">
  <img src="./assets/example5/ncsn_angular_error_medium_noise.jpg" width="282">
  <img src="./assets/example5/ncsn_angular_error_low_noise.jpg" width="282">
</div>

<p align="center">
  <em>
    Figure 7. Spatial error analysis of the NCSN model across different noise levels.
    The top row shows magnitude error between the true and learned score fields for
    high (left), medium (middle), and low (right) noise levels. The bottom row shows
    the corresponding angular error heatmaps.
  </em>
</p>

<p align="center">
 <img src="./assets/example5/ncsn_perturbed_density_animation.gif" width="400" alt="Perturbed data densities at different noise levels">
</p>

<p align="center">
  <em>Figure 8. Perturbation of the data density across noise levels. Higher noise smooths and spreads the density; structure re-emerges as noise decreases.</em>
</p>

<p align="center">
 <img src="./assets/example5/ncsn_sample_evolution_animation.gif" width="400" alt="Sample distribution evolution during denoising">
</p>

<p align="center">
  <em>Figure 9. Sample distribution evolving during annealed Langevin denoising. Samples gradually recover the true data density as noise is removed.</em>
</p>

**Thesis reference:** Example 5, Figures 3.5–3.9

---

### Experiment: Quick, Draw! (U-Net + VE-SDE)

Trains a U-Net on 28×28 grayscale sketches from the [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset (trees, cars, octopi). Uses a **Variance-Exploding SDE** for the forward process and generates new images by simulating the reverse SDE with Euler–Maruyama discretization.

| With stochastic noise | Without stochastic noise |
|:---:|:---:|
| Diverse, varied samples | Collapsed to modal shapes |

<div align="center">
  <img src="./assets/experiment/tree/tree1.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/tree/tree_nonoise1.jpg" width="35%"><br>
  <img src="./assets/experiment/tree/tree2.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/tree/tree_nonoise2.jpg" width="35%"><br>
  <img src="./assets/experiment/tree/tree3.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/tree/tree_nonoise3.jpg" width="35%"><br>
  <img src="./assets/experiment/tree/tree4.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/tree/tree_nonoise4.jpg" width="35%"><br>
  <img src="./assets/experiment/tree/tree5.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/tree/tree_nonoise5.jpg" width="35%">
</div>

<br>

<div align="center">
  <img src="./assets/experiment/car/car1.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/car/car_nonoise1.jpg" width="35%"><br>
  <img src="./assets/experiment/car/car2.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/car/car_nonoise2.jpg" width="35%"><br>
  <img src="./assets/experiment/car/car3.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/car/car_nonoise3.jpg" width="35%"><br>
  <img src="./assets/experiment/car/car4.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/car/car_nonoise4.jpg" width="35%"><br>
  <img src="./assets/experiment/car/car5.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/car/car_nonoise5.jpg" width="35%">
</div>

<br>

<div align="center">
  <img src="./assets/experiment/octopus/octo1.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/octopus/octo_nonoise1.jpg" width="35%"><br>
  <img src="./assets/experiment/octopus/octo2.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/octopus/octo_nonoise2.jpg" width="35%"><br>
  <img src="./assets/experiment/octopus/octo3.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/octopus/octo_nonoise3.jpg" width="35%"><br>
  <img src="./assets/experiment/octopus/octo4.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/octopus/octo_nonoise4.jpg" width="35%"><br>
  <img src="./assets/experiment/octopus/octo5.jpg" width="35%">&nbsp;&nbsp;&nbsp;<img src="./assets/experiment/octopus/octo_nonoise5.jpg" width="35%">
</div>

<p align="center">
  <em>Figure 10. Denoising process across the tree, car, and octopus datasets. Left: sampling with added Gaussian noise; right: without. Each row progresses from pure noise to a clean sample over 1,000 iterations (rows are independent samples).</em>
</p>

<div align="center">
  <img src="./assets/experiment/real-samples/tree_dataset_samples.jpg" width="300">
  <img src="./assets/experiment/real-samples/car_dataset_samples.jpg" width="300">
  <img src="./assets/experiment/real-samples/octopus_dataset_samples.jpg" width="300">
</div>

<p align="center">
  <em>Figure 11. Real samples from each dataset. Left: tree, Middle: car, Right: octopus.</em>
</p>

**Thesis reference:** Chapter 4, Figures 4.1–4.5


## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
- Google Colab (recommended for the QuickDraw experiment, trained on A100 GPU)

### Installation

```bash
git clone https://github.com/soroush-msd/Generative-modeling-through-diffusions.git
cd Generative-modeling-through-diffusions
pip install torch numpy scipy matplotlib
```

> **NOTE** :
It is best to use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to create an isolated environment for installing dependencies.

### Running the Examples

Examples 1-5 are Python scripts. Run them directly from the project root:

```bash
python "Example 1 - Fisher div 1/fisher_div_1.py"
python "Example 2 - Fisher div 2/fisher_div_2.py"
python "Example 3 - Langevin diffusion/LD.py"
python "Example 4 - Naive model/naive.py"
python "Example 5 - NCSN/ncsn.py"
```

### Running the QuickDraw Experiment

The QuickDraw experiment is a Jupyter Notebook and is recommended to run in [Google Colab](https://colab.research.google.com/) with GPU acceleration enabled (trained on an A100 GPU).

### Pretrained Checkpoints

Each dataset has a saved model checkpoint (a plain `state_dict`) so you can generate samples without retraining:

| Dataset | Checkpoint |
|:---|:---|
| Tree | `Experiment - QuickDraw/tree/ckpt_quickdraw_tree.pth` |
| Car | `Experiment - QuickDraw/car/ckpt_quickdraw_car.pth` |
| Octopus | `Experiment - QuickDraw/octopus/ckpt_quickdraw_octopus_new.pth` |

The model is wrapped in `torch.nn.DataParallel` during training, so rebuild it the same way before loading the weights:

```python
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn)).to(device)

ckpt = torch.load('ckpt_quickdraw_tree.pth', map_location=device)
score_model.load_state_dict(ckpt)
score_model.eval()  # ready for sampling
```

Then run the Euler–Maruyama sampler to generate new sketches. Per-dataset training loss curves (`training_loss_*.pdf`) are included as well.

## Thesis & Slides

- **Full thesis:** [*Generative Modeling Through Diffusions*](https://hdl.handle.net/10016/48727) - UC3M e-Archivo (open access)
- **Defense slides:** [`slides.pdf`](./slides/slides.pdf)


## Citation

If you find this work useful, please consider citing:

```bibtex
@mastersthesis{mosaddad2025diffusions,
  title   = {Generative Modeling Through Diffusions},
  author  = {Mosaddad, Seyed Amirhossein},
  school  = {Universidad Carlos III de Madrid},
  year    = {2025},
  month   = {September},
  type    = {Master's Thesis},
  note    = {Master's Degree in Statistics for Data Science},
  url     = {https://hdl.handle.net/10016/48727}
}
```


## Key References

- Hyvärinen, A. (2005). *Estimation of non-normalized statistical models by score matching.* JMLR.
- Song, Y. & Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution.* NeurIPS.
- Song, Y. et al. (2020). *Score-based generative modeling through stochastic differential equations.* arXiv.
- Vincent, P. (2011). *A connection between score matching and denoising autoencoders.* Neural Computation.
- Song, Y. (2021). [Generative modeling by estimating gradients of the data distribution](https://yang-song.net/blog/2021/score/) (blog post).


## License

This work is licensed under [Creative Commons Attribution–NonCommercial–NoDerivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).