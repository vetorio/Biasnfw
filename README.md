# Biasnfw: Correcting Numerical Central Cusp Bias in NFW Dark Matter Halos

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Astrophysics%20%26%20Cosmology-purple.svg)](#-scientific-context--background)

This repository hosts the complete computational pipeline, datasets, and high-resolution diagnostic figures for correcting systematic central potential smoothing in Navarro-Frenk-White (NFW) dark matter mini-halos ($M \le 10^8 M_\odot$) caused by Cloud-in-Cell (CIC) density deposition on discrete Cartesian grids solved via Fast Fourier Transform (FFT) Poisson solvers.

---

## 🌌 Scientific Context & Background

In the standard $\Lambda$CDM cosmological model, dark matter mini-halos form the primary potential wells necessary to trap primordial gas, acting as the birthplaces for Population III stars. Under the **Nadler et al.** criterion, a halo's capacity to form a galaxy is determined by the physical depth of its central potential.

However, Cartesian spatial discretization acts as a low-pass filter, artificially smoothing the cuspy central density of NFW profiles ($\rho \propto r^{-1}$) and shallowing the potential well by over **30%** at moderate grid resolutions. This systematic numerical bias artificially extinguishes physical star formation in simulations.

Our study develops a **2D spectral transfer function** binned in radius and concentration, interpolated via **Piecewise Cubic Hermite Interpolating Polynomials (PCHIP)**, which successfully restores the physical potential wells of NFW halos with **98.9%** systematic error mitigation.

<p align="center">
  <img src="figures/fig2_punch_gold_v2.png" width="85%" alt="Physical Stellar Recovery Scatter Plot">
  <br>
  <em>Figure 1: Restoring 100% of star-forming halos from numerical erasure (recovering 27 out of 27 false darks).</em>
</p>

---

## ⚙️ Requirements & Installation

This project requires **Python 3.12+** (tested through **3.13.0**). Standard Library dependencies (`sys`, `io`, `os`, `gc`, `csv`, `time`, `pathlib`, `warnings`, `datetime`) are built-in.

### External Dependencies

| Package | Minimum Version | Purpose |
| :--- | :--- | :--- |
| **NumPy** | `>= 2.0.0` | High-performance array structures and 3D FFT calculations |
| **SciPy** | `>= 1.10.0` | PCHIP 2D interpolation (`PchipInterpolator`) & curve fitting |
| **Matplotlib** | `>= 3.7.0` | Diagnostic visual generation & colorbar normalizations |

### Quick Start

```bash
# 1. Clone the repository
git clone [https://github.com/vetorio/Biasnfw.git](https://github.com/vetorio/Biasnfw.git)
cd Biasnfw

# 2. Create and activate a virtual environment
# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\activate

# 3. Install core dependencies
pip install "numpy>=2.0.0" scipy matplotlib
```

---

## 🛠️ Pipeline Architecture & Execution Flow

The core orchestrator script `biasnfw.py` triggers the complete numerical pipeline via `executar_pipeline_completo()` across six automated stages:

* **Calibration:** Maps the spectral bias $\epsilon(r,c)$ using systematic random sub-voxel grid-phase translations.
* **Corrector Interpolation:** Initializes the 2D PCHIP spectral transfer function matrix.
* **Validation & Auditing:** Performs blind cross-validation ($N=64$ vs $N=128$), Hernquist transferability testing, sub-voxel convergence sweeps ($M_{\text{sub}}$), box-size sensitivity, and CPU scaling audits.
* **Monte Carlo Population:** Generates a physical population of 169 halos to evaluate classification thresholds.
* **Observational Inference:** Fits galaxy occupation fraction ($f_{\text{occ}}$) sigmoids comparing analytical, uncorrected, and corrected states.

### Running the Full Simulation

Execute the orchestrator script to run the pipeline and output figures and audit tables:

```bash
python3 biasnfw.py
```

> **Execution Note:** CPU time depends on grid resolution and sub-voxel sampling. A reference run ($N=128$, $M_{\text{sub}}=8$) takes ~230 CPU minutes on a 12-core AMD64 processor.

---

## 📊 Core Datasets (Data Audit)

All pipeline executions automatically output structured logs and processed data to the `data/` directory:

| Dataset File | Primary Contents | Key Outcome / Proof |
| :--- | :--- | :--- |
| `data/resultados_mc_gold.csv` | 169 halos across 21 columns (mass, concentration, $V_{\text{max}}$, analytical vs. numerical potentials, Nadler scores) | Demonstrates PCHIP correction restores halo classification accuracy from **84% to 100%**. |
| `data/robustez_suplementar.csv` | Raw logs covering sub-voxel sweeps, periodic box shifts, reference concentration ($c_{\text{ref}}$) sensitivity, and timing benchmarks | Confirms saturation at $M_{\text{sub}} \ge 8$ and quantifies runtime scaling ($p = 3.10$). |

---

## 📚 Citation

If you utilize this spectral corrector, methodology, or generated dataset in your research, please cite this work as follows:

```bibtex
@software{silva_ferreira_2026_biasnfw,
  author       = {Silva Ferreira, Jonatas Vitorio},
  title        = {Biasnfw: Correcting Numerical Central Cusp Bias in NFW Dark Matter Halos},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{[https://github.com/vetorio/Biasnfw](https://github.com/vetorio/Biasnfw)}}
}
```
