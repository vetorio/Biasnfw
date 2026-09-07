Markdown# Correcting Numerical Central Cusp Bias in NFW Dark Matter Halos

This repository hosts the complete computational pipeline, datasets, and high-resolution diagnostic figures for correcting the systematic central potential smoothing in Navarro-Frenk-White (NFW) dark matter mini-halos ($M \le 10^8 M_\odot$) caused by Cloud-in-Cell (CIC) density deposition on discrete Cartesian grids solved via Fast Fourier Transform (FFT) Poisson solvers.

---

## 🌌 Scientific Context & Background

In the standard $\Lambda$CDM cosmological model, dark matter mini-halos form the primary potential wells necessary to trap primordial gas, acting as the birthplaces for Population III stars. Under the **Nadler et al.** criterion, a halo's capacity to form a galaxy is determined by the physical depth of its central potential.

However, Cartesian spatial discretization acts as a low-pass filter, artificially smoothing the cuspy central density of NFW profiles ($\rho \propto r^{-1}$), shallowing the potential well by over **30%** at moderate grid resolutions. This systematic numerical bias artificially extinguishes physical star formation in simulations.

Our study develops a **2D spectral transfer function** binned in radius and concentration, interpolated via **Piecewise Cubic Hermite Interpolating Polynomials (PCHIP)**, which successfully restores the physical potential wells of NFW halos with **98.9%** systematic error mitigation.

<p align="center">
  <img src="figures/fig2_punch_gold_v2.png" width="85%" alt="Physical Stellar Recovery Scatter Plot">
  <br>
  <em>Figure 1: The "Punch" – Restoring 100% of star-forming halos from numerical erasure (recovering 27 out of 27 false darks).</em>
</p>

---

## ⚙️ Installation & Setup Guide

This project requires **Python 3.12+** (tested up to **3.13.0**). All external dependencies are standard scientific computing packages available via `pip`.

### Prerequisites
The following modules used in this project belong to Python's **Standard Library** and do not require installation:
* `sys`, `io`, `os`, `gc`, `csv`, `time`, `pathlib`, `warnings`, `datetime`

### Required External Dependencies
* **NumPy** (`>= 2.0.0`) - For numerical array and FFT operations.
* **SciPy** - For PCHIP interpolation (`PchipInterpolator`) and non-linear curve fitting (`curve_fit`).
* **Matplotlib** - For generating diagnostic figures and colorbar normalizations.

---

### Quick Start

**1. Clone the repository:**
```bash
git clone [https://github.com/vetorio/Biasnfw.git](https://github.com/vetorio/Biasnfw.git)
cd Biasnfw
2. Create and activate a virtual environment (Recommended):Linux / macOS:Bashpython3 -m venv venv
source venv/bin/activate
Windows (PowerShell):PowerShellpython -m venv venv
.\venv\Scripts\activate
3. Install required packages:Bashpip install "numpy>=2.0.0" scipy matplotlib
🛠️ Pipeline Architecture & Execution FlowThe main orchestrator script biasnfw.py triggers the complete numerical pipeline via executar_pipeline_completo():Calibration: Runs systematic random sub-voxel grid-phase translations to map the NFW spectral bias $\epsilon(r,c)$.Corrector Interpolation: Initializes the 2D PCHIP spectral corrector.Validation & Auditing: Performs blind cross-validation on an independent grid ($N=64$ vs $N=128$), runs the Hernquist profile transferability test, conducts $M_{\text{sub}}$ subvoxel convergence sweeps, box size sensitivity, and CPU scaling audits.Monte Carlo Population: Generates a cosmologically physical population of 169 halos to evaluate the classification threshold.Observational Inference: Fits the galaxy occupation fraction ($f_{\text{occ}}$) sigmoid curves to compare analytical, uncorrected, and corrected populations.Running the PipelineTo run the full simulation locally and export all data and diagnostic figures, execute:Bashpython3 biasnfw.py
Note: Execution time depends heavily on grid resolution and subvoxel sampling. A full reference run with $N=128$ and $M_{\text{sub}}=8$ takes approximately 230 minutes of CPU time on a 12-core AMD64 processor.📊 Core Datasets (Data Audit)The pipeline automatically exports two comprehensive data audit files in the data/ directory:data/resultados_mc_gold.csv: Contains 169 halos with 21 columns mapping physical properties (mass, concentration, $V_{\text{max}}$), analytical vs. numerical potentials, Nadler scores, and classification flags. This dataset proves that PCHIP correction restores classification accuracy from 84% to 100%.data/robustez_suplementar.csv: Contains raw logs for:CIC Subvoxel Convergence: Showing saturation at $M_{\text{sub}} \ge 8$ (512 subvoxels).Periodic Box Sensitivity: Revealing error explosion up to 82.69% due to physical resolution loss.Reference Concentration ($c_{\text{ref}}$) Vulnerability: Quantifying the classification flip of a fixed physical halo ($c=8$) under different normalization choices.Performance Projections: Modeling the CPU execution time scaling with an empirical exponent of $p = 3.10$.📚 Cite This WorkIf you use this spectral corrector, dataset, or methodology in your cosmological research, please cite this repository:Snippet de código@software{silva_ferreira_2026_biasnfw,
  author = {Silva Ferreira, Jonatas Vitorio},
  title = {Biasnfw: Correcting Numerical Central Cusp Bias in NFW Dark Matter Halos},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{[https://github.com/vetorio/Biasnfw](https://github.com/vetorio/Biasnfw)}}
}
