# Correcting Numerical Central Cusp Bias in NFW Dark Matter Halos

This repository hosts the complete computational pipeline, datasets, and high-resolution diagnostic figures for correcting the systematic central potential smoothing in Navarro-Frenk-White (NFW) dark matter mini-halos ($M \le 10^8 M_\odot$) caused by Cloud-in-Cell (CIC) density deposition on discrete Cartesian grids solved via Fast Fourier Transform (FFT) Poisson solvers [3, 4].

---

## 🌌 Scientific Context & Background

In the standard $\Lambda$CDM cosmological model, dark matter mini-halos form the primary potential wells necessary to trap primordial gas, acting as the birthplaces for Population III stars [4, 5]. Under the **Nadler et al. (2025)** criterion, a halo's capacity to form a galaxy is determined by the physical depth of its central potential [3, 4].

However, Cartesian spatial discretization acts as a low-pass filter, artificially smoothing the cuspy central density of NFW profiles ($\rho \propto r^{-1}$), shallowing the potential well by over **30%** at moderate grid resolutions [3]. This systematic numerical bias artificially extinguishes physical star formation in simulations [3, 4]. 

Our study develops a **2D spectral transfer function** binned in radius and concentration, interpolated via **Piecewise Cubic Hermite Interpolating Polynomials (PCHIP)**, which successfully restores the physical potential wells of NFW halos with **98.9%** systematic error mitigation [6].

<p align="center">
  <img src="figures/fig2_punch_gold_v2.png" width="85%" alt="Physical Stellar Recovery Scatter Plot">
  <br>
  <em>Figure 1: The "Punch" – Restoring 100% of star-forming halos from numerical erasure (recovering 27 out of 27 false darks) [6, 7].</em>
</p>

---

## 🛠️ Reproducibility & How to Run the Code

This project is built with python code engineered for full reproducibility. All physical and numerical parameters are fixed via a seed mechanism [2].

### 1. Requirements & Dependencies
The pipeline was developed and verified under **Python 3.12+** (tested up to **3.13.0**) on Linux architectures [2, 8]. It relies on standard scientific libraries:
* `numpy >= 2.0.0`
* `scipy`
* `matplotlib`

You can install all dependencies via pip:
```bash
pip install numpy scipy matplotlib
2. Execution FlowThe orchestrator in biasnfw.py triggers the complete numerical pipeline via executar_pipeline_completo()3:Calibration: Runs systematic random sub-voxel grid-phase translations to map the NFW spectral bias $\epsilon(r,c)$3lock.Corrector Interpolation: Initializes the 2D PCHIP spectral corrector3.Validation & Auditing: Performs blind cross-validation on an independent grid ($N=64$ vs $N=128$), runs the Hernquist profile transferability test, conducts $M_{\text{sub}}$ subvoxel convergence sweeps, box size sensitivity, and CPU scaling audits3.Monte Carlo Population: Generates a cosmologically physical population of 169 halos to evaluate the Nadler et al. classification threshold34.Observational Inference: Fits the galaxy occupation fraction ($f_{\text{occ}}$) sigmoid curves to compare analytical, uncorrected, and corrected populationslock5.3. Running the PipelineTo run the full simulation locally and export all data and figures, execute:python3 biasnfw.py
Note: The execution time is highly dependent on the grid resolution and subvoxel sampling. A full reference run with $N=128$ and $M_{\text{sub}}=8$ takes approximately 230 minutes of CPU on a 12-core AMD64 processor2.📊 Core Datasets (Data Audit)The pipeline automatically exports two comprehensive data audit files in the data/ folder67:resultados_mc_gold.csv: Contains 169 halos with 21 columns mapping physical properties (mass, concentration, $V_{\text{max}}$), analytical vs. numerical potentials, Nadler scores, and classification flags89. This sheet proves that PCHIP correction restores classification accuracy from 84% to 100%10.robustez_suplementar.csv: Contains raw logs for:CIC Subvoxel Convergence: Showing saturation at $M_{\text{sub}} \ge 8$ (512 subvoxels)11.Periodic Box Sensitivity: Revealing error explosion up to 82.69% due to physical resolution loss12.Reference concentration ($c_{\text{ref}}$) vulnerability: Quantifying the classification flip of a fixed physical halo ($c=8$) under different normalization choices1314.Performance Projections: Modeling the CPU execution time scaling with an empirical exponent of $p = 3.10$15.📚 Cite this WorkIf you use this spectral corrector, datasets, or methodology in your cosmological research, please cite our repository:@software{silva_ferreira_2026_biasnfw,
  author = {Silva Ferreira, Jonatas Vitorio},
  title = {Biasnfw: Correcting Numerical Central Cusp Bias in NFW Dark Matter Halos},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/vetorio/Biasnfw}}
}

