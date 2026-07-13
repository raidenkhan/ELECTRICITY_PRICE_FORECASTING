# Electricity Price Forecasting: Day-Ahead Market Modeling with Deep Probabilistic Methods

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

> **Master's Thesis / Research Project** — *Electricity Price Forecasting for Day-Ahead Markets using Deep Probabilistic Forecasting, Gaussian Process Neural Random Features, and Mixture-of-Experts Architectures*

---

## 📖 Overview

This repository contains the complete implementation of a **day-ahead electricity price forecasting (EPF)** system developed for a Master's thesis in Energy Economics / Data Science. The project implements state-of-the-art deep probabilistic forecasting methods tailored to the unique characteristics of electricity markets: **non-stationarity, regime-switching behavior, extreme price spikes, negative prices, and complex seasonalities**.

### Key Contributions

| Component | Description |
|-----------|-------------|
| **Mixture-of-Experts TFT** | Temporal Fusion Transformer with regime-aware mixture heads for regime-switching price dynamics |
| **Gaussian Process Neural Random Features (GP-NRF)** | Scalable GP inference via random Fourier features with neural spectral learning |
| **Latent Force GP-NRF** | Physics-informed GP with latent force models capturing exogenous drivers (load, renewables, fuel prices) |
| **Regime-Switching LEA-R** | Local Expert Attention with regime-switching for extreme price regime detection |
| **Mixture Volatility TFT** | Heteroscedastic volatility modeling via mixture density networks |
| **Curriculum Learning** | Progressive training from normal to extreme regimes for robust spike forecasting |
| **Comprehensive Benchmarking** | EPFtoolbox benchmarks, statistical significance testing (Diebold-Mariano), and probabilistic calibration |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ELECTRICITY PRICE FORECASTING PIPELINE                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │   EPF Data   │───▶│  Feature Eng.    │───▶│   Multi-Model Training   │   │
│  │  (EPFtoolbox)│    │  (Calendars,     │    │  ┌────────────────────┐  │   │
│  │  + Custom    │    │   Lags, Fourier, │    │  │  Mixture TFT       │  │   │
│  │  Exogenous)  │    │   Interactions)  │    │  │  GP-NRF / LF-GP    │  │   │
│  └──────────────┘    └──────────────────┘    │  │  Regime LEA-R      │  │   │
│                                                │  │  MixVol TFT        │  │   │
│                                                │  │  Curriculum TFT    │  │   │
│                                                │  └────────────────────┘  │   │
│                                                └───────────┬──────────────┘   │
│                                                             │                │
│                                                ┌────────────▼──────────────┐   │
│                                                │   Ensemble & Evaluation   │   │
│                                                │  • Probabilistic Scores   │   │
│                                                │  • Diebold-Mariano Tests  │   │
│                                                │  • PIT / Calibration      │   │
│                                                │  • Spike/Regime Metrics   │   │
│                                                └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
epf-day-ahead/
├── configs/                    # YAML configuration files
│   ├── base_config.yaml
│   ├── mixture_tft.yaml
│   ├── gp_nrf.yaml
│   ├── regime_lear.yaml
│   └── curriculum.yaml
├── data/
│   ├── raw/                    # Raw EPFtoolbox data (not tracked)
│   ├── processed/              # Processed features (not tracked)
│   └── external/               # External data sources (fuel, weather)
├── models/
│   ├── __init__.py
│   ├── mixture_model.py        # Mixture density networks
│   ├── mixvol_tft.py           # Mixture Volatility TFT
│   ├── gp_nrf/
│   │   ├── model.py            # GP-NRF base model
│   │   ├── force_kernel.py     # Latent force kernels
│   │   ├── latent_sde.py       # Latent SDE formulation
│   │   └── force_gp.py         # Latent Force GP
│   ├── regime_lear.py          # Regime-switching LEA-R
│   ├── mrs_lear_ensemble.py    # Mixture Regime-Switching Ensemble
│   └── lf_gp_nrf/
│       └── model.py            # Latent Force GP-NRF
├── training/
│   ├── train.py                # Main training script
│   ├── train_cnn_bilstm.py     # Baseline CNN-BiLSTM training
│   ├── curriculum.py           # Curriculum learning scheduler
│   └── losses.py               # CRPS, Quantile, Mixture NLL losses
├── evaluation/
│   ├── evaluate.py             # Comprehensive evaluation
│   ├── metrics.py              # CRPS, Winkler, PIT, DM tests
│   ├── backtest.py             # Rolling window backtesting
│   └── visualization.py        # Calibration plots, spike analysis
├── utils/
│   ├── data_loader.py          # EPFtoolbox wrapper
│   ├── features.py             # Feature engineering pipeline
│   ├── preprocessing.py        # Scaling, outlier handling
│   └── helpers.py              # Utilities, seeding, logging
├── experiments/
│   ├── run_benchmarks.py       # EPFtoolbox benchmark runner
│   ├── ablation_studies.py     # Ablation experiments
│   └── hyperparameter_tuning.py # Optuna-based HPO
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_spike_analysis.ipynb
│   └── 04_thesis_figures.ipynb
├── tests/
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_data_pipeline.py
├── scripts/
│   ├── download_data.py
│   ├── run_all_experiments.sh
│   └── generate_tables.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING.git
cd ELECTRICITY_PRICE_FORECASTING

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install EPFtoolbox (for benchmark data)
pip install epftoolbox
```

### Data Preparation

```bash
# Download EPFtoolbox benchmark data (NP, PJM, BE, DE, FR, etc.)
python scripts/download_data.py --markets NP PJM BE DE FR --years 2014-2024

# Generate features (calendars, lags, Fourier terms, exogenous)
python -m utils.features --config configs/base_config.yaml
```

### Training

```bash
# Train Mixture TFT (main model)
python training/train.py --config configs/mixture_tft.yaml --gpu 0

# Train GP-NRF with Latent Forces
python training/train.py --config configs/gp_nrf.yaml --gpu 0

# Train with Curriculum Learning
python training/train.py --config configs/curriculum.yaml --gpu 0

# Run all experiments (reproduces thesis results)
bash scripts/run_all_experiments.sh
```

### Evaluation

```bash
# Comprehensive evaluation with statistical tests
python evaluation/evaluate.py \
    --model-path outputs/mixture_tft/best_model.pt \
    --config configs/mixture_tft.yaml \
    --test-market NP \
    --metrics crps,winkler,pit,dm_test,spike_metrics

# Generate thesis tables and figures
python scripts/generate_tables.py --results-dir outputs/
python notebooks/04_thesis_figures.ipynb
```

---

## 📊 Models Implemented

### 1. Mixture-of-Experts Temporal Fusion Transformer (`mixture_model.py`, `mixvol_tft.py`)

- **Architecture**: TFT backbone with regime-specific expert heads
- **Mixture Density Network**: Gaussian Mixture Model output for multi-modal price distributions
- **Regime Gating**: Learnable attention over regimes (normal, spike, negative, high-volatility)
- **Volatility Modeling**: Separate mixture components for conditional heteroscedasticity

```python
from models.mixture_model import MixtureTFT

model = MixtureTFT(
    input_size=128,
    hidden_size=256,
    num_heads=8,
    num_experts=4,           # Normal, Spike, Negative, High-Vol
    mixture_components=3,    # GMM components per expert
    dropout=0.1
)
```

### 2. Gaussian Process Neural Random Features (`models/gp_nrf/`)

- **Random Fourier Features**: Scalable GP approximation with learnable spectral densities
- **Neural Spectral Learning**: MLP parameterizing the spectral measure
- **Latent Force Model**: Physics-informed GP with ODE-constrained latent forces
- **Variational Inference**: ELBO optimization with reparameterization

```python
from models.gp_nrf.model import GPNRF
from models.gp_nrf.force_gp import LatentForceGPNRF

# Base GP-NRF
model = GPNRF(
    input_dim=24,
    num_features=1024,
    spectral_net_hidden=[64, 64],
    kernel_type='spectral_mixture'
)

# Latent Force GP-NRF (with exogenous drivers)
lf_model = LatentForceGPNRF(
    input_dim=24,
    force_dim=5,  # load, wind, solar, gas, coal
    ode_solver='rk4'
)
```

### 3. Regime-Switching LEA-R (`regime_lear.py`, `mrs_lear_ensemble.py`)

- **Local Expert Attention**: Mixture of specialized attention heads
- **Markov Regime Switching**: Hidden Markov Model over price regimes
- **Ensemble Aggregation**: Bayesian model averaging over regime paths

### 4. Curriculum Learning (`training/curriculum.py`)

- **Difficulty Scheduling**: Progressive exposure to extreme regimes
- **Regime-Aware Sampling**: Stratified mini-batches by price regime
- **Loss Reweighting**: Focal loss for rare spike events

---

## ⚙️ Configuration

All experiments configured via YAML files in `configs/`:

```yaml
# configs/mixture_tft.yaml
model:
  type: "MixtureTFT"
  input_size: 128
  hidden_size: 256
  num_heads: 8
  num_experts: 4
  mixture_components: 3
  dropout: 0.1
  horizon: 24
  quantiles: [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]

data:
  market: "NP"
  train_years: [2014, 2015, 2016, 2017, 2018, 2019]
  val_years: [2020, 2021]
  test_years: [2022, 2023]
  features:
    - calendar
    - lags: [1, 2, 3, 7, 24, 168]
    - fourier: [daily, weekly, yearly]
    - exogenous: [load, wind, solar, gas, coal, co2]

training:
  batch_size: 64
  learning_rate: 1e-3
  weight_decay: 1e-5
  max_epochs: 200
  early_stopping_patience: 20
  gradient_clip_val: 1.0
  curriculum:
    enabled: true
    warmup_epochs: 20
    regime_schedule: "progressive"

loss:
  type: "MixtureNLL"
  crps_weight: 0.5
  quantile_weight: 0.3
  mixture_weight: 0.2
```

---

## 📈 Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Point Forecast** | MAE, RMSE, MAPE, sMAPE |
| **Probabilistic** | CRPS, Quantile Score, Winkler Score, Interval Score |
| **Calibration** | PIT Histogram, Reliability Diagram, Sharpness |
| **Statistical Tests** | Diebold-Mariano (DM), Giacomini-White, Model Confidence Set |
| **Electricity-Specific** | Spike MAE (price > 95th pctl), Negative Price MAE, Ramp Score, Peak Timing Error |

---

## 📊 Reproducing Thesis Results

### Main Results (Table 1 in thesis)

```bash
# Run all benchmark models on all markets
python experiments/run_benchmarks.py \
    --markets NP PJM BE DE FR \
    --models MixtureTFT GPNRF RegimeLEAR MixVolTFT CurriculumTFT \
    --baselines Naive SeasonalARIMA LEAR TFT QuantileRF \
    --output results/benchmark_results.csv
```

### Ablation Studies (Table 2)

```bash
python experiments/ablation_studies.py \
    --base-model MixtureTFT \
    --ablations no_mixture no_regime_gating no_curriculum no_exogenous \
    --output results/ablation_results.csv
```

### Statistical Significance (Table 3)

```bash
python evaluation/metrics.py \
    --results-dir results/ \
    --test dm_test \
    --alpha 0.05 \
    --correction bonferroni \
    --output results/dm_test_results.csv
```

---

## 📝 Thesis Chapter Mapping

| Chapter | Notebook / Script | Description |
|---------|-------------------|-------------|
| 2. Literature Review | `notebooks/01_literature_review.ipynb` | EPF taxonomy, probabilistic forecasting, GP literature |
| 3. Methodology | `notebooks/02_methodology_details.ipynb` | Mathematical derivations, model diagrams |
| 4. Data & Features | `notebooks/01_data_exploration.ipynb` | EPFtoolbox, feature importance, regime identification |
| 5. Experiments | `notebooks/02_model_comparison.ipynb` | Benchmark results, ablation, hyperparameter sensitivity |
| 6. Spike Analysis | `notebooks/03_spike_analysis.ipynb` | Extreme event forecasting, tail calibration |
| 7. Conclusion | `notebooks/04_thesis_figures.ipynb` | Publication-ready figures and tables |

---

## 🔬 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{khanelectricity2024,
  title     = {Electricity Price Forecasting for Day-Ahead Markets using Deep Probabilistic Methods},
  author    = {Khan, Raiden},
  school    = {University Name},
  year      = {2024},
  type      = {Master's Thesis},
  note      = {Available at: https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **EPFtoolbox** — [N. Espinosa et al.](https://github.com/jeslago/epftoolbox) for benchmark data and baseline implementations
- **PyTorch Forecasting** — TFT implementation reference
- **GPyTorch** — GP-NRF spectral kernel implementations
- **Thesis Supervisor** — Prof. [Name], [Department], [University]

---

## 📧 Contact

**Raiden Khan** — raiden.khan@university.edu

Project Link: [https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING](https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING)