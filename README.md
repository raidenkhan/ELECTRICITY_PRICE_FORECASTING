# Hierarchical Regime-Aware LEAR (HR-LEAR): Resolving the Neural MoE Collapse in Electricity Price Forecasting

[![Paper](https://img.shields.io/badge/Paper-AIMS%20Journal-0066cc.svg)](https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen.svg)]()

> **AIMS Journal Submission** — *Research Article*  
> **Author:** Kamal (Kwame Nkrumah University of Science and Technology)  
> **Market:** German Day-Ahead (EPEX SPOT DE) — 2015–2026 (39,216 samples)  
> **Key Result:** HR-LEAR achieves **CRPS 7.95 EUR/MWh** (15.3% improvement over LEAR), **12.5% Energy Score reduction**, **5.25% BESS profit uplift**

---

## 📖 Abstract

The energy transition has introduced unprecedented non-linearity and regime-dependency into electricity price forecasting (EPF), necessitating models that navigate between surplus renewable generation and thermal scarcity. While recent literature favors high-capacity neural Mixture-of-Experts (MoE) for this task, we document a **"Neural Paradox"** in the German day-ahead market: increasing architectural complexity often triggers **"expert collapse"**, where neural routers converge to a single dominant regime, leaving extreme states poorly modeled.

To resolve this, we propose the **Hierarchical Regime-Aware LEAR (HR-LEAR)** model. Unlike flat MoE structures, HR-LEAR employs a **"Global-to-Local" hierarchy**: a **Level 0 Global Anchor** (ElasticNet ensemble) ensures baseline stability across 39,216 samples, while **Level 1 Residual Specialists** target regime-specific errors identified by a **Hidden Markov Model (HMM)**. A **Stability-Preserving Indicator (SPI)** prevents over-correction in transition zones.

**Results on German market (2015–2026):**

| Metric | Global LEAR | HR-LEAR (Proposed) | Improvement |
|--------|-------------|-------------------|-------------|
| **CRPS (EUR/MWh)** | 9.38 | **7.95** | **−15.3%** |
| **MAE (EUR/MWh)** | 11.72 | **9.92** | −15.4% |
| **Energy Score (24h)** | 63.59 | **55.66** | **−12.5%** |
| **BESS Arbitrage Profit** | €24,082 | **€25,346** | **+5.25%** |
| **Spike-Regime MAE** | 39.44 | **32.18** | −18.4% |

**Advanced Ablation Champion — OT-HR-LEAR** (Optimal Transport): **CRPS 7.39, MAE 9.22** (−7.1% vs HR-LEAR)

---

## 🏗️ Architecture: Global-to-Local Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL REGIME-AWARE LEAR (HR-LEAR)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT: Price Lags (1,2,3,7,24,168) + Calendars + Exogenous (Load, RES)   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ LEVEL 0: GLOBAL ANCHOR (ElasticNet Ensemble)                        │   │
│   │ • Trained on ALL 39,216 samples                                     │   │
│   │ • Robust baseline across Surplus / Base / Spike regimes             │   │
│   │ • Output: ŷ_base ∈ ℝ²⁴                                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ REGIME IDENTIFICATION (HMM: 3 States — Surplus, Base, Spike)        │   │
│   │ • Hidden Markov Model on residuals + exogenous features             │   │
│   │ • Provides posterior γ_t = [γ₀ₜ, γ₁ₜ, γ₂ₜ]                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ LEVEL 1: RESIDUAL SPECIALISTS (ElasticNet per regime)               │   │
│   │ • Target: ε_k = y − ŷ_base (residuals in each regime)               │   │
│   │ • Trained on regime-filtered data (γ_kt > τ)                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STABILITY-PRESERVING INDICATOR (SPI) — Core Innovation              │   │
│   │ Φ(γ, τ) = 𝟙(max(γ) > τ)  with τ = 0.6                               │   │
│   │ • "Locks" specialists when HMM confidence ≤ 0.6                     │   │
│   │ • Prevents High-Confidence Jitter in transition zones               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   OUTPUT: ŷ_HR = ŷ_base + Φ(γ, τ) · Σ γ_kt · δ_kt                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Mathematical Formulation:**

Standard MRS-LEAR (unstable):
```
ŷ_std = Σ γ_kt · f_k(X_t)
```

HR-LEAR (stable):
```
ŷ_HR = ŷ_global,t + Φ(γ_t, τ) · Σ γ_kt · δ_kt(X_t)

where:
  Φ(γ_t, τ) = 𝟙(max(γ_t) > τ)    (SPI gate)
  τ = 0.6                         (empirically optimal threshold)
```

---

## 🔬 Key Innovations

### 1. Neural MoE Collapse Diagnosis
Documented failure of CNN-BiLSTM-AR and MixVol MoE on German market:
- **CRPS 24.02** (MixVol MoE) vs **9.38** (Global LEAR)
- Neural routers collapse to "winner-take-all" in sparse spike regimes (<2% of samples)
- Deep models over-smooth extreme price movements

### 2. Stability-Preserving Indicator (SPI)
A simple threshold gate `Φ(γ, τ) = 𝟙(max(γ) > 0.6)` that:
- **Unlocks** specialists when HMM confident (e.g., clear spike/surplus)
- **Locks** to global anchor during ambiguous transitions
- **Eliminates** "High-Confidence Jitter" without hyperparameter sensitivity (robust τ ∈ [0.1, 0.7])

### 3. Hierarchical Residual Learning
Specialists learn **residuals**, not prices directly:
- Guarantees bounded forecasts even in unobserved regimes
- Global anchor provides "control-inspired fallback"
- Level 1 models remain shallow (ElasticNet) → no overfitting

### 4. Optimal Transport Domain Adaptation (OT-HR-LEAR)
Transports Base-regime residual "shapes" to Spike regime via 1D Wasserstein mapping:
- **CRPS 7.39** (−7.1% vs HR-LEAR)
- Resolves "Data Pooling Paradox": borrowing innovations without diluting regime signals

---

## 📊 Experimental Results

### Main Benchmark (Test Set: 2025–2026)

| Model | CRPS | MAE | RMSE | Status |
|-------|------|-----|------|--------|
| Naive Persistence (D-7) | 26.91 | 33.50 | 50.74 | Baseline |
| SARIMA-GARCH | 18.32 | 22.47 | 33.71 | Econometric |
| **MixVol MoE (Neural)** | **24.02** | 19.14 | 42.67 | **Failed** |
| Global LEAR Benchmark | 9.38 | 11.46 | 18.40 | Benchmark |
| CNN-BiLSTM-AR | 10.06 | 12.62 | 23.10 | DL Benchmark |
| MRS-LEAR (Flat Hybrid) | 10.62 | 12.91 | 26.80 | Underperformed |
| **HR-LEAR (Proposed)** | **7.95** | **9.92** | **17.75** | **Proposed** |

### Temporal Coherence & Economic Utility

| Metric | Global LEAR | HR-LEAR | Δ |
|--------|-------------|---------|---|
| Energy Score (24h) | 63.59 | **55.66** | **−12.5%** |
| BESS Arbitrage Profit | €24,082 | **€25,346** | **+5.25%** |
| Spike-Regime MAE | 39.44 | **32.18** | **−18.4%** |

*BESS Setup: 100 MWh Li-ion, 90% round-trip, 0.5% transaction cost, €5/MWh threshold, Sharpe 1.35, Max DD 8%*

### Advanced Ablations (Test Set: 2025–2026)

| Approach | MAE | CRPS | Δ MAE |
|----------|-----|------|-------|
| Baseline (HR-LEAR) | 9.92 | 7.95 | — |
| PI-HR-LEAR (Physics) | 9.87 | 7.97 | −0.5% |
| MV-HR-LEAR (Multivariate) | 10.10 | 8.14 | +1.8% |
| **OT-HR-LEAR (Champion)** | **9.22** | **7.39** | **−7.1%** |

---

## 📁 Repository Structure

```
epf-day-ahead/
├── configs/                    # YAML configuration files
│   ├── base_config.yaml
│   ├── hr_lear.yaml            # HR-LEAR main config
│   ├── ot_hr_lear.yaml         # Optimal Transport variant
│   ├── pi_hr_lear.yaml         # Physics-Informed variant
│   └── mv_hr_lear.yaml         # Multivariate variant
├── data/
│   ├── raw/                    # Raw EPEX SPOT data (not tracked)
│   ├── processed/              # Features, HMM states (not tracked)
│   └── external/               # Fuel prices, weather (not tracked)
├── models/
│   ├── __init__.py
│   ├── hr_lear.py              # Core HR-LEAR implementation
│   ├── elasticnet_lear.py      # Level 0 Global Anchor
│   ├── residual_specialist.py  # Level 1 specialists
│   ├── hmm_regime.py           # Sticky HDP-HMM (Student-t)
│   ├── spi_gate.py             # Stability-Preserving Indicator
│   ├── ot_transport.py         # 1D Wasserstein mapping
│   └── baselines/
│       ├── lear.py             # Standard LEAR
│       ├── mrs_lear.py         # Flat MRS-LEAR
│       ├── cnn_bilstm_ar.py    # Neural benchmark
│       └── mixvol_moe.py       # MixVol MoE
├── training/
│   ├── train_hr_lear.py        # Main training script
│   ├── train_baselines.py      # Baseline training
│   ├── curriculum.py           # SPI threshold scheduling
│   └── losses.py               # CRPS, Quantile, Energy Score
├── evaluation/
│   ├── evaluate.py             # Comprehensive evaluation
│   ├── metrics.py              # CRPS, Winkler, PIT, DM tests
│   ├── backtest_bess.py        # BESS arbitrage simulation
│   └── visualization.py        # Regime plots, forecast ribbons
├── experiments/
│   ├── run_benchmarks.py       # Table 1 reproduction
│   ├── ablation_studies.py     # SPI, feature subsets
│   ├── advanced_ablations.py   # PI, MV, OT variants
│   └── sensitivity_tau.py      # τ threshold sweep
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_spike_analysis.ipynb
│   └── 04_paper_figures.ipynb
├── scripts/
│   ├── download_epex.py
│   ├── run_all_experiments.sh
│   └── generate_tables.py
├── tests/
│   ├── test_hr_lear.py
│   ├── test_metrics.py
│   └── test_data_pipeline.py
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
- 16GB+ RAM recommended
- No GPU required (all models are ElasticNet-based)

### Installation
```bash
git clone https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING.git
cd ELECTRICITY_PRICE_FORECASTING

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Data Preparation
```bash
# Download EPEX SPOT DE data (2015–2026)
python scripts/download_epex.py --market DE --start 2015 --end 2026

# Generate features: lags, calendars, exogenous (load, wind, solar, gas)
python -m utils.features --config configs/base_config.yaml
```

### Training & Evaluation

```bash
# Train HR-LEAR (main model)
python training/train_hr_lear.py --config configs/hr_lear.yaml

# Train baselines for comparison
python training/train_baselines.py --config configs/hr_lear.yaml

# Comprehensive evaluation (reproduces Table 1)
python evaluation/evaluate.py \
    --model-path outputs/hr_lear/best_model.pkl \
    --config configs/hr_lear.yaml \
    --test-years 2025 2026 \
    --metrics crps,mae,rmse,energy_score,bess

# Advanced ablation: OT-HR-LEAR (champion)
python training/train_hr_lear.py --config configs/ot_hr_lear.yaml
python evaluation/evaluate.py --model-path outputs/ot_hr_lear/best_model.pkl ...

# Generate paper tables & figures
python scripts/generate_tables.py --results-dir outputs/
python notebooks/04_paper_figures.ipynb
```

### Configuration Example (`configs/hr_lear.yaml`)
```yaml
data:
  market: "DE"
  train_years: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
  test_years: [2025, 2026]
  target_transform: "asinh"  # inverse hyperbolic sine for negative prices
  features:
    price_lags: [1, 2, 3, 7, 24, 168]
    calendars: ["hour", "dayofweek", "month", "holiday"]
    exogenous: ["load_fc", "wind_fc", "solar_fc", "gas_price", "coal_price"]

model:
  level0:
    type: "ElasticNetEnsemble"
    n_estimators: 24  # one per hour
    alpha: 0.01
    l1_ratio: 0.5
  
  regime_detector:
    type: "StickyHDPHMM"
    n_states: 3
    emission: "student_t"
    sticky_prior: 10.0
  
  level1:
    type: "ResidualSpecialists"
    n_specialists: 3  # Surplus, Base, Spike
    alpha: 0.01
    l1_ratio: 0.5
  
  spi:
    tau: 0.6
    adaptive: false

training:
  horizon: 24
  random_seed: 42
```

---

## 📈 Reproducing Key Results

| Result | Command |
|--------|---------|
| **Table 1** (Main benchmark) | `python experiments/run_benchmarks.py --output results/table1.csv` |
| **Table 2** (Energy Score, BESS) | `python evaluation/backtest_bess.py --model hr_lear --output results/table2.csv` |
| **Table 3** (Advanced ablations) | `python experiments/advanced_ablations.py --output results/table3.csv` |
| **Figure 1** (Architecture) | See `figures/architecture.tex` (TikZ) |
| **Figure 2** (Regime transitions) | `python notebooks/04_paper_figures.ipynb` |
| **Figure 3** (τ sensitivity) | `python experiments/sensitivity_tau.py --plot` |
| **Figure 4** (7-day volatility window) | `python notebooks/04_paper_figures.ipynb` |

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{kamal2026hrlear,
  title     = {Hierarchical Regime-Aware LEAR: Resolving the Neural Mixture-of-Experts Collapse in Electricity Price Forecasting},
  author    = {Kamal},
  journal   = {AIMS Energy},
  year      = {2026},
  volume    = {5},
  number    = {x},
  pages     = {xxx--xxx},
  doi       = {xxxxx},
  note      = {Code: https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **EPFtoolbox** — [N. Espinosa et al.](https://github.com/jeslago/epftoolbox) for benchmark data and baseline implementations
- **EPEX SPOT** — German day-ahead market data
- **Thesis Supervisor** — Prof. [Name], Department of Electrical and Electronic Engineering, KNUST
- **Funding** — [Funding body if applicable]

---

## 📧 Contact

**Kamal** — Department of Electrical and Electronic Engineering, Kwame Nkrumah University of Science and Technology, Kumasi, Ghana  
✉️ kamal@knust.edu.gh  

Project Link: [https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING](https://github.com/raidenkhan/ELECTRICITY_PRICE_FORECASTING)