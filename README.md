# var-es-forecasting

**Value-at-Risk and Expected Shortfall Forecasting using Path Signatures**

A production-style financial risk forecasting pipeline benchmarking six models — from classical statistical methods to deep learning with path signature features — evaluated on 1,507 out-of-sample S&P 500 trading days (2020–2025) using a rigorous statistical backtesting framework.

---

## Overview

This project implements, trains, and evaluates models for one-step-ahead **VaR** and **ES** forecasting at confidence level τ = 5%, with full regime analysis (low-stress vs high-stress) and compliance-grade backtests.

> **Regulatory context:** Basel IV requires daily VaR and ES reporting for market risk capital. Classical models systematically fail during tail-risk regimes — this project quantifies exactly where and why.

---

## Models

| Model | Type | VaR | ES |
|---|---|---|---|
| Historical Simulation | Non-parametric | ✅ | ✅ |
| GARCH(1,1) | Parametric | ✅ | ✅ |
| Quantile Regression | Semi-parametric | ✅ | ✅ |
| LSTM-Classical | Deep learning | ✅ | ✅ |
| LSTM + Path Signatures | Deep learning | ✅ | ✅ |

---

## Key Results

### VaR Backtest (τ = 5%, N = 1,507 days)

| Model | Coverage | CC Test | DQ Test | Pinball |
|---|---|---|---|---|
| **LSTM+Sig** | 4.45% | Pass | Pass* | 0.1582 |
| LSTM-Cls | 4.38% | Pass | **Fail** (lag-2, t=2.85) | 0.1582 |
| GARCH(1,1) | 4.78% | Pass | Fail | **0.1556 ★** |
| Hist-Sim | 5.37% | Fail | Fail | 0.1677 |
| Quant-Reg | 5.11% | Pass | Fail | 0.1583 |

*DQ result from aligned comparison (Model_comparisons.ipynb). Full-window DQ dominated by COVID cluster.

### ES Backtest (τ = 5%, N = 1,507 days)

| Model | FZ Score | MF Test | ER Test | ES/VaR |
|---|---|---|---|---|
| **LSTM+Sig** | **−1.583 ★** | Fail | Fail | 2.01 |
| LSTM-Cls | −1.525 | Fail | Fail | 1.94 |
| Quant-Reg | −1.023 | **Pass** | **Pass** | 1.57 |
| GARCH(1,1) | −0.863 | Pass | Fail | 1.40 |
| Hist-Sim | −0.768 | Fail | Fail | 1.50 |

### Regime Analysis

| Model | Low-stress coverage | Low-stress Kupiec p | High-stress coverage |
|---|---|---|---|
| **LSTM+Sig** | **4.4%** | **0.61** | 9.9% |
| LSTM-Cls | 5.5% | 0.44 | 13.7% |
| GARCH(1,1) | 5.2% | 0.81 | 9.0% |
| Hist-Sim | 3.5% | 0.014 | 11.7% |

> **Key finding:** LSTM+Sig is the best-calibrated model in calm markets. All models fail during COVID crash and 2022 bear market — a universal distribution-shift problem, not model-specific.

---

## Project Structure

```
var-es-forecasting/
│
├── data/
│   └── fetch_data.py              # S&P 500 + VIX download via yfinance
│
├── features/
│   ├── classical_features.py      # Lagged returns, RV20, GARCH vol, VIX
│   └── signature_features.py      # Path signature computation (depth 3, rolling 20d)
│
├── models/
│   ├── historical_simulation.py
│   ├── garch.py                   # GARCH(1,1) Normal & Student-t
│   ├── quantile_regression.py
│   ├── lstm_model.py              # LSTM-Cls and LSTM+Sig (shared architecture)
│
├── loss/
│   └── joint_loss.py              # FZ score + coverage penalty + ES/VaR ratio
│
├── backtesting/
│   ├── var_backtests.py           # Kupiec, Christoffersen CC, DQ test
│   └── es_backtests.py            # McNeil-Frey, Exceedance Residuals, FZ scoring
│
├── notebooks/
│   ├── Model_comparisons.ipynb    # Full model comparison + DQ test results
│
├── results/
│   ├── var_results.csv
│   ├── es_results.csv
│   ├── backtest_results.csv
│   └── regime_results.csv
│
├── figures/                       # All output plots
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/var-es-forecasting.git
cd var-es-forecasting
pip install -r requirements.txt
```

**Requirements:**
```
torch
numpy
pandas
scipy
yfinance
arch          # GARCH estimation
esig          # Path signatures
statsmodels
matplotlib
seaborn
```

---

## Usage

### 1. Fetch data
```python
python data/fetch_data.py
# Downloads S&P 500 and VIX from 2000-01-01 to 2025-03-31
```

### 2. Run baseline models
```python
python models/historical_simulation.py
python models/garch.py
python models/quantile_regression.py
```

### 3. Train LSTM models
```python
# Classical features only
python models/lstm_model.py --variant classical

# With path signatures
python models/lstm_model.py --variant signatures --depth 3 --window 20
```

### 4. Train DLN
```python
python models/dln_model.py --taus 0.01 0.05 0.10 --n-lattices 50
```

### 5. Run backtests
```python
python backtesting/var_backtests.py   # Kupiec, CC, DQ test
python backtesting/es_backtests.py    # MF test, FZ score
```

Or run the full comparison notebook: `notebooks/Model_comparisons.ipynb`

---

## Methodology

### Path Signatures
A path signature is a collection of iterated integrals of a time series path, providing a **universal, order-sensitive fingerprint** of the trajectory. For return path X over a rolling 20-day window at depth 3:

```
S(X) = (1, S¹, S², S¹², S²¹, S¹¹, S²², S¹²¹, ...)
```

Depth-3 signatures over a 2-dimensional path (returns, time) yield **14 features** that capture nonlinear co-movement and momentum-reversal patterns invisible to classical lag features.

**Why signatures help:** Classical LSTM fails the Dynamic Quantile test due to lag-2 autocorrelation in violations (t=2.85). Signature features encode the *order* of past moves — up-then-down vs down-then-up are treated differently — which breaks this dependence.

### Joint VaR–ES Loss
Models are trained to simultaneously minimise:

```
L = S_FZ(q̂, ê; r) + λ₁·L_coverage + λ₂·L_ratio
```

where `S_FZ` is the Fissler-Ziegel strictly consistent scoring rule for the (VaR, ES) pair — the only loss function that provides consistent joint estimation.

### Backtesting Framework
- **Kupiec POF test** — correct violation frequency
- **Christoffersen CC test** — correct frequency + independence
- **Dynamic Quantile (DQ) test** — no autocorrelation in violation indicator (6 lags + VaR regressor)
- **McNeil-Frey test** — unbiased ES on violation days
- **Fissler-Ziegel (FZ0) score** — strictly consistent joint VaR+ES evaluation (lower = better)

---

## Data

| | Details |
|---|---|
| Asset | S&P 500 Index (^GSPC) |
| Auxiliary | CBOE VIX |
| Source | `yfinance` |
| Full period | Jan 2000 – Mar 2025 |
| Training | Jan 2000 – Dec 2017 |
| Validation | Jan 2018 – Dec 2019 |
| Test | Jan 2020 – Mar 2025 (1,507 days) |
| Stress periods | COVID crash (Feb–Jun 2020), 2022 bear market (Jan–Dec 2022) |

---

## References

- Chevyrev & Kormilitzin (2016). *A Primer on the Signature Method in Machine Learning.* arXiv:1603.03788
- Narayan et al. (2021). *Regularization Strategies for Quantile Regression.* arXiv:2102.05135
- Fissler & Ziegel (2016). *Higher Order Elicitability and Osband's Principle.* Annals of Statistics, 44(4)
- Chronopoulos, Raftapostolos & Kapetanios (2024). *Forecasting VaR Using Deep Neural Network Quantile Regression.* Journal of Financial Econometrics, 22(3), 636–669
- Kupiec (1995). *Techniques for Verifying the Accuracy of Risk Measurement Models.* Journal of Derivatives
- Christoffersen (1998). *Evaluating Interval Forecasts.* International Economic Review, 39(4)
- McNeil & Frey (2000). *Estimation of Tail-Related Risk Measures.* Journal of Empirical Finance

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Built as part of a B.Tech Major Project — Dhirubhai Ambani University, April 2026*
