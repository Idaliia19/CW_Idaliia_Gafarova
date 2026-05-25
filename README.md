# Realized Volatility Forecasting with Neural Networks

Replication of **Bucci, A. (2020),** "Realized Volatility Forecasting with Neural Networks," *Journal of Financial Econometrics* 18(3), 502–531, applied to high-frequency single-stock data.

Coursework for *Financial Analytics and Machine Learning* (May 2026). The original study forecasts monthly S&P 500 realized volatility (1950–2017); here the same methodology is applied to **daily** realized volatility built from 5-minute returns for three individual stocks — AAPL, AMZN, JPM — over 2016–2024 (2,264 trading days each).

## Quick start

```bash
pip install -r requirements.txt
jupyter lab          # or: jupyter notebook
```

Run the notebooks in numerical order (`notebooks/01` → `notebooks/15`). Each notebook caches its intermediate outputs to `data/processed/`, so later notebooks read those files and can also be run on their own. Figures are written to `results/figures/` (PNG, 300 dpi) and tables to `results/tables/` (CSV).

## Methodology

- **Target.** Daily realized volatility = sum of squared 5-minute log-returns within the regular session; the forecasting target is log(√RV), approximately Gaussian (Andersen et al., 2001). 5-minute sampling follows Liu, Patton & Sheppard (2015).
- **Models (12).**
  - *Econometric:* Random Walk, AR (BIC lag selection), HAR (Corsi 2009), ARFIMA(0,d,1), LSTAR (Teräsvirta 1994), ARFIMAX.
  - *Neural (PyTorch):* FNN, LSTM (Hochreiter–Schmidhuber 1997), NAR, ENN (Elman), JNN (Jordan), NARX (Lin et al. 1996).
- **Exogenous block.** Seven daily macro-financial features — five reconstructing Bucci's predictors (MKT, TB, TS, DEF, INF) plus log VIX and the dollar-index return — sourced from FRED and Yahoo Finance, feeding ARFIMAX and NARX.
- **Evaluation.** Rolling window (train 1,585 / test 679), one-step and iterative five-step forecasts; MSE on log-RV and QLIKE on the variance scale (Patton 2011); Diebold–Mariano with the Harvey–Leybourne–Newbold correction; Model Confidence Set (Hansen–Lunde–Nason 2011, stationary bootstrap, 10,000 replications); Fair–Shiller encompassing with HAC standard errors.

## Project structure

```
data/
  raw/        — 1-minute OHLCV bars for AAPL, AMZN, JPM (2016–2024)
  processed/  — daily log-RV series, rolling forecasts, macro panel
src/
  data_loader.py          — load and clean intraday bars
  rv_estimator.py         — 5-minute realized volatility
  macro_data.py           — FRED + Yahoo macro-financial features
  econometric_models.py   — RW, AR, HAR, ARFIMA, ARFIMAX
  neural_models.py        — FNN, LSTM, NAR, NARX
  recurrent_networks.py   — ENN, JNN
  lstar_model.py          — logistic smooth-transition AR
  forecast_engine.py      — rolling-window forecasting
  multistep_engine.py     — iterative k-step forecasts
  metrics.py              — MSE, QLIKE, Diebold–Mariano
  mcs_test.py             — Model Confidence Set
  encompassing.py         — Fair–Shiller / Chong–Hendry encompassing
  stress_periods.py       — stress-window identification
notebooks/    — 15 analysis notebooks (01–15), run in order
results/
  figures/    — PNG plots (300 dpi)
  tables/     — CSV result tables
paper/        — Bucci (2020) reference PDF
```

## Main findings

On daily single-stock data the pooled Model Confidence Set (90% level, on both MSE and QLIKE) retains exactly three models: **ARFIMA, ARFIMAX and NARX**.

- **ARFIMAX** posts the lowest project-wide QLIKE (0.1300 vs ARFIMA's 0.1329); its average MSE is a statistical wash with ARFIMA.
- **NARX** records the single lowest MSE on JPM (0.0565) and significantly improves on its exog-free twin NAR (−2.5% MSE, −5.2% QLIKE; Diebold–Mariano significant on AMZN, marginal on JPM).
- Adding the macro-financial block is what moves the neural model from *rejected* (exog-free) to *retained* in the superior set — the direction of Bucci's headline mechanism.
- The exog-free LSTM and NAR only tie statistically with HAR rather than dominating it; the shallow FNN/ENN/JNN sit at the bottom of the ranking, consistent with Bucci's vanishing-gradient argument.

These results reproduce the *direction* of Bucci's finding (exogenous predictors make the neural architecture statistically competitive) without fully reversing the ranking: daily US-equity log-RV has extraordinarily strong long memory (Hurst ≈ 0.94), which already makes ARFIMA hard to beat, and only five of Bucci's eleven predictors are reconstructed at daily frequency.

## Test window

- Training: 1,585 trading days (2016-01-04 → 2022-04-19).
- Testing: 679 trading days (2022-04-20 → 2024-12-31).
- Rolling window throughout; neural networks refit every 22 trading days for tractability, RW and HAR refit daily.

## Reproducibility

All models use `seed=42`, reset on every `fit()` call for within-window determinism. Notebooks run end-to-end without manual intervention. Developed and tested with Python 3.12 (see `requirements.txt`).

## Author

Idaliia Gafarova — Student #25194643
