# AI Market Research

Independent research project for prototyping machine learning models for financial market analysis and algorithmic trading research.

I use this repo to develop and benchmark models on market data, run backtests on historical data, and evaluate GPU performance for training and inference — including ROCm on AMD GPUs via AMD Developer Cloud.

> Educational research only. Not financial advice, and no live trading.

## What this repo does now

- Generates a reproducible synthetic OHLCV dataset (seeded, so results are comparable run to run)
- Runs a simple, explainable baseline strategy: SMA 20 / SMA 50 crossover
- Backtests it and reports:
  - Total return and CAGR
  - Annualised volatility and Sharpe ratio
  - Max drawdown
  - Win rate and number of trades
- Saves the equity curve to `results/equity_curve.png` (if matplotlib is installed)

This baseline is intentionally simple. It is the benchmark that later ML models have to beat.

## Project structure

```
ai-market-research/
├── README.md
├── LICENSE
├── requirements.txt
├── main.py                 # synthetic data + SMA backtest, run this first
├── data/
│   └── README.md           # where real market data will live (not committed)
└── results/
    └── .gitkeep
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Example output (seed 42 — your numbers will be the same with this seed):

```
Period:          252 trading days
Total return:    -14.41%
CAGR:            -14.41%
Volatility:      11.67% annualised
Sharpe ratio:    -1.27
Max drawdown:    -15.67%
Trades:          5
Win rate:        40.2% (invested days)
```

On synthetic random data the baseline can lose money — that is expected. It is a benchmark to beat, not a trading signal. The point is the pipeline: data → signal → backtest → metrics.

## Roadmap

1. **Real market data** — replace the synthetic generator with CCXT / Yahoo Finance data, cached locally in `data/`
2. **Feature engineering** — returns, volatility, volume, momentum and regime features
3. **ML models** — logistic regression and gradient boosting first, then sequence models, all benchmarked against the SMA baseline in this repo
4. **GPU evaluation** — benchmark training and inference on AMD GPUs (ROCm) on AMD Developer Cloud, and compare throughput / cost vs. CPU
5. **Walk-forward validation** — no look-ahead, out-of-sample testing only

## Tech stack

- Python 3.10+
- pandas, numpy — data and backtest
- matplotlib — charts (optional for the core backtest)

## Licence

MIT — see `LICENSE`.
