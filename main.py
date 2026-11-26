"""
AI Market Research — baseline backtest.

Generates a reproducible synthetic OHLCV series, runs an SMA 20 / SMA 50
crossover strategy, and reports standard backtest metrics.

Run:  python main.py
No API keys, no network access, and no live trading.
"""

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 252
SMA_FAST = 20
SMA_SLOW = 50


def generate_prices(n: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """Synthetic daily prices: geometric random walk, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    # ~0.05% daily drift, ~1.5% daily volatility
    daily_returns = rng.normal(loc=0.0005, scale=0.015, size=n)
    close = 100 * np.exp(np.cumsum(daily_returns))
    df = pd.DataFrame({"date": dates, "close": close})
    df["open"] = df["close"].shift(1).fillna(100.0)
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + rng.uniform(0, 0.005, size=n))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - rng.uniform(0, 0.005, size=n))
    df["volume"] = rng.integers(800_000, 2_500_000, size=n)
    return df.set_index("date")


def backtest(df: pd.DataFrame) -> dict:
    """SMA crossover: long (1) when SMA_FAST > SMA_SLOW, otherwise flat (0)."""
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(SMA_FAST).mean()
    df["sma_slow"] = df["close"].rolling(SMA_SLOW).mean()
    df["signal"] = (df["sma_fast"] > df["sma_slow"]).astype(int)
    # Position taken at next open: shift signal by 1 day to avoid look-ahead.
    df["position"] = df["signal"].shift(1).fillna(0)
    df["daily_ret"] = df["close"].pct_change().fillna(0)
    df["strategy_ret"] = df["position"] * df["daily_ret"]
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    df["buy_hold_equity"] = (1 + df["daily_ret"]).cumprod()

    equity = df["equity"]
    total_return = equity.iloc[-1] - 1
    n_years = len(df) / 252
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0
    volatility = df["strategy_ret"].std() * np.sqrt(252)
    sharpe = df["strategy_ret"].mean() / df["strategy_ret"].std() * np.sqrt(252) if df["strategy_ret"].std() else 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    trades = int((df["position"].diff().fillna(0) != 0).sum() // 2)
    # Win rate on days the strategy was invested and moved.
    invested = df[df["position"] == 1]["strategy_ret"]
    win_rate = (invested > 0).mean() if len(invested) else 0.0

    return {
        "df": df,
        "period_days": len(df),
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "win_rate": win_rate,
    }


def main() -> None:
    df = generate_prices()
    result = backtest(df)

    print(f"Period:          {result['period_days']} trading days")
    print(f"Total return:    {result['total_return']:.2%}")
    print(f"CAGR:            {result['cagr']:.2%}")
    print(f"Volatility:      {result['volatility']:.2%} annualised")
    print(f"Sharpe ratio:    {result['sharpe']:.2f}")
    print(f"Max drawdown:    {result['max_drawdown']:.2%}")
    print(f"Trades:          {result['trades']}")
    print(f"Win rate:        {result['win_rate']:.1%} (invested days)")

    # Optional chart — skip gracefully if matplotlib is not installed.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        result["df"]["equity"].plot(ax=ax, label="SMA 20/50 strategy")
        result["df"]["buy_hold_equity"].plot(ax=ax, label="Buy and hold", alpha=0.7)
        ax.set_title("Equity curve — SMA crossover vs buy and hold (synthetic data)")
        ax.set_ylabel("Equity (start = 1.0)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/equity_curve.png", dpi=150)
        print("Chart saved:     results/equity_curve.png")
    except ImportError:
        print("matplotlib not installed — chart skipped (pip install -r requirements.txt)")


if __name__ == "__main__":
    main()
