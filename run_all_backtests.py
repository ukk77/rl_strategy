"""Run backtests on all trained models and produce a summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_strategy.backtest.engine import run_backtest

models_dir = Path("models")
results = {}
failed = []

model_files = sorted(models_dir.glob("*_ppo.zip"))
print(f"Found {len(model_files)} model files\n")

for model_path in model_files:
    ticker = model_path.stem.replace("_ppo", "")
    if "_v20" in ticker:
        continue
    try:
        print(f"Backtesting {ticker}...", end=" ", flush=True)
        metrics = run_backtest(
            ticker, str(model_path), num_episodes=5, train_split=0, save=True
        )
        results[ticker] = {
            "mean_return": metrics.mean_return,
            "sharpe": metrics.mean_sharpe,
            "win_rate": metrics.trade_win_rate,
            "max_dd": metrics.mean_max_drawdown,
            "trades": metrics.total_trades,
        }
        print(f"Return={metrics.mean_return:>10.2f}  Sharpe={metrics.mean_sharpe:>7.2f}  "
              f"WinRate={metrics.trade_win_rate:>6.1%}  MaxDD={metrics.mean_max_drawdown:>6.1f}%")
    except Exception as e:
        print(f"FAILED: {e}")
        failed.append(ticker)

# Summary
print()
print("=" * 75)
print(f"BACKTEST SUMMARY: {len(results)} succeeded, {len(failed)} failed")
print("=" * 75)
print(f"{'Ticker':<10} {'Return':>12} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8} {'Trades':>8}")
print("-" * 60)

for ticker, r in sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True):
    print(f"{ticker:<10} {r['mean_return']:>12.2f} {r['sharpe']:>8.2f} "
          f"{r['win_rate']:>7.1%} {r['max_dd']:>7.1f}% {r['trades']:>8}")

if failed:
    print(f"\nFailed tickers: {', '.join(failed)}")

# Top/Bottom performers
print(f"\nTop 5 by Sharpe:")
for ticker, r in sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)[:5]:
    print(f"  {ticker}: Sharpe={r['sharpe']:.2f}, Return={r['mean_return']:.2f}")

print(f"\nBottom 5 by Sharpe:")
for ticker, r in sorted(results.items(), key=lambda x: x[1]["sharpe"])[:5]:
    print(f"  {ticker}: Sharpe={r['sharpe']:.2f}, Return={r['mean_return']:.2f}")
