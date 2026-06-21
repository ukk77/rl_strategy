"""Auto-retrain trigger for RL agents.

Monitors paper trading P&L and evaluation metrics to detect
performance degradation. Triggers retraining when thresholds are breached.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from rl_strategy.config import get_config
from rl_strategy.paper_trading.db import PaperTradingDB
from rl_strategy.agent.train import train_single_ticker


# Default degradation thresholds
DEFAULT_THRESHOLDS = {
    "pnl_decline_pct": -15.0,        # Retrain if 30-day P&L drops > 15%
    "win_rate_min": 0.35,            # Retrain if win rate falls below 35%
    "max_consecutive_losses": 5,     # Retrain after 5 consecutive losing trades
    "min_trades_for_check": 10,      # Need at least 10 trades before checking
    "lookback_days": 30,             # Look back 30 days for P&L calculation
}


def check_degradation(
    ticker: str,
    thresholds: Optional[Dict] = None,
) -> Tuple[bool, str, Dict]:
    """Check if a model's performance has degraded enough to warrant retraining.

    Args:
        ticker: Stock symbol
        thresholds: Dict of threshold values (uses defaults if None)

    Returns:
        Tuple of (should_retrain, reason, diagnostics_dict)
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    db = PaperTradingDB()
    trades = db.get_trades(ticker=ticker, limit=100)
    positions = db.get_all_positions()
    stats = db.get_performance_stats()

    diagnostics = {
        "ticker": ticker,
        "total_trades": stats.get("total_trades", 0),
        "win_rate": stats.get("win_rate", 0),
        "total_pnl": stats.get("total_pnl", 0),
        "checked_at": datetime.now().isoformat(),
    }

    # Not enough trades to assess
    if len(trades) < thresholds["min_trades_for_check"]:
        return False, f"Only {len(trades)} trades (need {thresholds['min_trades_for_check']})", diagnostics

    # Check win rate
    win_rate = stats.get("win_rate", 0)
    diagnostics["win_rate"] = win_rate
    if win_rate < thresholds["win_rate_min"]:
        return True, f"Win rate {win_rate:.1%} below minimum {thresholds['win_rate_min']:.1%}", diagnostics

    # Check total P&L
    total_pnl = stats.get("total_pnl", 0)
    diagnostics["total_pnl"] = total_pnl
    if total_pnl < 0 and abs(total_pnl) > 1000:
        pnl_pct = (total_pnl / 100000.0) * 100  # Assuming $100K initial capital
        diagnostics["pnl_pct"] = pnl_pct
        if pnl_pct < thresholds["pnl_decline_pct"]:
            return True, f"P&L {pnl_pct:.1f}% below threshold {thresholds['pnl_decline_pct']:.1f}%", diagnostics

    # Check consecutive losses
    if trades:
        consecutive_losses = 0
        for trade in trades:
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                if trade.pnl < 0:
                    consecutive_losses += 1
                else:
                    break
        diagnostics["consecutive_losses"] = consecutive_losses
        if consecutive_losses >= thresholds["max_consecutive_losses"]:
            return True, f"{consecutive_losses} consecutive losses (threshold: {thresholds['max_consecutive_losses']})", diagnostics

    return False, "Performance within acceptable range", diagnostics


def check_all_tickers(
    thresholds: Optional[Dict] = None,
    auto_retrain: bool = False,
    retrain_timesteps: int = 50000,
) -> Dict[str, Dict]:
    """Check all configured tickers for degradation.

    Args:
        thresholds: Degradation thresholds
        auto_retrain: If True, automatically retrain degraded models
        retrain_timesteps: Timesteps for auto-retraining

    Returns:
        Dict mapping ticker -> diagnostics
    """
    cfg = get_config()
    results = {}

    print(f"\n{'='*60}")
    print("AUTO-RETRAIN CHECK")
    print(f"{'='*60}")
    print(f"Checking {len(cfg.tickers)} tickers...\n")

    degraded = []

    for ticker in cfg.tickers:
        should_retrain, reason, diagnostics = check_degradation(ticker, thresholds)
        results[ticker] = diagnostics

        status = "DEGRADED" if should_retrain else "OK"
        print(f"  {ticker:<8} [{status:<10}] {reason}")

        if should_retrain:
            degraded.append(ticker)

    print(f"\nSummary: {len(degraded)}/{len(cfg.tickers)} tickers degraded")

    if degraded and auto_retrain:
        print(f"\nAuto-retraining {len(degraded)} degraded models...")
        for ticker in degraded:
            print(f"\n  Retraining {ticker}...")
            model_path = train_single_ticker(ticker, timesteps=retrain_timesteps)
            if model_path:
                print(f"  [OK] New model: {model_path}")
            else:
                print(f"  [X] Retrain failed for {ticker}")

    # Save check results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "retrain_check.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": thresholds or DEFAULT_THRESHOLDS,
        "degraded_tickers": degraded,
        "auto_retrained": auto_retrain,
        "results": results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[OK] Check results saved to: {output_path}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check for model degradation")
    parser.add_argument("--ticker", help="Check specific ticker")
    parser.add_argument("--all", action="store_true", help="Check all tickers")
    parser.add_argument("--auto-retrain", action="store_true", help="Auto-retrain degraded models")
    parser.add_argument("--retrain-timesteps", type=int, default=50000, help="Timesteps for retraining")
    args = parser.parse_args()

    if args.ticker:
        should_retrain, reason, diagnostics = check_degradation(args.ticker)
        print(f"\n{ticker}: {'DEGRADED' if should_retrain else 'OK'} — {reason}")
        print(f"  Win rate: {diagnostics.get('win_rate', 0):.1%}")
        print(f"  Total P&L: ${diagnostics.get('total_pnl', 0):.2f}")
        print(f"  Total trades: {diagnostics.get('total_trades', 0)}")
    elif args.all:
        check_all_tickers(auto_retrain=args.auto_retrain, retrain_timesteps=args.retrain_timesteps)
    else:
        parser.print_help()
