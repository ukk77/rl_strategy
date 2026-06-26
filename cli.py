"""Command Line Interface for RL Strategy.

Provides commands: signals, train, backtest, paper, positions
"""
import argparse
import sys
from pathlib import Path
from typing import List

# Setup imports
def setup_paths():
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))


setup_paths()

from rl_strategy.agent.train import train_single_ticker, train_all_tickers
from rl_strategy.paper_trading.tracker import PaperTradingTracker
from rl_strategy.signals.generator import create_generator
from rl_strategy.config import get_config


def cmd_signals(ticker: str = None, all_tickers: bool = False):
    """Generate current signals."""
    cfg = get_config()
    tickers = cfg.tickers if all_tickers else ([ticker] if ticker else ["AAPL"])
    
    print("\n" + "="*60)
    print("RL STRATEGY - SIGNALS")
    print("="*60 + "\n")
    
    for t in tickers:
        # Load trained model if available
        model_path = f"{cfg.models_dir}/{t}_ppo.zip"
        generator = create_generator(t, model_path=model_path if Path(model_path).exists() else None)
        if generator is None:
            print(f"✗ {t}: Failed to create generator")
            continue
        
        signal = generator.generate_signal()
        print(f"  {signal}")
    
    print()


def cmd_train(ticker: str = None, all_tickers: bool = False, timesteps: int = 100000):
    """Train RL agent."""
    if all_tickers:
        train_all_tickers(timesteps=timesteps)
    elif ticker:
        train_single_ticker(ticker, timesteps=timesteps)
    else:
        print("Error: Specify --ticker or --all")


def cmd_backtest(ticker: str, episodes: int = 10, train_split: float = 0.3):
    """Run backtest with trained agent using the RL-native backtest engine."""
    from rl_strategy.backtest.engine import run_backtest
    
    print("\n" + "="*65)
    print(f"RL STRATEGY - BACKTEST: {ticker}")
    print("="*65 + "\n")
    
    cfg = get_config()
    model_path = f"{cfg.models_dir}/{ticker}_ppo.zip"
    
    if not Path(model_path).exists():
        print(f"[X] Model not found: {model_path}")
        print(f"  Train first with: python -m rl_strategy.cli train --ticker {ticker}")
        return
    
    try:
        metrics = run_backtest(
            ticker=ticker,
            model_path=model_path,
            num_episodes=episodes,
            train_split=train_split,
            save=True,
        )
    except FileNotFoundError as e:
        print(f"[X] {e}")
        return
    except Exception as e:
        print(f"[X] Backtest failed: {e}")
        return
    
    # ── Episode-Level Summary ──
    print(f"{'Episode':<10} {'Return':>12} {'Return%':>10} {'Trades':>8} {'Steps':>8} {'Max DD%':>10} {'Sharpe':>8}")
    print("-" * 70)
    for ep in metrics.episodes:
        print(f"{ep.episode_id:<10} ${ep.total_return:>10.2f} {ep.total_return_pct:>8.1f}% "
              f"{ep.num_trades:>8} {ep.num_steps:>8} {ep.max_drawdown_pct:>8.1f}% {ep.sharpe_ratio:>8.2f}")
    print("-" * 70)
    print(f"{'AGGREGATE':<10} ${metrics.mean_return:>10.2f} "
          f"{'':>10} {metrics.mean_num_trades:>8.1f} {metrics.mean_episode_length:>8.1f} "
          f"{metrics.mean_max_drawdown:>8.1f}% {metrics.mean_sharpe:>8.2f}")
    print()
    
    # ── Aggregate Stats ──
    print("Episode Statistics:")
    print(f"  Mean Return:     ${metrics.mean_return:>10.2f}  (+/- ${metrics.std_return:.2f})")
    print(f"  Median Return:   ${metrics.median_return:>10.2f}")
    print(f"  Range:           ${metrics.min_return:>10.2f}  to  ${metrics.max_return:.2f}")
    print(f"  Mean Sharpe:      {metrics.mean_sharpe:>10.2f}  (+/- {metrics.std_sharpe:.2f})")
    print(f"  Mean Max DD:      {metrics.mean_max_drawdown:>9.1f}%  (+/- {metrics.std_max_drawdown:.1f}%)")
    print()
    
    # ── Action Distribution ──
    print("Action Distribution (avg across episodes):")
    print(f"  HOLD:  {metrics.hold_pct:>5.1f}%")
    print(f"  BUY:   {metrics.buy_pct:>5.1f}%")
    print(f"  SELL:  {metrics.sell_pct:>5.1f}%")
    print()
    
    # ── Trade-Level Stats ──
    if metrics.total_trades > 0:
        print(f"Trade Statistics ({metrics.total_trades} total trades):")
        print(f"  Win Rate:        {metrics.trade_win_rate*100:>5.1f}%")
        print(f"  Mean Trade P&L:  ${metrics.mean_trade_pnl:>10.2f}")
        print(f"  Mean Trade P&L%:  {metrics.mean_trade_pnl_pct:>9.2f}%")
        print(f"  Mean Holding:     {metrics.mean_holding_steps:>9.1f} steps")
        print()
    
    # ── Overfitting Check ──
    if metrics.overfit_ratio is not None:
        verdict = "OK" if metrics.overfit_ratio > 0.5 else "WARNING: Possible overfitting"
        print("Overfitting Check (train/test split):")
        print(f"  Train Return:    ${metrics.train_return:>10.2f}")
        print(f"  Test Return:     ${metrics.test_return:>10.2f}")
        print(f"  Test/Train Ratio: {metrics.overfit_ratio:>8.2f}  [{verdict}]")
        print()
    
    print(f"[OK] Results saved to: results/{ticker}_backtest.json\n")


def cmd_paper():
    print("ERROR: Independent paper trading is disabled. Please use the unified harness: python -m harness.cli run")
    import sys
    sys.exit(1)


def cmd_positions():
    print("ERROR: Independent paper trading is disabled. Please use the unified harness: python -m harness.cli positions")
    import sys
    sys.exit(1)


def cmd_results(ticker: str = None):
    """View evaluation results."""
    import json
    
    print("\n" + "="*60)
    print("RL STRATEGY - EVALUATION RESULTS")
    print("="*60 + "\n")
    
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("No results directory found\n")
        return
    
    if ticker:
        # Show specific ticker
        result_file = results_dir / f"{ticker}_evaluation.json"
        if not result_file.exists():
            print(f"No results found for {ticker}\n")
            return
        
        with open(result_file) as f:
            result = json.load(f)
        
        print(f"Ticker: {result['ticker']}")
        print(f"Trained: {result['timestamp']}")
        print(f"Timesteps: {result['timesteps']:,}")
        print(f"\nMetrics:")
        print(f"  Mean Return:  {result['metrics']['mean_return']:.4f}")
        print(f"  Std Return:   {result['metrics']['std_return']:.4f}")
        print(f"  Mean Length:  {result['metrics']['mean_length']:.1f} steps")
        print(f"  Mean Trades:  {result['metrics']['mean_trades']:.1f}")
        print(f"  Returns:      {[f'{r:.2f}' for r in result['metrics']['returns']]}")
        print()
    else:
        # Show all results
        result_files = list(results_dir.glob("*_evaluation.json"))
        
        if not result_files:
            print("No evaluation results found\n")
            return
        
        print(f"{'Ticker':<10} {'Timestamp':<20} {'Mean Return':>12} {'Trades':>8}")
        print("-" * 60)
        
        for rf in sorted(result_files):
            with open(rf) as f:
                result = json.load(f)
            
            ts = result['timestamp'][:19].replace('T', ' ')
            mean_ret = result['metrics']['mean_return']
            trades = result['metrics']['mean_trades']
            
            print(f"{result['ticker']:<10} {ts:<20} {mean_ret:>12.4f} {trades:>8.1f}")
        
        print()


def cmd_versions(ticker: str = None):
    """List model versions."""
    from rl_strategy.agent.train import list_model_versions
    
    print("\n" + "="*60)
    print("RL STRATEGY - MODEL VERSIONS")
    print("="*60 + "\n")
    
    if ticker:
        versions = list_model_versions(ticker)
        if not versions:
            print(f"No versions found for {ticker}\n")
            return
        print(f"Versions for {ticker}:")
        print(f"{'Version':<20} {'Return':>12} {'Trades':>8} {'Timesteps':>10}")
        print("-" * 55)
        for v in versions:
            print(f"{v['version']:<20} {v['mean_return']:>12.4f} {v['mean_trades']:>8.1f} {v['timesteps']:>10,}")
    else:
        # Show all tickers with versions
        from rl_strategy.agent.train import _load_registry
        registry = _load_registry()
        if not registry:
            print("No model versions found\n")
            return
        for t, data in sorted(registry.items()):
            versions = data.get("versions", [])
            if versions:
                latest = versions[0]
                print(f"  {t:<8} {len(versions)} versions | Latest: "
                      f"return={latest['mean_return']:.2f}, "
                      f"v={latest['version']}")
    print()


def cmd_rollback(ticker: str, version: str = None):
    """Rollback to a previous model version."""
    from rl_strategy.agent.train import rollback_model, list_model_versions
    
    print("\n" + "="*60)
    print(f"RL STRATEGY - ROLLBACK: {ticker}")
    print("="*60 + "\n")
    
    # Show available versions
    versions = list_model_versions(ticker)
    if versions:
        print("Available versions:")
        for i, v in enumerate(versions):
            marker = " <-- current" if i == 0 else ""
            print(f"  [{i}] {v['version']}  return={v['mean_return']:.4f}  trades={v['mean_trades']:.1f}{marker}")
        print()
    
    result = rollback_model(ticker, version)
    if result:
        print(f"\n[OK] Model rolled back successfully\n")
    else:
        print(f"\n[X] Rollback failed\n")


def cmd_tune(ticker: str, timesteps: int = 30000, eval_episodes: int = 3):
    """Run hyperparameter tuning."""
    from rl_strategy.agent.tune import run_grid_search
    
    run_grid_search(
        ticker=ticker,
        timesteps=timesteps,
        eval_episodes=eval_episodes,
    )


def cmd_retrain(ticker: str = None, all_tickers: bool = False, auto: bool = False, retrain_timesteps: int = 50000):
    """Check for model degradation and optionally retrain."""
    from rl_strategy.agent.retrain import check_degradation, check_all_tickers
    
    if ticker:
        should_retrain, reason, diagnostics = check_degradation(ticker)
        print(f"\n{ticker}: {'DEGRADED' if should_retrain else 'OK'} — {reason}")
        print(f"  Win rate: {diagnostics.get('win_rate', 0):.1%}")
        print(f"  Total P&L: ${diagnostics.get('total_pnl', 0):.2f}")
        print(f"  Total trades: {diagnostics.get('total_trades', 0)}")
        if should_retrain and auto:
            print(f"\n  Auto-retraining {ticker}...")
            from rl_strategy.agent.train import train_single_ticker
            train_single_ticker(ticker, timesteps=retrain_timesteps)
    elif all_tickers:
        check_all_tickers(auto_retrain=auto, retrain_timesteps=retrain_timesteps)
    else:
        print("Specify --ticker or --all")


def cmd_portfolio():
    """Show portfolio status with position limits."""
    from rl_strategy.portfolio.position_manager import PositionManager
    
    pm = PositionManager()
    pm.print_status()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RL Trading Strategy CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rl_strategy.cli signals --ticker AAPL
  python -m rl_strategy.cli train --ticker AAPL --timesteps 100000
  python -m rl_strategy.cli train --all
  python -m rl_strategy.cli backtest --ticker AAPL
  python -m rl_strategy.cli paper
  python -m rl_strategy.cli positions
  python -m rl_strategy.cli portfolio
  python -m rl_strategy.cli versions
  python -m rl_strategy.cli rollback --ticker AAPL
  python -m rl_strategy.cli tune --ticker AAPL
  python -m rl_strategy.cli retrain --all
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # signals command
    signals_parser = subparsers.add_parser("signals", help="Generate trading signals")
    signals_parser.add_argument("--ticker", help="Ticker symbol (default: all)")
    signals_parser.add_argument("--all", action="store_true", help="All tickers")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train RL agent")
    train_parser.add_argument("--ticker", help="Ticker to train")
    train_parser.add_argument("--all", action="store_true", help="Train all tickers")
    train_parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    
    # backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--ticker", required=True, help="Ticker to backtest")
    backtest_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes (default: 10)")
    backtest_parser.add_argument("--train-split", type=float, default=0.3, help="Fraction for train/test split (0=no split, default: 0.3)")
    
    # paper command
    paper_parser = subparsers.add_parser("paper", help="Run paper trading job")
    
    # positions command
    positions_parser = subparsers.add_parser("positions", help="Show open positions")
    
    # results command
    results_parser = subparsers.add_parser("results", help="View evaluation results")
    results_parser.add_argument("--ticker", help="Show results for specific ticker")
    
    # versions command
    versions_parser = subparsers.add_parser("versions", help="List model versions")
    versions_parser.add_argument("--ticker", help="Show versions for specific ticker")
    
    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous model version")
    rollback_parser.add_argument("--ticker", required=True, help="Ticker to rollback")
    rollback_parser.add_argument("--version", help="Specific version timestamp (default: previous)")
    
    # tune command
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tuning")
    tune_parser.add_argument("--ticker", required=True, help="Ticker to tune")
    tune_parser.add_argument("--timesteps", type=int, default=30000, help="Timesteps per trial")
    tune_parser.add_argument("--eval-episodes", type=int, default=3, help="Eval episodes per trial")
    
    # retrain command
    retrain_parser = subparsers.add_parser("retrain", help="Check degradation / auto-retrain")
    retrain_parser.add_argument("--ticker", help="Check specific ticker")
    retrain_parser.add_argument("--all", action="store_true", help="Check all tickers")
    retrain_parser.add_argument("--auto", action="store_true", help="Auto-retrain degraded models")
    retrain_parser.add_argument("--retrain-timesteps", type=int, default=50000, help="Timesteps for retraining")
    
    # portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", help="Show portfolio status with limits")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Dispatch commands
    if args.command == "signals":
        cmd_signals(ticker=args.ticker, all_tickers=args.all)
    elif args.command == "train":
        cmd_train(ticker=args.ticker, all_tickers=args.all, timesteps=args.timesteps)
    elif args.command == "backtest":
        cmd_backtest(ticker=args.ticker, episodes=args.episodes, train_split=args.train_split)
    elif args.command == "paper":
        cmd_paper()
    elif args.command == "positions":
        cmd_positions()
    elif args.command == "results":
        cmd_results(ticker=args.ticker)
    elif args.command == "versions":
        cmd_versions(ticker=args.ticker)
    elif args.command == "rollback":
        cmd_rollback(ticker=args.ticker, version=args.version)
    elif args.command == "tune":
        cmd_tune(ticker=args.ticker, timesteps=args.timesteps, eval_episodes=args.eval_episodes)
    elif args.command == "retrain":
        cmd_retrain(ticker=args.ticker, all_tickers=args.all, auto=args.auto, retrain_timesteps=args.retrain_timesteps)
    elif args.command == "portfolio":
        cmd_portfolio()


if __name__ == "__main__":
    main()
