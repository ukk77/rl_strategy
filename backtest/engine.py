"""Episode-based backtest engine for RL strategy.

Runs trained RL agents through historical data episodes, tracking:
  - Per-trade entry/exit prices, P&L, holding periods
  - Equity curves per episode
  - Action distributions
  - Train/test split for overfitting detection

Outputs RL-native metrics only — cross-strategy comparison is handled by the harness.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Setup paths
_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from rl_strategy.agent.env import create_env_from_ticker
from rl_strategy.config import get_config
from rl_strategy.backtest.metrics import (
    TradeRecord,
    EpisodeResult,
    BacktestMetrics,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_action_distribution,
    aggregate_metrics,
)


class BacktestEngine:
    """Episode-based backtest runner for RL agents.

    Runs the trained agent through historical data, tracking detailed
    trade-level metrics and RL-native performance statistics.
    """

    def __init__(self, ticker: str, model_path: str, lookback: int = 2000):
        """Initialize backtest engine.

        Args:
            ticker: Stock symbol
            model_path: Path to trained model (.zip)
            lookback: Number of historical periods for backtest data
        """
        self.ticker = ticker
        self.model_path = model_path
        self.cfg = get_config()

        # Create environment with full lookback for backtesting
        env = create_env_from_ticker(ticker, lookback=lookback)
        if env is None:
            raise ValueError(f"Could not create environment for {ticker}")

        # Load the trained model into this environment
        from rl_strategy.agent.model import RLAgent
        self.agent = RLAgent(env, model_path=model_path)
        self.env = env

    def run(
        self,
        num_episodes: int = 10,
        train_split: float = 0.0,
        seed: int = 42,
    ) -> BacktestMetrics:
        """Run backtest episodes.

        Args:
            num_episodes: Number of evaluation episodes
            train_split: Fraction of data for training episodes (0 = no split).
                         If > 0, runs train_split * num_episodes on early data
                         and the rest on later data for overfitting detection.
            seed: Random seed for reproducibility

        Returns:
            Aggregated BacktestMetrics
        """
        np.random.seed(seed)

        if train_split > 0:
            train_episodes_count = max(1, int(num_episodes * train_split))
            test_episodes_count = num_episodes - train_episodes_count

            # Train episodes: use first portion of data
            train_episodes = self._run_episodes(
                count=train_episodes_count,
                data_slice=(0.0, train_split),
                seed=seed,
            )

            # Test episodes: use later portion of data
            test_episodes = self._run_episodes(
                count=test_episodes_count,
                data_slice=(train_split, 1.0),
                seed=seed + 1000,
            )

            metrics = aggregate_metrics(
                ticker=self.ticker,
                episodes=test_episodes,
                train_episodes=train_episodes,
            )
        else:
            episodes = self._run_episodes(
                count=num_episodes,
                data_slice=(0.0, 1.0),
                seed=seed,
            )
            metrics = aggregate_metrics(
                ticker=self.ticker,
                episodes=episodes,
            )

        return metrics

    def _run_episodes(
        self,
        count: int,
        data_slice: Tuple[float, float],
        seed: int,
    ) -> List[EpisodeResult]:
        """Run multiple episodes on a slice of the data.

        Args:
            count: Number of episodes
            data_slice: (start_pct, end_pct) of data to use
            seed: Random seed

        Returns:
            List of EpisodeResult
        """
        episodes = []

        for ep in range(count):
            np.random.seed(seed + ep * 100)
            result = self._run_single_episode(
                episode_id=ep,
                data_slice=data_slice,
            )
            episodes.append(result)

        return episodes

    def _run_single_episode(
        self,
        episode_id: int,
        data_slice: Tuple[float, float],
    ) -> EpisodeResult:
        """Run a single episode with detailed trade tracking.

        Args:
            episode_id: Episode number
            data_slice: (start_pct, end_pct) of data to use

        Returns:
            EpisodeResult with trade details
        """
        # Determine data range for this slice
        total_steps = len(self.env.features)
        slice_start = int(total_steps * data_slice[0])
        slice_end = int(total_steps * data_slice[1])

        if slice_end - slice_start < 100:
            # Not enough data, use full range
            slice_start = 0
            slice_end = total_steps

        # Pick a random starting point within the slice for episode variety
        # Leave at least 100 steps for the episode to run
        max_start = max(slice_start, slice_end - 100)
        if max_start > slice_start:
            start_step = np.random.randint(slice_start, max_start)
        else:
            start_step = slice_start

        # Reset environment to start of slice
        obs, info = self.env.reset()
        self.env.current_step = start_step

        # Tracking variables
        step_returns = []
        equity_curve = [self.env.portfolio_value]
        action_counts = {i: 0 for i in range(7)}
        trades: List[TradeRecord] = []

        # Trade tracking state
        pending_entry_price = None
        pending_entry_step = None
        pending_shares = None
        prev_shares = 0.0

        done = False
        step_count = 0

        while not done and self.env.current_step < slice_end:
            # Get current price
            price_col = 'close' if 'close' in self.env.prices.columns else 'Close'
            current_price = float(
                self.env.prices.iloc[self.env.current_step][price_col]
            )

            # Predict and execute action
            action, _ = self.agent.predict(obs, deterministic=True)
            action_counts[action] = action_counts.get(action, 0) + 1

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step_count += 1

            # Track returns and equity
            step_returns.append(reward)
            equity_curve.append(info["portfolio_value"])

            # Track trades
            current_shares = info["position_shares"]

            if current_shares > prev_shares:
                # BUY executed — record entry
                shares_bought = current_shares - prev_shares
                if pending_entry_price is None:
                    # New position opened
                    pending_entry_price = current_price
                    pending_entry_step = self.env.current_step - 1
                    pending_shares = shares_bought
                else:
                    # Adding to existing position — average up entry
                    total_shares = pending_shares + shares_bought
                    pending_entry_price = (
                        (pending_entry_price * pending_shares) +
                        (current_price * shares_bought)
                    ) / total_shares
                    pending_shares = total_shares

            elif current_shares < prev_shares and pending_entry_price is not None:
                # SELL executed — record trade
                shares_sold = prev_shares - current_shares
                pnl = (current_price - pending_entry_price) * shares_sold
                pnl_pct = (current_price / pending_entry_price - 1) * 100

                trades.append(TradeRecord(
                    entry_step=pending_entry_step,
                    exit_step=self.env.current_step - 1,
                    entry_price=pending_entry_price,
                    exit_price=current_price,
                    shares=shares_sold,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    action="SELL",
                ))

                # Reset or adjust pending position
                if current_shares == 0:
                    pending_entry_price = None
                    pending_entry_step = None
                    pending_shares = None
                else:
                    pending_shares = current_shares

            prev_shares = current_shares

        # If still holding at end, mark-to-market as a trade
        if pending_entry_price is not None and pending_shares is not None and pending_shares > 0:
            price_col = 'close' if 'close' in self.env.prices.columns else 'Close'
            final_price = float(
                self.env.prices.iloc[min(self.env.current_step, len(self.env.prices) - 1)][price_col]
            )
            pnl = (final_price - pending_entry_price) * pending_shares
            pnl_pct = (final_price / pending_entry_price - 1) * 100

            trades.append(TradeRecord(
                entry_step=pending_entry_step,
                exit_step=self.env.current_step,
                entry_price=pending_entry_price,
                exit_price=final_price,
                shares=pending_shares,
                pnl=pnl,
                pnl_pct=pnl_pct,
                action="SELL",
            ))

        # Compute episode metrics
        total_return = self.env.portfolio_value - self.env.initial_cash
        total_return_pct = (total_return / self.env.initial_cash) * 100
        sharpe = compute_sharpe_ratio(step_returns) if step_returns else 0.0
        max_dd = compute_max_drawdown(equity_curve)
        action_pct = compute_action_distribution(action_counts)

        return EpisodeResult(
            episode_id=episode_id,
            total_return=total_return,
            total_return_pct=total_return_pct,
            num_trades=len(trades),
            num_steps=step_count,
            final_portfolio_value=self.env.portfolio_value,
            max_drawdown_pct=max_dd * 100,
            sharpe_ratio=sharpe,
            action_counts=action_counts,
            action_pct=action_pct,
            trades=trades,
            equity_curve=equity_curve,
        )

    def save_results(self, metrics: BacktestMetrics, output_dir: str = "results") -> str:
        """Save backtest results to JSON.

        Args:
            metrics: Aggregated backtest metrics
            output_dir: Directory for output files

        Returns:
            Path to saved file
        """
        results_dir = Path(output_dir)
        results_dir.mkdir(exist_ok=True)

        result = {
            "ticker": metrics.ticker,
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "num_episodes": metrics.num_episodes,
            "total_trades": metrics.total_trades,
            "episode_metrics": {
                "mean_return": metrics.mean_return,
                "std_return": metrics.std_return,
                "median_return": metrics.median_return,
                "min_return": metrics.min_return,
                "max_return": metrics.max_return,
                "mean_sharpe": metrics.mean_sharpe,
                "std_sharpe": metrics.std_sharpe,
                "mean_max_drawdown_pct": metrics.mean_max_drawdown,
                "std_max_drawdown_pct": metrics.std_max_drawdown,
                "mean_num_trades": metrics.mean_num_trades,
                "mean_episode_length": metrics.mean_episode_length,
                "episode_returns": metrics.episode_returns,
            },
            "action_distribution": {
                "hold_pct": metrics.hold_pct,
                "buy_pct": metrics.buy_pct,
                "sell_pct": metrics.sell_pct,
            },
            "trade_metrics": {
                "win_rate": metrics.trade_win_rate,
                "mean_trade_pnl": metrics.mean_trade_pnl,
                "mean_trade_pnl_pct": metrics.mean_trade_pnl_pct,
                "mean_holding_steps": metrics.mean_holding_steps,
            },
        }

        if metrics.overfit_ratio is not None:
            result["overfitting_check"] = {
                "train_return": metrics.train_return,
                "test_return": metrics.test_return,
                "overfit_ratio": metrics.overfit_ratio,
                "verdict": "OK" if metrics.overfit_ratio > 0.5 else "OVERFITTING — test return < 50% of train",
            }

        # Per-episode trade details
        result["episodes"] = []
        for ep in metrics.episodes:
            ep_data = {
                "episode_id": ep.episode_id,
                "total_return": ep.total_return,
                "total_return_pct": ep.total_return_pct,
                "num_trades": ep.num_trades,
                "num_steps": ep.num_steps,
                "max_drawdown_pct": ep.max_drawdown_pct,
                "sharpe_ratio": ep.sharpe_ratio,
                "action_pct": ep.action_pct,
                "trades": [
                    {
                        "entry_step": t.entry_step,
                        "exit_step": t.exit_step,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "shares": t.shares,
                        "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct,
                    }
                    for t in ep.trades
                ],
            }
            result["episodes"].append(ep_data)

        output_path = results_dir / f"{metrics.ticker}_backtest.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        return str(output_path)


def run_backtest(
    ticker: str,
    model_path: Optional[str] = None,
    num_episodes: int = 10,
    train_split: float = 0.3,
    save: bool = True,
) -> BacktestMetrics:
    """Convenience function to run a full backtest.

    Args:
        ticker: Stock symbol
        model_path: Path to model (default: models/{ticker}_ppo.zip)
        num_episodes: Number of evaluation episodes
        train_split: Fraction for train/test split (0 = no split)
        save: Whether to save results to JSON

    Returns:
        Aggregated BacktestMetrics
    """
    cfg = get_config()

    if model_path is None:
        model_path = f"{cfg.models_dir}/{ticker}_ppo.zip"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    engine = BacktestEngine(ticker, model_path)
    metrics = engine.run(
        num_episodes=num_episodes,
        train_split=train_split,
    )

    if save:
        output_path = engine.save_results(metrics)
        print(f"\nBacktest results saved to: {output_path}")

    return metrics
