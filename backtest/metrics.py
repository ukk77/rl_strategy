"""RL-native backtest metrics.

Computes metrics specific to reinforcement learning agents:
  - Episode-level: return, Sharpe, max drawdown, action distribution
  - Trade-level: P&L distribution, win rate, holding periods
  - Aggregate: mean/median/std across episodes
  - Overfitting: train vs test divergence
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TradeRecord:
    """Single trade executed during backtest."""
    entry_step: int
    exit_step: int
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    action: str  # "BUY" or "SELL"


@dataclass
class EpisodeResult:
    """Results from a single backtest episode."""
    episode_id: int
    total_return: float
    total_return_pct: float
    num_trades: int
    num_steps: int
    final_portfolio_value: float
    max_drawdown_pct: float
    sharpe_ratio: float
    action_counts: Dict[int, int]  # action -> count
    action_pct: Dict[str, float]   # "HOLD", "BUY", "SELL" -> %
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


@dataclass
class BacktestMetrics:
    """Aggregated metrics across all episodes."""
    ticker: str
    num_episodes: int
    total_trades: int

    # Episode-level aggregates
    mean_return: float
    std_return: float
    median_return: float
    min_return: float
    max_return: float

    mean_sharpe: float
    std_sharpe: float

    mean_max_drawdown: float
    std_max_drawdown: float

    mean_num_trades: float
    mean_episode_length: float

    # Action distribution (averaged across episodes)
    hold_pct: float
    buy_pct: float
    sell_pct: float

    # Trade-level aggregates
    trade_win_rate: float
    mean_trade_pnl: float
    mean_trade_pnl_pct: float
    mean_holding_steps: float

    # Overfitting check (if train/test split)
    train_return: Optional[float] = None
    test_return: Optional[float] = None
    overfit_ratio: Optional[float] = None

    # Raw episode data
    episode_returns: List[float] = field(default_factory=list)
    episodes: List[EpisodeResult] = field(default_factory=list)


def compute_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio from per-step returns.

    Args:
        returns: List of per-step P&L values
        risk_free_rate: Annual risk-free rate (default 0 for paper trading)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize: hourly data -> ~252 trading days * 6.5 hours
    hourly_sharpe = mean_return / std_return
    annualized = hourly_sharpe * np.sqrt(252 * 6.5)

    return float(annualized)


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """Compute maximum drawdown as a percentage.

    Args:
        equity_curve: List of portfolio values over time

    Returns:
        Max drawdown as a negative percentage (e.g., -0.15 = -15%)
    """
    if not equity_curve:
        return 0.0

    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = np.min(drawdowns)

    return float(max_dd)


def compute_action_distribution(action_counts: Dict[int, int]) -> Dict[str, float]:
    """Convert raw action counts to HOLD/BUY/SELL percentages.

    Action mapping:
        0: HOLD
        1-3: BUY (25%, 50%, 100%)
        4-6: SELL (25%, 50%, 100%)
    """
    total = sum(action_counts.values())
    if total == 0:
        return {"HOLD": 100.0, "BUY": 0.0, "SELL": 0.0}

    hold = action_counts.get(0, 0)
    buy = sum(action_counts.get(a, 0) for a in [1, 2, 3])
    sell = sum(action_counts.get(a, 0) for a in [4, 5, 6])

    return {
        "HOLD": (hold / total) * 100,
        "BUY": (buy / total) * 100,
        "SELL": (sell / total) * 100,
    }


def aggregate_metrics(
    ticker: str,
    episodes: List[EpisodeResult],
    train_episodes: Optional[List[EpisodeResult]] = None,
) -> BacktestMetrics:
    """Aggregate episode-level results into summary metrics.

    Args:
        ticker: Stock symbol
        episodes: List of episode results (test set)
        train_episodes: Optional training episodes for overfitting check

    Returns:
        BacktestMetrics with aggregated statistics
    """
    if not episodes:
        raise ValueError("No episodes to aggregate")

    returns = [e.total_return for e in episodes]
    sharpes = [e.sharpe_ratio for e in episodes]
    drawdowns = [e.max_drawdown_pct for e in episodes]
    num_trades = [e.num_trades for e in episodes]
    lengths = [e.num_steps for e in episodes]

    # Aggregate action distribution
    total_actions = {}
    for e in episodes:
        for action, count in e.action_counts.items():
            total_actions[action] = total_actions.get(action, 0) + count
    action_pct = compute_action_distribution(total_actions)

    # Aggregate trade-level stats
    all_trades = []
    for e in episodes:
        all_trades.extend(e.trades)

    if all_trades:
        trade_pnls = [t.pnl for t in all_trades]
        trade_pnl_pcts = [t.pnl_pct for t in all_trades]
        holding_steps = [t.exit_step - t.entry_step for t in all_trades]
        winning_trades = [t for t in all_trades if t.pnl > 0]
        trade_win_rate = len(winning_trades) / len(all_trades)
        mean_trade_pnl = np.mean(trade_pnls)
        mean_trade_pnl_pct = np.mean(trade_pnl_pcts)
        mean_holding_steps = np.mean(holding_steps)
    else:
        trade_win_rate = 0.0
        mean_trade_pnl = 0.0
        mean_trade_pnl_pct = 0.0
        mean_holding_steps = 0.0

    # Overfitting check
    train_return = None
    test_return = None
    overfit_ratio = None
    if train_episodes:
        train_returns = [e.total_return for e in train_episodes]
        train_return = float(np.mean(train_returns))
        test_return = float(np.mean(returns))
        if abs(train_return) > 1e-8:
            overfit_ratio = test_return / train_return

    return BacktestMetrics(
        ticker=ticker,
        num_episodes=len(episodes),
        total_trades=len(all_trades),
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns, ddof=1)),
        median_return=float(np.median(returns)),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        mean_sharpe=float(np.mean(sharpes)),
        std_sharpe=float(np.std(sharpes, ddof=1)),
        mean_max_drawdown=float(np.mean(drawdowns)),
        std_max_drawdown=float(np.std(drawdowns, ddof=1)),
        mean_num_trades=float(np.mean(num_trades)),
        mean_episode_length=float(np.mean(lengths)),
        hold_pct=action_pct["HOLD"],
        buy_pct=action_pct["BUY"],
        sell_pct=action_pct["SELL"],
        trade_win_rate=trade_win_rate,
        mean_trade_pnl=mean_trade_pnl,
        mean_trade_pnl_pct=mean_trade_pnl_pct,
        mean_holding_steps=mean_holding_steps,
        train_return=train_return,
        test_return=test_return,
        overfit_ratio=overfit_ratio,
        episode_returns=returns,
        episodes=episodes,
    )
