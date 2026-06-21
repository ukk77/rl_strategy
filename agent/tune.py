"""Hyperparameter tuning via grid search for RL agents.

Tests combinations of learning_rate, gamma, n_steps, and batch_size
to find optimal PPO hyperparameters for a given ticker.
"""
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from rl_strategy.agent.env import create_env_from_ticker
from rl_strategy.agent.model import RLAgent
from rl_strategy.config import get_config, RLConfig


# Default grid search space
DEFAULT_GRID = {
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "gamma": [0.95, 0.99, 0.995],
    "n_steps": [1024, 2048],
    "batch_size": [32, 64],
}


def _create_config_override(overrides: Dict) -> RLConfig:
    """Create a config with specific hyperparameter overrides."""
    cfg = get_config()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _train_and_evaluate(
    ticker: str,
    overrides: Dict,
    timesteps: int,
    eval_episodes: int,
) -> Optional[Dict]:
    """Train and evaluate a model with given hyperparameters.

    Args:
        ticker: Stock symbol
        overrides: Hyperparameter overrides
        timesteps: Training timesteps
        eval_episodes: Number of evaluation episodes

    Returns:
        Dict with metrics or None on failure
    """
    try:
        # Create environment
        env = create_env_from_ticker(ticker, lookback=1000)
        if env is None:
            return None

        # Apply overrides to env's config
        for key, value in overrides.items():
            if hasattr(env.cfg, key):
                setattr(env.cfg, key, value)

        # Create and train agent
        agent = RLAgent(env)
        agent.train(total_timesteps=timesteps)

        # Evaluate
        metrics = agent.evaluate(num_episodes=eval_episodes)

        return {
            "mean_return": float(metrics["mean_return"]),
            "std_return": float(metrics["std_return"]),
            "mean_trades": float(metrics["mean_trades"]),
            "mean_length": float(metrics["mean_length"]),
        }
    except Exception as e:
        print(f"  [X] Failed: {e}")
        return None


def run_grid_search(
    ticker: str,
    grid: Optional[Dict[str, List]] = None,
    timesteps: int = 30000,
    eval_episodes: int = 3,
    top_n: int = 5,
) -> List[Dict]:
    """Run hyperparameter grid search for a ticker.

    Args:
        ticker: Stock symbol
        grid: Dict of parameter -> list of values (default: DEFAULT_GRID)
        timesteps: Training timesteps per trial (use fewer for speed)
        eval_episodes: Evaluation episodes per trial
        top_n: Number of top results to return

    Returns:
        List of result dicts sorted by mean_return descending
    """
    if grid is None:
        grid = DEFAULT_GRID

    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(product(*param_values))

    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: {ticker}")
    print(f"{'='*60}")
    print(f"Grid size: {len(combinations)} combinations")
    print(f"Parameters: {', '.join(param_names)}")
    print(f"Timesteps/trial: {timesteps:,}")
    print()

    results = []
    best_return = -float("inf")

    for i, combo in enumerate(combinations):
        overrides = dict(zip(param_names, combo))
        combo_str = ", ".join(f"{k}={v}" for k, v in overrides.items())

        print(f"[{i+1}/{len(combinations)}] {combo_str}")

        metrics = _train_and_evaluate(
            ticker=ticker,
            overrides=overrides,
            timesteps=timesteps,
            eval_episodes=eval_episodes,
        )

        if metrics is None:
            continue

        result = {
            "trial": i + 1,
            **overrides,
            **metrics,
        }
        results.append(result)

        if metrics["mean_return"] > best_return:
            best_return = metrics["mean_return"]
            print(f"  [NEW BEST] Mean return: {metrics['mean_return']:.4f}")
        else:
            print(f"  Mean return: {metrics['mean_return']:.4f}")

    # Sort by mean_return descending
    results.sort(key=lambda r: r["mean_return"], reverse=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TOP {min(top_n, len(results))} RESULTS")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'LR':<10} {'Gamma':<8} {'n_steps':<8} {'Batch':<8} {'Return':>12} {'Std':>10}")
    print("-" * 70)

    for rank, r in enumerate(results[:top_n]):
        print(f"{rank+1:<6} {r['learning_rate']:<10.0e} {r['gamma']:<8.3f} "
              f"{r['n_steps']:<8} {r['batch_size']:<8} {r['mean_return']:>12.2f} {r['std_return']:>10.2f}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"{ticker}_tuning.json"

    output = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "grid": {k: list(v) for k, v in grid.items()},
        "timesteps_per_trial": timesteps,
        "eval_episodes": eval_episodes,
        "total_trials": len(combinations),
        "completed_trials": len(results),
        "results": results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to: {output_path}")
    print(f"Best: lr={results[0]['learning_rate']}, gamma={results[0]['gamma']}, "
          f"n_steps={results[0]['n_steps']}, batch_size={results[0]['batch_size']}")
    print(f"Best mean return: {results[0]['mean_return']:.4f}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for RL agents")
    parser.add_argument("--ticker", required=True, help="Ticker to tune")
    parser.add_argument("--timesteps", type=int, default=30000, help="Timesteps per trial")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Eval episodes per trial")
    args = parser.parse_args()

    run_grid_search(
        ticker=args.ticker,
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
    )
