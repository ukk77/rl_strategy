"""Training script for RL agents.

Command-line interface for training PPO agents on specific tickers.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path for imports
def setup_paths():
    """Setup import paths."""
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


setup_paths()

from rl_strategy.agent.env import create_env_from_ticker
from rl_strategy.agent.model import RLAgent
from rl_strategy.config import get_config


# ─── Model Versioning ─────────────────────────────────────────────────────────

MODEL_REGISTRY_PATH = "models/model_registry.json"
MAX_VERSIONS_PER_TICKER = 5


def _load_registry() -> Dict:
    """Load model registry from JSON."""
    registry_path = Path(MODEL_REGISTRY_PATH)
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f)
    return {}


def _save_registry(registry: Dict):
    """Save model registry to JSON."""
    registry_path = Path(MODEL_REGISTRY_PATH)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def save_versioned_model(ticker: str, model_path: str, metrics: Dict, timesteps: int):
    """Save a timestamped version of the model and update registry.

    Args:
        ticker: Stock symbol
        model_path: Path to the main model file
        metrics: Evaluation metrics dict
        timesteps: Training timesteps used
    """
    registry = _load_registry()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create versioned copy
    versioned_name = f"{ticker}_v{timestamp}.zip"
    versioned_path = Path(model_path).parent / versioned_name
    shutil.copy2(model_path, versioned_path)

    # Update registry
    if ticker not in registry:
        registry[ticker] = {"versions": [], "current": None}

    version_entry = {
        "version": timestamp,
        "path": str(versioned_path),
        "timesteps": timesteps,
        "mean_return": float(metrics["mean_return"]),
        "std_return": float(metrics["std_return"]),
        "mean_trades": float(metrics["mean_trades"]),
        "created": datetime.now().isoformat(),
    }

    registry[ticker]["versions"].insert(0, version_entry)
    registry[ticker]["current"] = str(model_path)

    # Keep only MAX_VERSIONS_PER_TICKER
    if len(registry[ticker]["versions"]) > MAX_VERSIONS_PER_TICKER:
        old_versions = registry[ticker]["versions"][MAX_VERSIONS_PER_TICKER:]
        for old in old_versions:
            old_path = Path(old["path"])
            if old_path.exists():
                old_path.unlink()
        registry[ticker]["versions"] = registry[ticker]["versions"][:MAX_VERSIONS_PER_TICKER]

    _save_registry(registry)
    print(f"[OK] Versioned model saved: {versioned_path}")


def list_model_versions(ticker: str) -> List[Dict]:
    """List all saved versions for a ticker.

    Args:
        ticker: Stock symbol

    Returns:
        List of version entries
    """
    registry = _load_registry()
    return registry.get(ticker, {}).get("versions", [])


def rollback_model(ticker: str, version: Optional[str] = None) -> Optional[str]:
    """Rollback to a previous model version.

    Args:
        ticker: Stock symbol
        version: Version timestamp to rollback to (default: previous version)

    Returns:
        Path to the restored model, or None if no versions available
    """
    registry = _load_registry()
    ticker_data = registry.get(ticker, {})
    versions = ticker_data.get("versions", [])

    if not versions:
        print(f"No versions available for {ticker}")
        return None

    if version:
        target = next((v for v in versions if v["version"] == version), None)
        if not target:
            print(f"Version {version} not found for {ticker}")
            return None
    else:
        # Rollback to previous version (index 1, since index 0 is current)
        if len(versions) < 2:
            print(f"Only one version available for {ticker}, cannot rollback")
            return None
        target = versions[1]

    # Copy versioned model over the current model
    cfg = get_config()
    current_path = Path(f"{cfg.models_dir}/{ticker}_ppo.zip")
    versioned_path = Path(target["path"])

    if not versioned_path.exists():
        print(f"Versioned model file not found: {versioned_path}")
        return None

    # Backup current before overwriting
    backup_path = current_path.with_suffix(".zip.bak")
    if current_path.exists():
        shutil.copy2(current_path, backup_path)

    shutil.copy2(versioned_path, current_path)

    # Update registry
    ticker_data["current"] = str(current_path)
    _save_registry(registry)

    print(f"[OK] Rolled back {ticker} to version {target['version']}")
    print(f"  Mean return: {target['mean_return']:.4f}")
    print(f"  Backup saved: {backup_path}")

    return str(current_path)


def train_single_ticker(ticker: str, timesteps: int = 100000) -> str:
    """Train agent for a single ticker.
    
    Args:
        ticker: Stock symbol
        timesteps: Total training timesteps
        
    Returns:
        Path to saved model
    """
    print(f"\n{'='*60}")
    print(f"Training RL Agent: {ticker}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = create_env_from_ticker(ticker, lookback=1000)
    if env is None:
        print(f"[X] Failed to create environment for {ticker}")
        return None
    
    print(f"[OK] Environment created")
    print(f"  Data points: {len(env.features)}")
    print(f"  Feature dims: {env.observation_space.shape[0]}")
    print(f"  Action space: {env.action_space.n} actions\n")
    
    # Create agent
    agent = RLAgent(env)
    
    # Train
    model_path = agent.train(total_timesteps=timesteps)
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating Trained Agent")
    print("="*60)
    
    metrics = agent.evaluate(num_episodes=5)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean episode return: {metrics['mean_return']:.4f} ± {metrics['std_return']:.4f}")
    print(f"  Mean episode length: {metrics['mean_length']:.1f} steps")
    print(f"  Mean trades/episode: {metrics['mean_trades']:.1f}")
    
    # Save versioned model
    save_versioned_model(ticker, model_path, metrics, timesteps)

    # Save evaluation results
    result = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "timesteps": timesteps,
        "metrics": {
            "mean_return": float(metrics['mean_return']),
            "std_return": float(metrics['std_return']),
            "mean_length": float(metrics['mean_length']),
            "mean_trades": float(metrics['mean_trades']),
            "returns": [float(r) for r in metrics['returns']]
        }
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    result_file = results_dir / f"{ticker}_evaluation.json"
    
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[OK] Model saved: {model_path}")
    print(f"[OK] Evaluation saved: {result_file}\n")
    
    return model_path


def train_all_tickers(timesteps: int = 100000):
    """Train agents for all configured tickers.
    
    Args:
        timesteps: Training timesteps per ticker
    """
    cfg = get_config()
    
    print(f"\nTraining agents for {len(cfg.tickers)} tickers")
    print(f"Tickers: {', '.join(cfg.tickers)}\n")
    
    results = {}
    
    for ticker in cfg.tickers:
        model_path = train_single_ticker(ticker, timesteps)
        results[ticker] = model_path
    
    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    for ticker, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {ticker}: {path if path else 'FAILED'}")
    
    return results


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train RL trading agents")
    parser.add_argument(
        "--ticker",
        type=str,
        help="Single ticker to train (e.g., AAPL)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all configured tickers"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        train_all_tickers(timesteps=args.timesteps)
    elif args.ticker:
        train_single_ticker(args.ticker, timesteps=args.timesteps)
    else:
        parser.print_help()
        print("\nError: Specify --ticker TICKER or --all")


if __name__ == "__main__":
    main()
