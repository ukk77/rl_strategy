"""Hourly Job Scheduler for RL Strategy.

Runs every hour to generate signals and execute paper trades.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Setup paths
def setup_paths():
    current_dir = Path(__file__).resolve().parent.parent
    trading_root = current_dir.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    if str(trading_root) not in sys.path:
        sys.path.insert(0, str(trading_root))


setup_paths()

from rl_strategy.agent.model import load_agent
from rl_strategy.config import get_config
from rl_strategy.paper_trading.tracker import PaperTradingTracker


def run_hourly():
    """Execute hourly paper trading job."""
    cfg = get_config()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"RL Strategy Hourly Job - {timestamp}")
    print(f"{'='*60}\n")
    
    results = []
    
    for ticker in cfg.tickers:
        print(f"Processing {ticker}...")
        
        # Load model
        model_path = f"{cfg.models_dir}/{ticker}_ppo.zip"
        
        if not Path(model_path).exists():
            print(f"  ✗ Model not found: {model_path}")
            print(f"  Skipping {ticker} (train first)")
            results.append({"ticker": ticker, "status": "NO_MODEL"})
            continue
        
        try:
            # Load agent
            agent = load_agent(ticker, model_path)
            print(f"  ✓ Loaded model")
            
            # Create tracker
            tracker = PaperTradingTracker(
                ticker=ticker,
                agent=agent,
                initial_cash=100000.0
            )
            
            # Generate and execute signal
            signal = tracker.generate_and_execute()
            
            if signal:
                print(f"  ✓ Signal: {signal}")
                results.append({
                    "ticker": ticker,
                    "status": "EXECUTED",
                    "action": signal.action,
                    "shares": signal.shares,
                    "price": signal.price,
                    "confidence": signal.confidence
                })
            else:
                print(f"  → HOLD")
                results.append({
                    "ticker": ticker,
                    "status": "HOLD"
                })
            
            # Record equity snapshot
            tracker.record_equity_snapshot()
            
            # Print position summary
            state = tracker.get_portfolio_state()
            print(f"  Position: {tracker.position.shares:.2f} shares, "
                  f"P&L: ${state.unrealized_pnl + state.realized_pnl:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({"ticker": ticker, "status": "ERROR", "error": str(e)})
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    executed = sum(1 for r in results if r["status"] == "EXECUTED")
    holds = sum(1 for r in results if r["status"] == "HOLD")
    errors = sum(1 for r in results if r["status"] in ["NO_MODEL", "ERROR"])
    
    print(f"  Executed: {executed}")
    print(f"  Hold:     {holds}")
    print(f"  Errors:   {errors}")
    print(f"{'='*60}\n")
    
    return results


def schedule_windows_task():
    """Create Windows Task Scheduler entry."""
    import subprocess
    
    script_path = Path(__file__).resolve()
    python_exe = sys.executable
    
    task_name = "RL_Strategy_Hourly"
    
    # Command to create task (runs every hour)
    cmd = [
        "schtasks",
        "/create",
        "/tn", task_name,
        "/tr", f'"{python_exe} {script_path}"',
        "/sc", "hourly",
        "/mo", "1",
        "/f"  # Force overwrite
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Scheduled task '{task_name}' created successfully")
            print(f"  Runs: Every hour")
            print(f"  Command: {python_exe} {script_path}")
        else:
            print(f"✗ Failed to create task: {result.stderr}")
    except Exception as e:
        print(f"✗ Error creating task: {e}")


def remove_windows_task():
    """Remove Windows Task Scheduler entry."""
    import subprocess
    
    task_name = "RL_Strategy_Hourly"
    
    cmd = ["schtasks", "/delete", "/tn", task_name, "/f"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Task '{task_name}' removed")
        else:
            print(f"✗ Failed to remove task: {result.stderr}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_hourly():
    """Test hourly job manually."""
    print("Testing hourly job...\n")
    
    # Run one cycle
    results = run_hourly()
    
    print(f"\nTest completed with {len(results)} tickers processed")


if __name__ == "__main__":
    # Check for command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--schedule":
            schedule_windows_task()
        elif sys.argv[1] == "--remove":
            remove_windows_task()
        elif sys.argv[1] == "--test":
            test_hourly()
        else:
            print("Usage: python hourly_job.py [--schedule|--remove|--test]")
    else:
        # Normal hourly run
        run_hourly()
