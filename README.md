# RL Strategy - Reinforcement Learning Trading

A FinRL-based trading strategy that runs hourly alongside existing mean_reversion and trend_following strategies.

## Overview

This service uses Proximal Policy Optimization (PPO) to learn optimal trading strategies from market data, sentiment, and risk signals combined.

## Features

- **17-dim Observation Space**: Price features, sentiment, risk metrics, position
- **7 Action Space**: HOLD, BUY 25/50/100%, SELL 25/50/100%
- **Risk-Adjusted Rewards**: Sharpe-like reward function
- **Hourly Execution**: Automated signal generation
- **Paper Trading**: Full position tracking with SQLite storage

## Installation

```bash
cd rl_strategy
pip install -r requirements.txt
```

## Quick Start

### 1. Train Agent

```bash
# Train single ticker
python -m rl_strategy.cli train --ticker AAPL --timesteps 100000

# Train all configured tickers
python -m rl_strategy.cli train --all
```

### 2. Generate Signals

```bash
# Current signals
python -m rl_strategy.cli signals --ticker AAPL

# All tickers
python -m rl_strategy.cli signals --all
```

### 3. Backtest

```bash
python -m rl_strategy.cli backtest --ticker AAPL --episodes 5
```

### 4. Paper Trading

```bash
# Manual run
python -m rl_strategy.cli paper

# Schedule hourly (Windows)
python -m rl_strategy.scheduler.hourly_job --schedule
```

### 5. View Positions

```bash
python -m rl_strategy.cli positions
```

## Project Structure

```
rl_strategy/
├── agent/              # RL agent (env, model, training)
├── data/               # Feature engineering
├── signals/            # Signal generation
├── scheduler/          # Hourly job scheduling
├── paper_trading/      # Position tracking & DB
├── backtest/           # Backtesting (Phase 6)
├── models/             # Trained model storage
├── config.py           # Hyperparameters
├── cli.py              # Command line interface
└── requirements.txt    # Dependencies
```

## Configuration

Edit `config.py` to adjust:

- Tickers to trade
- PPO hyperparameters (learning rate, gamma, etc.)
- Training timesteps
- Data source paths

## Architecture

### Data Flow

```
market_data/ (hourly parquet)
    ↓
feature_engineering.py (merge OHLCV + sentiment + risk)
    ↓
TradingEnv (17-dim observation)
    ↓
PPO Agent → Action (0-6)
    ↓
SignalGenerator → BUY/SELL/HOLD
    ↓
PaperTradingTracker → Execute → SQLite DB
```

### Integration Points

| Source | Purpose |
|--------|---------|
| `market_data/hourly/` | OHLCV price data |
| `sentiment_history.db` | Sentiment snapshots |
| `risk_history.db` | Risk metrics |
| API fallbacks | HTTP endpoints for latest data |

## Development Phases

| Phase | Duration | Status |
|-------|----------|--------|
| 1. Foundation | Week 1 | ✅ Complete |
| 2. RL Environment | Week 2 | ✅ Complete |
| 3. Training Pipeline | Week 3 | Ready |
| 4. Signal Generation | Week 4 | ✅ Complete |
| 5. Docker/Scheduling | Week 5 | Pending |
| 6. Backtesting | Week 6 | Pending |
| 7. Live Trading | Weeks 7-8 | Pending |

## CLI Commands

| Command | Description |
|---------|-------------|
| `signals [--ticker] [--all]` | Generate current signals |
| `train [--ticker] [--all] [--timesteps]` | Train RL agent |
| `backtest --ticker [--episodes]` | Run backtest |
| `paper` | Execute paper trading job |
| `positions` | Show open positions |

## Testing

```bash
# Test feature engineering
python -m rl_strategy.data.feature_engineering

# Test environment
python -m rl_strategy.agent.env

# Test model
python -m rl_strategy.agent.model

# Test database
python -m rl_strategy.paper_trading.db
```

## License

Private - Trading Platform Component
