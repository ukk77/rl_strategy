"""Configuration for RL Strategy.

Hyperparameters and feature selection for FinRL PPO agent.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class RLConfig:
    """Master configuration for RL strategy."""
    
    # Tickers to trade (All available tickers with hourly data)
    tickers: List[str] = field(default_factory=lambda: [
        # Tech & Growth
        "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "AMZN", "GOOGL", "META", 
        "NFLX", "UBER", "PLTR", "ASML", "AVGO", "LITE", "MU", "NVTS", "SMCI",
        # Financial & Value
        "JPM", "V", "MA", "BRK.B", "XLF", "GS", "MS", "BLK",
        # Healthcare & Defensive
        "LLY", "UNH", "JNJ", "MRK", "XLV", "ABBV", "GILD",
        # Industrials & Materials
        "CAT", "BA", "LMT", "GE", "NUE", "XLB", "FCX", "MP", "RTX", "CAT",
        # Energy & Commodities
        "XOM", "VST", "GLD", "XLE", "EQT", "KMI", "WMB", "USAR", "UUUU",
        # Consumer & Retail
        "COST", "HD", "WMT", "MCD", "XLP", "BABA", "NB",
        # Crypto-Related
        "COIN", "MARA", "MSTR",
        # Index ETFs
        "SPY", "QQQ", "IWM", "TLT", "SQQQ", "VIX", "QQQ", "XLU", "XLRE", "XLK"
    ])
    
    # Observation space: 17 dimensions
    # Price features: 10
    price_feature_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Sentiment features: 3
    sentiment_enabled: bool = True
    sentiment_features: List[str] = field(default_factory=lambda: [
        "avg_sentiment", "confidence", "direction"
    ])
    
    # Risk features: 3
    risk_enabled: bool = True
    risk_features: List[str] = field(default_factory=lambda: [
        "composite_risk_score", "var_95", "beta"
    ])
    
    # Position feature: 1
    
    # Action space: 7 discrete actions
    # 0=HOLD, 1=BUY25%, 2=BUY50%, 3=BUY100%, 4=SELL25%, 5=SELL50%, 6=SELL100%
    
    # PPO Hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Training parameters
    total_timesteps: int = 100000
    eval_freq: int = 5000
    save_freq: int = 10000
    
    # Paths
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    # Data sources
    market_data_path: str = "../market_data/hourly"
    sentiment_db_path: str = "../sentiment_analysis/backend/sentiment_history.db"
    risk_db_path: str = "../risk_calculator/backend/risk_history.db"
    paper_trades_db: str = "rl_paper_trades.db"
    
    # API fallbacks
    sentiment_api_url: str = "http://localhost:8000"
    risk_api_url: str = "http://localhost:8100"
    
    # Scheduling
    run_hourly: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert len(self.tickers) > 0, "At least one ticker required"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.total_timesteps > 0, "Total timesteps must be positive"


# Singleton instance
_config_instance = None


def get_config() -> RLConfig:
    """Get or create config singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = RLConfig()
    return _config_instance
