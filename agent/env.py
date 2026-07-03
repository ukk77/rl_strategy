"""Custom Gymnasium Trading Environment for FinRL.

17-dim observation space with 7 discrete actions.
"""
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..config import get_config
from ..data.feature_engineering import FeatureEngineer


class TradingEnv(gym.Env):
    """Custom trading environment for reinforcement learning.
    
    Observation Space (17 dims):
        - Price features (10): returns, volatility, volume ratios, BB position, SMA ratio, momentum
        - Sentiment (3): avg_sentiment, confidence, direction
        - Risk (3): composite_risk_score, VaR, beta
        - Position (1): current position flag (0=no position, 1=holding)
    
    Action Space (7 discrete):
        0: HOLD
        1: BUY 25% of available cash
        2: BUY 50% of available cash
        3: BUY 100% of available cash
        4: SELL 25% of position
        5: SELL 50% of position
        6: SELL 100% of position
    
    Reward: Risk-adjusted return (PnL / volatility)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        ticker: str,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
    ):
        """Initialize trading environment.
        
        Args:
            ticker: Stock symbol
            features_df: DataFrame with 17-dim feature columns
            prices_df: DataFrame with Close prices
            initial_cash: Starting cash
            transaction_cost: Transaction cost as fraction
        """
        super().__init__()
        
        self.cfg = get_config()
        self.ticker = ticker
        self.features = features_df.copy()
        self.prices = prices_df.copy()
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        
        # Ensure features has all required columns
        self._validate_features()
        
        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)
        
        # Observation space: 17 dims (16 features + position)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),  # 16 price/sentiment/risk features + position
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.cash = initial_cash
        self.position_shares = 0.0
        self.portfolio_value = initial_cash
        self.prev_portfolio_value = initial_cash
        
        # Episode tracking
        self.episode_returns = []
        self.episode_trades = 0
        
    def _validate_features(self):
        """Validate that features dataframe has all required columns."""
        required_cols = [
            'return_1h', 'return_24h',
            'volatility_5h', 'volatility_20h',
            'volume_ratio_5h', 'volume_ratio_20h',
            'high_low_range', 'bb_position',
            'sma_ratio', 'momentum',
            'avg_sentiment', 'confidence', 'sentiment_direction',
            'composite_risk_score', 'var_95', 'beta'
        ]
        
        for col in required_cols:
            if col not in self.features.columns:
                print(f"Warning: Missing feature column {col}, filling with zeros")
                self.features[col] = 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current state.
        
        Returns:
            17-dim observation array
        """
        if self.current_step >= len(self.features):
            # End of data, return zeros
            return np.zeros(17, dtype=np.float32)
        
        # Get current features (17 dims)
        feature_cols = [
            'return_1h', 'return_24h',
            'volatility_5h', 'volatility_20h',
            'volume_ratio_5h', 'volume_ratio_20h',
            'high_low_range', 'bb_position',
            'sma_ratio', 'momentum',
            'avg_sentiment', 'confidence', 'sentiment_direction',
            'composite_risk_score', 'var_95', 'beta'
        ]
        
        features = self.features.iloc[self.current_step][feature_cols].values.astype(np.float32)
        
        # Add position flag (1 dim)
        position_flag = np.array([1.0 if self.position_shares > 0 else 0.0], dtype=np.float32)
        
        # Concatenate: 16 + 1 = 17 dims
        obs = np.concatenate([features, position_flag])
        
        # Clip to reasonable bounds
        obs = np.clip(obs, -10, 10)
        
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.cash = self.initial_cash
        self.position_shares = 0.0
        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        self.episode_returns = []
        self.episode_trades = 0
        
        # Random starting point for training variety (uses seeded np_random for reproducibility)
        if len(self.features) > 1000:
            self.current_step = int(self.np_random.integers(0, len(self.features) - 1000))
        
        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position_shares": self.position_shares,
            "step": self.current_step
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step.
        
        Args:
            action: Integer 0-6 representing trading action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current price
        if self.current_step >= len(self.prices):
            # End of data
            obs = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = {"portfolio_value": self.portfolio_value}
            return obs, reward, terminated, truncated, info
        
        # Handle both lowercase and capitalized column names
        price_col = 'close' if 'close' in self.prices.columns else 'Close'
        current_price = float(self.prices.iloc[self.current_step][price_col])
        
        # Execute action
        trade_value = 0.0
        
        if action == 0:
            # HOLD - no action
            pass
            
        elif action in [1, 2, 3]:
            # BUY actions
            buy_fraction = {1: 0.25, 2: 0.50, 3: 1.0}[action]
            max_buy_cash = self.cash * buy_fraction
            shares_to_buy = max_buy_cash / (current_price * (1 + self.transaction_cost))
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.cash -= cost
                self.position_shares += shares_to_buy
                trade_value = cost
                self.episode_trades += 1
        
        elif action in [4, 5, 6] and self.position_shares > 0:
            # SELL actions
            sell_fraction = {4: 0.25, 5: 0.50, 6: 1.0}[action]
            shares_to_sell = self.position_shares * sell_fraction
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.position_shares -= shares_to_sell
                trade_value = proceeds
                self.episode_trades += 1
        
        # Update portfolio value
        position_value = self.position_shares * current_price
        self.portfolio_value = self.cash + position_value
        
        # Calculate reward: risk-adjusted return
        pnl = self.portfolio_value - self.prev_portfolio_value
        
        # Get volatility for normalization
        if self.current_step > 20 and self.current_step < len(self.features):
            volatility = max(
                self.features.iloc[self.current_step]['volatility_20h'],
                0.001  # Minimum volatility floor
            )
        else:
            volatility = 0.01  # Default volatility
        
        # Risk-adjusted reward (Sharpe-like) — normalise by current portfolio value
        reward = pnl / (volatility * max(self.prev_portfolio_value, 1.0))
        
        # Penalty for excessive trading
        if trade_value > 0:
            reward -= 0.001  # Small transaction penalty
        
        # Update tracking
        self.episode_returns.append(pnl)
        self.prev_portfolio_value = self.portfolio_value
        
        # Advance step
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= len(self.features) - 1
        truncated = False
        
        # Get new observation
        obs = self._get_observation()
        
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position_shares": self.position_shares,
            "position_value": position_value,
            "current_price": current_price,
            "step": self.current_step,
            "trade_executed": trade_value > 0,
            "episode_trades": self.episode_trades
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """Render current state."""
        if self.current_step < len(self.prices):
            price_col = 'close' if 'close' in self.prices.columns else 'Close'
            current_price = self.prices.iloc[self.current_step][price_col]
            print(f"Step {self.current_step}: Price=${current_price:.2f}, "
                  f"Cash=${self.cash:.2f}, Position={self.position_shares:.2f} shares, "
                  f"Portfolio=${self.portfolio_value:.2f}")


def create_env_from_ticker(
    ticker: str,
    lookback: int = 1000,
    initial_cash: float = 100000.0
) -> Optional[TradingEnv]:
    """Factory function to create environment from ticker.
    
    Args:
        ticker: Stock symbol
        lookback: Number of historical periods
        initial_cash: Starting cash
        
    Returns:
        TradingEnv instance or None if data unavailable
    """
    engineer = FeatureEngineer()
    
    # Load data
    ohlc = engineer.load_ohlcv(ticker)
    if ohlc is None or ohlc.empty:
        print(f"No OHLCV data for {ticker}")
        return None
    
    sentiment = engineer.fetch_sentiment_from_db(ticker, limit=lookback)
    risk = engineer.fetch_risk_from_db(ticker, limit=lookback)
    
    # Build features
    features = engineer.build_features(ticker, ohlc, sentiment, risk)
    
    if features.empty:
        print(f"No features built for {ticker}")
        return None
    
    # Use only last N periods
    if len(features) > lookback:
        features = features.iloc[-lookback:]
        ohlc = ohlc.iloc[-lookback:]
    
    # Create environment
    env = TradingEnv(
        ticker=ticker,
        features_df=features,
        prices_df=ohlc,
        initial_cash=initial_cash
    )
    
    return env


def test_environment():
    """Test the trading environment."""
    print("Testing trading environment...")
    
    ticker = "AAPL"
    env = create_env_from_ticker(ticker, lookback=100)
    
    if env is None:
        print("✗ Failed to create environment")
        return False
    
    print(f"✓ Created environment for {ticker}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset environment")
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Initial portfolio: ${info['portfolio_value']:.2f}")
    
    # Test step
    obs, reward, terminated, truncated, info = env.step(3)  # BUY 100%
    print(f"✓ Executed BUY action")
    print(f"  Reward: {reward:.6f}")
    print(f"  Portfolio: ${info['portfolio_value']:.2f}")
    print(f"  Position: {info['position_shares']:.2f} shares")
    
    # Test a few more steps
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(0)  # HOLD
        if terminated:
            break
    
    print(f"✓ Ran 5 HOLD steps")
    
    # Test SELL
    if info['position_shares'] > 0:
        obs, reward, terminated, truncated, info = env.step(6)  # SELL 100%
        print(f"✓ Executed SELL action")
        print(f"  Final cash: ${info['cash']:.2f}")
        print(f"  Final position: {info['position_shares']:.2f} shares")
    
    print("\n✓ All environment tests passed!")
    return True


if __name__ == "__main__":
    test_environment()
