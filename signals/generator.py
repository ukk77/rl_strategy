"""Signal Generator for RL Strategy.

Converts agent actions to standardized Signal objects matching MR/TF format.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..agent.env import TradingEnv
from ..agent.model import RLAgent


@dataclass
class Signal:
    """Trading signal output matching existing strategy format."""
    ticker: str
    timestamp: datetime
    action: str  # BUY, SELL, HOLD
    confidence: float
    shares: Optional[float] = None
    price: Optional[float] = None
    position_before: float = 0.0
    position_after: Optional[float] = None
    portfolio_value: Optional[float] = None
    features: Optional[Dict] = None
    
    def __str__(self) -> str:
        if self.action == "HOLD":
            return f"[{self.ticker}] HOLD (conf: {self.confidence:.2f})"
        price_str = f"${self.price:.2f}" if self.price else "N/A"
        shares_str = f"{self.shares:.2f}" if self.shares else "N/A"
        return f"[{self.ticker}] {self.action} {shares_str} shares @ {price_str} (conf: {self.confidence:.2f})"


class SignalGenerator:
    """Generate trading signals from RL agent predictions."""
    
    def __init__(self, agent: RLAgent):
        """Initialize with trained agent.
        
        Args:
            agent: Trained RLAgent instance
        """
        self.agent = agent
        self.ticker = agent.ticker
    
    def generate_signal(
        self,
        observation: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Signal:
        """Generate signal from current environment state.
        
        Args:
            observation: Optional observation (uses env current if None)
            deterministic: Use deterministic policy
            
        Returns:
            Signal object
        """
        env = self.agent.env
        
        # Get current observation if not provided
        if observation is None:
            observation = env._get_observation()
        
        # Get current state
        current_price = None
        portfolio_value = env.portfolio_value
        position_before = env.position_shares
        
        if env.current_step < len(env.prices):
            price_col = 'close' if 'close' in env.prices.columns else 'Close'
            current_price = float(env.prices.iloc[env.current_step][price_col])
        
        # Predict action
        action, _ = self.agent.predict(observation, deterministic=deterministic)
        
        # Map action to signal
        action_map = {
            0: "HOLD",
            1: "BUY",
            2: "BUY",
            3: "BUY",
            4: "SELL",
            5: "SELL",
            6: "SELL"
        }
        
        action_str = action_map[action]
        
        # Calculate shares based on action
        shares = None
        position_after = position_before
        
        if action_str == "BUY" and current_price:
            buy_fraction = {1: 0.25, 2: 0.50, 3: 1.0}[action]
            max_cash = env.cash * buy_fraction
            shares = max_cash / (current_price * 1.001)  # Include transaction cost
            position_after = position_before + shares
        elif action_str == "SELL" and position_before > 0:
            sell_fraction = {4: 0.25, 5: 0.50, 6: 1.0}[action]
            shares = position_before * sell_fraction
            position_after = position_before - shares
        
        # Confidence = policy's softmax probability for the chosen action
        try:
            import torch as th
            obs_t, _ = self.agent.model.policy.obs_to_tensor(observation)
            with th.no_grad():
                dist = self.agent.model.policy.get_distribution(obs_t)
                confidence = float(dist.distribution.probs[0, action])
        except Exception:
            confidence = 1.0 / 7  # uniform fallback (7 actions)
        
        # Build features dict for logging
        features = None
        if hasattr(env, 'features') and env.current_step < len(env.features):
            features = env.features.iloc[env.current_step].to_dict()
        
        signal = Signal(
            ticker=self.ticker,
            timestamp=datetime.now(),
            action=action_str,
            confidence=confidence,
            shares=shares,
            price=current_price,
            position_before=position_before,
            position_after=position_after,
            portfolio_value=portfolio_value,
            features=features
        )
        
        return signal
    
    def generate_signals_batch(
        self,
        steps: int = 1,
        execute: bool = False
    ) -> List[Signal]:
        """Generate multiple signals in sequence.
        
        Args:
            steps: Number of steps to generate
            execute: Whether to actually execute actions in environment
            
        Returns:
            List of Signal objects
        """
        signals = []
        env = self.agent.env
        
        for _ in range(steps):
            # Generate signal
            signal = self.generate_signal()
            signals.append(signal)
            
            # Execute in environment if requested
            if execute:
                action = self._signal_to_action(signal)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
        
        return signals
    
    def _signal_to_action(self, signal: Signal) -> int:
        """Convert signal back to action number.
        
        Args:
            signal: Signal object
            
        Returns:
            Action integer 0-6
        """
        if signal.action == "HOLD":
            return 0
        elif signal.action == "BUY":
            # Estimate based on shares
            env = self.agent.env
            if signal.shares and env.cash > 0:
                fraction = (signal.shares * signal.price) / env.cash if signal.price else 0
                if fraction >= 0.9:
                    return 3  # BUY 100%
                elif fraction >= 0.4:
                    return 2  # BUY 50%
                else:
                    return 1  # BUY 25%
            return 1
        elif signal.action == "SELL":
            # Estimate based on shares
            env = self.agent.env
            if signal.shares and env.position_shares > 0:
                fraction = signal.shares / env.position_shares
                if fraction >= 0.9:
                    return 6  # SELL 100%
                elif fraction >= 0.4:
                    return 5  # SELL 50%
                else:
                    return 4  # SELL 25%
            return 6
        return 0


def create_generator(ticker: str, model_path: Optional[str] = None) -> Optional[SignalGenerator]:
    """Factory function to create signal generator.
    
    Args:
        ticker: Stock symbol
        model_path: Optional path to trained model
        
    Returns:
        SignalGenerator or None
    """
    from ..agent.model import load_agent
    from ..agent.env import create_env_from_ticker
    
    # If model path provided, load it
    if model_path and Path(model_path).exists():
        try:
            agent = load_agent(ticker, model_path)
            return SignalGenerator(agent)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    
    # Otherwise create untrained agent
    env = create_env_from_ticker(ticker, lookback=100)
    if env is None:
        return None
    
    agent = RLAgent(env)
    return SignalGenerator(agent)


def test_generator():
    """Test signal generator."""
    print("Testing signal generator...")
    
    ticker = "AAPL"
    
    # Create generator with untrained agent
    print(f"\nCreating generator for {ticker}...")
    generator = create_generator(ticker)
    
    if generator is None:
        print("✗ Failed to create generator")
        return False
    
    print("✓ Generator created")
    
    # Generate signal
    signal = generator.generate_signal()
    print(f"✓ Generated signal: {signal}")
    
    # Generate batch
    print("\nGenerating batch of 3 signals...")
    signals = generator.generate_signals_batch(steps=3)
    for i, sig in enumerate(signals):
        print(f"  {i+1}. {sig}")
    
    print("\n✓ Signal generator tests passed!")
    return True


if __name__ == "__main__":
    test_generator()
