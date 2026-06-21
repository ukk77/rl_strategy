"""FinRL Model Wrapper for PPO Agent.

Provides train, load, and predict functionality.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..config import get_config
from .env import TradingEnv


class TrainingCallback(BaseCallback):
    """Custom callback for logging training progress."""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Log training metrics
            mean_reward = np.mean(self.locals.get('rewards', [0]))
            print(f"Step {self.num_timesteps}: Mean reward = {mean_reward:.4f}")
        return True


class RLAgent:
    """Reinforcement Learning Agent wrapper."""
    
    def __init__(self, env: TradingEnv, model_path: Optional[str] = None):
        """Initialize RL agent.
        
        Args:
            env: Trading environment
            model_path: Optional path to load existing model
        """
        self.cfg = get_config()
        self.env = env
        self.ticker = env.ticker
        
        # Wrap environment for Stable Baselines3
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Initialize or load model
        if model_path and os.path.exists(model_path):
            self.model = self.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = self._create_model()
            print("Created new PPO model")
    
    def _create_model(self) -> PPO:
        """Create new PPO model with config hyperparameters."""
        model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.cfg.learning_rate,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_range=self.cfg.clip_range,
            verbose=1
            # tensorboard_log disabled - requires tensorboard package
        )
        return model
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_env: Optional[TradingEnv] = None,
        save_freq: int = 10000
    ) -> str:
        """Train the agent.
        
        Args:
            total_timesteps: Total training steps (default: config value)
            eval_env: Optional evaluation environment
            save_freq: Save checkpoint every N steps
            
        Returns:
            Path to saved model
        """
        if total_timesteps is None:
            total_timesteps = self.cfg.total_timesteps
        
        # Setup callbacks (simplified - no tensorboard)
        callbacks = [TrainingCallback(log_freq=5000)]
        # EvalCallback removed - requires tensorboard
        
        # Train
        print(f"\nTraining {self.ticker} for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        model_path = self._get_model_path()
        self.save(model_path)
        
        return model_path
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Optional[np.ndarray]]:
        """Predict action from observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, state)
        """
        action, state = self.model.predict(
            observation,
            deterministic=deterministic
        )
        return int(action), state
    
    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk.
        
        Args:
            path: Save path (default: models/{ticker}_ppo.zip)
            
        Returns:
            Path to saved model
        """
        if path is None:
            path = self._get_model_path()
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path)
        print(f"Model saved to {path}")
        return path
    
    def load(self, path: str) -> PPO:
        """Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded PPO model
        """
        model = PPO.load(path, env=self.vec_env)
        return model
    
    def _get_model_path(self) -> str:
        """Get default model save path."""
        return f"{self.cfg.models_dir}/{self.ticker}_ppo.zip"
    
    def evaluate(self, num_episodes: int = 5) -> dict:
        """Evaluate agent performance.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics dict
        """
        episode_returns = []
        episode_lengths = []
        episode_trades = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_trades.append(info.get('episode_trades', 0))
        
        metrics = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_length': np.mean(episode_lengths),
            'mean_trades': np.mean(episode_trades),
            'returns': episode_returns
        }
        
        return metrics


def train_agent(
    ticker: str,
    total_timesteps: int = 100000,
    save_path: Optional[str] = None
) -> Tuple[RLAgent, str]:
    """Train an RL agent for a specific ticker.
    
    Args:
        ticker: Stock symbol
        total_timesteps: Training timesteps
        save_path: Optional custom save path
        
    Returns:
        Tuple of (trained_agent, model_path)
    """
    from .env import create_env_from_ticker
    
    # Create environment
    env = create_env_from_ticker(ticker, lookback=1000)
    if env is None:
        raise ValueError(f"Could not create environment for {ticker}")
    
    # Create agent
    agent = RLAgent(env)
    
    # Train
    model_path = agent.train(total_timesteps=total_timesteps)
    
    # Evaluate
    print("\nEvaluating trained agent...")
    metrics = agent.evaluate(num_episodes=5)
    print(f"Mean return: {metrics['mean_return']:.4f}")
    print(f"Std return: {metrics['std_return']:.4f}")
    print(f"Mean trades per episode: {metrics['mean_trades']:.1f}")
    
    return agent, model_path


def load_agent(ticker: str, model_path: str) -> RLAgent:
    """Load a trained agent.
    
    Args:
        ticker: Stock symbol
        model_path: Path to saved model
        
    Returns:
        Loaded RLAgent
    """
    from .env import create_env_from_ticker
    
    # Create environment
    env = create_env_from_ticker(ticker, lookback=100)
    if env is None:
        raise ValueError(f"Could not create environment for {ticker}")
    
    # Load agent
    agent = RLAgent(env, model_path=model_path)
    
    return agent


def test_model():
    """Test model wrapper."""
    print("Testing model wrapper...")
    
    ticker = "AAPL"
    
    # Test training
    print(f"\nTraining agent for {ticker} (10,000 timesteps for test)...")
    agent, model_path = train_agent(ticker, total_timesteps=10000)
    
    print(f"\n✓ Model saved to {model_path}")
    
    # Test loading
    print("\nLoading saved model...")
    loaded_agent = load_agent(ticker, model_path)
    print("✓ Model loaded successfully")
    
    # Test prediction
    obs, info = loaded_agent.env.reset()
    action, _ = loaded_agent.predict(obs)
    print(f"✓ Prediction: action={action}")
    
    # Cleanup
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
    
    print("\n✓ All model tests passed!")


if __name__ == "__main__":
    test_model()
