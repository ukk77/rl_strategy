"""Paper Trading Tracker for RL Strategy.

Manages positions, executes trades, and tracks P&L.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ..agent.model import RLAgent
from ..paper_trading.db import PaperTradingDB, Position
from ..signals.generator import Signal


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    positions_value: float
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float


class PaperTradingTracker:
    """Track paper trading positions and execute trades."""
    
    def __init__(
        self,
        ticker: str,
        agent: RLAgent,
        initial_cash: float = 100000.0,
        db_path: Optional[str] = None
    ):
        """Initialize tracker.
        
        Args:
            ticker: Stock symbol
            agent: Trained RL agent
            initial_cash: Starting cash
            db_path: Optional database path
        """
        self.ticker = ticker
        self.agent = agent
        self.initial_cash = initial_cash
        
        # Initialize database
        self.db = PaperTradingDB(db_path)
        
        # Load or create position
        self.position = self.db.get_position(ticker)
        if self.position is None:
            self.position = Position(
                ticker=ticker,
                shares=0.0,
                entry_price=None,
                entry_time=None,
                current_price=None,
                current_time=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
    
    def update_market_price(self, current_price: float):
        """Update current market price and recalculate P&L.
        
        Args:
            current_price: Current stock price
        """
        self.position.current_price = current_price
        self.position.current_time = datetime.now()
        
        # Calculate unrealized P&L
        if self.position.shares > 0 and self.position.entry_price:
            self.position.unrealized_pnl = (
                (current_price - self.position.entry_price) * self.position.shares
            )
        else:
            self.position.unrealized_pnl = 0.0
        
        # Update in database
        self.db.update_position(
            self.ticker,
            self.position.shares,
            entry_price=self.position.entry_price,
            current_price=current_price
        )
    
    def execute_signal(self, signal: Signal) -> bool:
        """Execute a trading signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            True if trade executed
        """
        if signal.action == "HOLD" or signal.price is None:
            return False
        
        price = signal.price
        pnl = 0.0
        
        if signal.action == "BUY" and signal.shares and signal.shares > 0:
            # Execute buy
            cost = signal.shares * price * 1.001  # Include transaction cost
            
            # Update position
            old_shares = self.position.shares
            self.position.shares += signal.shares
            
            # Calculate new average entry price
            if old_shares > 0:
                total_cost = (old_shares * self.position.entry_price) + cost
                self.position.entry_price = total_cost / self.position.shares
            else:
                self.position.entry_price = price
                self.position.entry_time = datetime.now()
            
        elif signal.action == "SELL" and signal.shares and signal.shares > 0:
            # Calculate P&L
            if self.position.entry_price:
                pnl = (price - self.position.entry_price) * signal.shares
            
            # Update realized P&L
            self.position.realized_pnl += pnl
            
            # Update position
            self.position.shares -= signal.shares
            
            # Clear entry price if fully closed
            if self.position.shares <= 0:
                self.position.shares = 0
                self.position.entry_price = None
                self.position.entry_time = None
        
        # Update market price
        self.update_market_price(price)
        
        # Record trade
        self.db.record_trade(
            ticker=self.ticker,
            action=signal.action,
            shares=signal.shares or 0,
            price=price,
            pnl=pnl if signal.action == "SELL" else None,
            signal_confidence=signal.confidence,
            features_snapshot=str(signal.features) if signal.features else None
        )
        
        return True
    
    def generate_and_execute(self) -> Optional[Signal]:
        """Generate signal and execute trade.
        
        Returns:
            Executed signal or None
        """
        from ..signals.generator import SignalGenerator
        
        # Generate signal
        generator = SignalGenerator(self.agent)
        signal = generator.generate_signal()
        
        # Execute if not HOLD
        if signal.action != "HOLD":
            self.execute_signal(signal)
            return signal
        
        # Still update market price even on HOLD
        if signal.price:
            self.update_market_price(signal.price)
        
        return None
    
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        positions_value = 0.0
        if self.position.shares > 0 and self.position.current_price:
            positions_value = self.position.shares * self.position.current_price
        
        # Calculate cash (not tracking exact cash separately, assuming full investment)
        # In real implementation, track cash separately
        cash = self.initial_cash - positions_value
        
        total_equity = cash + positions_value
        
        return PortfolioState(
            cash=cash,
            positions_value=positions_value,
            total_equity=total_equity,
            unrealized_pnl=self.position.unrealized_pnl,
            realized_pnl=self.position.realized_pnl
        )
    
    def record_equity_snapshot(self):
        """Record current equity to database."""
        state = self.get_portfolio_state()
        
        self.db.record_equity(
            total_equity=state.total_equity,
            cash_balance=state.cash,
            positions_value=state.positions_value,
            unrealized_pnl=state.unrealized_pnl,
            realized_pnl=state.realized_pnl
        )
    
    def get_trade_history(self, limit: int = 20) -> List[Dict]:
        """Get recent trade history."""
        trades = self.db.get_trades(self.ticker, limit=limit)
        return [
            {
                "time": t.timestamp.isoformat(),
                "action": t.action,
                "shares": t.shares,
                "price": t.price,
                "pnl": t.pnl,
                "confidence": t.signal_confidence
            }
            for t in trades
        ]
    
    def print_status(self):
        """Print current status."""
        state = self.get_portfolio_state()
        
        print(f"\n{'='*60}")
        print(f"RL Strategy - {self.ticker} Status")
        print(f"{'='*60}")
        print(f"Position:     {self.position.shares:.2f} shares")
        print(f"Entry Price:  ${self.position.entry_price:.2f}" if self.position.entry_price else "Entry Price:  N/A")
        print(f"Current:      ${self.position.current_price:.2f}" if self.position.current_price else "Current:      N/A")
        print(f"Unrealized:   ${state.unrealized_pnl:.2f}")
        print(f"Realized:     ${state.realized_pnl:.2f}")
        print(f"Total Equity: ${state.total_equity:.2f}")
        print(f"{'='*60}\n")


def test_tracker():
    """Test paper trading tracker."""
    print("Testing paper trading tracker...")
    
    from ..agent.env import create_env_from_ticker
    from ..agent.model import RLAgent
    
    ticker = "AAPL"
    
    # Create environment and agent
    env = create_env_from_ticker(ticker, lookback=100)
    if env is None:
        print("✗ Failed to create environment")
        return False
    
    agent = RLAgent(env)
    
    # Create tracker
    tracker = PaperTradingTracker(ticker, agent, initial_cash=100000.0)
    print(f"✓ Created tracker for {ticker}")
    
    # Print initial status
    tracker.print_status()
    
    # Generate and execute some signals
    print("Generating signals...")
    for i in range(3):
        signal = tracker.generate_and_execute()
        if signal:
            print(f"  {signal}")
        else:
            print("  HOLD")
    
    # Print final status
    tracker.print_status()
    
    # Show trade history
    trades = tracker.get_trade_history()
    print(f"\nTrade History ({len(trades)} trades):")
    for t in trades[:5]:
        print(f"  {t['time'][:19]} {t['action']} {t['shares']:.2f} @ ${t['price']:.2f}")
    
    print("\n✓ Tracker tests passed!")
    return True


if __name__ == "__main__":
    test_tracker()
