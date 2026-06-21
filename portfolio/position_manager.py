"""Per-ticker position coordination for RL strategy.

Manages position limits, sector exposure, and cash allocation
across all tickers traded by the RL strategy.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from rl_strategy.config import get_config
from rl_strategy.paper_trading.db import PaperTradingDB


# Sector mappings for common tickers
SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "GOOGL": "Technology", "META": "Technology",
    "NFLX": "Technology", "AMZN": "Consumer", "TSLA": "Consumer",
    "UBER": "Technology", "PLTR": "Technology", "ASML": "Technology",
    "AVGO": "Technology", "LITE": "Technology", "MU": "Technology",
    "NVTS": "Technology", "SMCI": "Technology",
    "JPM": "Financial", "V": "Financial", "MA": "Financial",
    "BRK.B": "Financial", "XLF": "Financial", "GS": "Financial",
    "MS": "Financial", "BLK": "Financial",
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare",
    "MRK": "Healthcare", "XLV": "Healthcare", "ABBV": "Healthcare",
    "GILD": "Healthcare",
    "CAT": "Industrial", "BA": "Industrial", "LMT": "Industrial",
    "GE": "Industrial", "NUE": "Materials", "XLB": "Materials",
    "FCX": "Materials", "MP": "Materials", "RTX": "Industrial",
    "XOM": "Energy", "VST": "Energy", "XLE": "Energy",
    "EQT": "Energy", "KMI": "Energy", "WMB": "Energy",
    "USAR": "Energy", "UUUU": "Energy",
    "COST": "Consumer", "HD": "Consumer", "WMT": "Consumer",
    "MCD": "Consumer", "XLP": "Consumer", "BABA": "Consumer",
    "NB": "Consumer",
    "COIN": "Crypto", "MARA": "Crypto", "MSTR": "Crypto",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index",
    "TLT": "Bond", "SQQQ": "Index", "VIX": "Volatility",
    "XLU": "Utilities", "XLRE": "RealEstate", "XLK": "Technology",
    "GLD": "Commodity",
}


@dataclass
class PositionLimits:
    """Position limit configuration."""
    max_positions: int = 10
    max_position_pct: float = 0.15       # Max 15% of portfolio per ticker
    max_sector_pct: float = 0.40         # Max 40% per sector
    min_cash_reserve_pct: float = 0.10   # Keep 10% cash reserve


@dataclass
class AllocationResult:
    """Result of position allocation check."""
    ticker: str
    can_open: bool
    reason: str
    max_shares: float = 0.0
    max_dollars: float = 0.0
    current_exposure_pct: float = 0.0
    sector_exposure_pct: float = 0.0


class PositionManager:
    """Manages position limits and allocation across RL strategy tickers."""

    def __init__(self, limits: Optional[PositionLimits] = None):
        """Initialize position manager.

        Args:
            limits: Position limit configuration (uses defaults if None)
        """
        self.cfg = get_config()
        self.limits = limits or PositionLimits()
        self.db = PaperTradingDB()

    def get_portfolio_state(self) -> dict:
        """Get current portfolio state from paper trading DB.

        Returns:
            Dict with cash, positions, total_equity, etc.
        """
        positions = self.db.get_all_positions()
        stats = self.db.get_performance_stats()

        positions_value = sum(
            p.shares * (p.current_price or 0) for p in positions
        )
        # Estimate cash from initial capital minus position costs
        initial_capital = 100000.0
        position_cost = sum(
            p.shares * (p.entry_price or 0) for p in positions
        )
        cash = initial_capital - position_cost + stats.get('total_pnl', 0)
        total_equity = cash + positions_value

        return {
            "cash": cash,
            "positions_value": positions_value,
            "total_equity": total_equity,
            "num_positions": len(positions),
            "positions": positions,
        }

    def can_open_position(
        self,
        ticker: str,
        desired_dollars: float,
        current_price: float,
    ) -> AllocationResult:
        """Check if a new position can be opened within limits.

        Args:
            ticker: Stock symbol
            desired_dollars: Desired position size in dollars
            current_price: Current stock price

        Returns:
            AllocationResult with can_open flag and reason
        """
        state = self.get_portfolio_state()
        total_equity = state["total_equity"]
        positions = state["positions"]

        # Check max positions limit
        if len(positions) >= self.limits.max_positions:
            # Check if we already hold this ticker
            existing = [p for p in positions if p.ticker == ticker]
            if not existing:
                return AllocationResult(
                    ticker=ticker,
                    can_open=False,
                    reason=f"Max positions ({self.limits.max_positions}) reached",
                )

        # Check per-ticker exposure
        max_ticker_dollars = total_equity * self.limits.max_position_pct
        existing_ticker = [p for p in positions if p.ticker == ticker]
        current_ticker_exposure = sum(
            p.shares * (p.current_price or 0) for p in existing_ticker
        )
        new_total_exposure = current_ticker_exposure + desired_dollars
        current_ticker_pct = (current_ticker_exposure / total_equity * 100) if total_equity > 0 else 0

        if new_total_exposure > max_ticker_dollars:
            return AllocationResult(
                ticker=ticker,
                can_open=False,
                reason=f"Position would exceed {self.limits.max_position_pct*100:.0f}% limit "
                       f"(${new_total_exposure:.0f} > ${max_ticker_dollars:.0f})",
                max_shares=max_ticker_dollars / current_price if current_price > 0 else 0,
                max_dollars=max_ticker_dollars,
                current_exposure_pct=current_ticker_pct,
            )

        # Check sector exposure
        sector = SECTOR_MAP.get(ticker, "Unknown")
        sector_positions = [
            p for p in positions
            if SECTOR_MAP.get(p.ticker, "Unknown") == sector
        ]
        sector_exposure = sum(
            p.shares * (p.current_price or 0) for p in sector_positions
        )
        new_sector_exposure = sector_exposure + desired_dollars
        max_sector_dollars = total_equity * self.limits.max_sector_pct
        sector_pct = (sector_exposure / total_equity * 100) if total_equity > 0 else 0

        if new_sector_exposure > max_sector_dollars:
            return AllocationResult(
                ticker=ticker,
                can_open=False,
                reason=f"Sector '{sector}' would exceed {self.limits.max_sector_pct*100:.0f}% limit "
                       f"(${new_sector_exposure:.0f} > ${max_sector_dollars:.0f})",
                max_shares=(max_sector_dollars - sector_exposure) / current_price if current_price > 0 else 0,
                max_dollars=max_sector_dollars - sector_exposure,
                sector_exposure_pct=sector_pct,
            )

        # Check cash reserve
        cash_after_trade = state["cash"] - desired_dollars
        min_cash = total_equity * self.limits.min_cash_reserve_pct
        if cash_after_trade < min_cash:
            max_affordable = state["cash"] - min_cash
            if max_affordable <= 0:
                return AllocationResult(
                    ticker=ticker,
                    can_open=False,
                    reason=f"Insufficient cash reserve (need ${min_cash:.0f})",
                )
            # Can still open but at reduced size
            return AllocationResult(
                ticker=ticker,
                can_open=True,
                reason=f"Reduced to ${max_affordable:.0f} to maintain cash reserve",
                max_shares=max_affordable / current_price if current_price > 0 else 0,
                max_dollars=max_affordable,
                current_exposure_pct=current_ticker_pct,
                sector_exposure_pct=sector_pct,
            )

        return AllocationResult(
            ticker=ticker,
            can_open=True,
            reason="OK",
            max_shares=desired_dollars / current_price if current_price > 0 else 0,
            max_dollars=desired_dollars,
            current_exposure_pct=current_ticker_pct,
            sector_exposure_pct=sector_pct,
        )

    def get_sector_exposure(self) -> Dict[str, float]:
        """Get current exposure by sector.

        Returns:
            Dict mapping sector name to dollar exposure
        """
        state = self.get_portfolio_state()
        positions = state["positions"]

        sector_exposure: Dict[str, float] = {}
        for p in positions:
            sector = SECTOR_MAP.get(p.ticker, "Unknown")
            exposure = p.shares * (p.current_price or 0)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + exposure

        return sector_exposure

    def print_status(self):
        """Print current portfolio status with limits."""
        state = self.get_portfolio_state()
        sector_exposure = self.get_sector_exposure()

        print("\n" + "=" * 60)
        print("RL STRATEGY - PORTFOLIO STATUS")
        print("=" * 60)

        print(f"\nTotal Equity:    ${state['total_equity']:>12,.2f}")
        print(f"Cash:            ${state['cash']:>12,.2f}")
        print(f"Positions Value: ${state['positions_value']:>12,.2f}")
        print(f"Open Positions:   {state['num_positions']}/{self.limits.max_positions}")

        if state["positions"]:
            print(f"\n{'Ticker':<10} {'Shares':>10} {'Value':>12} {'% of Port':>10}")
            print("-" * 45)
            for p in state["positions"]:
                value = p.shares * (p.current_price or 0)
                pct = (value / state['total_equity'] * 100) if state['total_equity'] > 0 else 0
                print(f"{p.ticker:<10} {p.shares:>10.2f} ${value:>11,.2f} {pct:>9.1f}%")

        if sector_exposure:
            print(f"\n{'Sector':<15} {'Exposure':>12} {'% of Port':>10}")
            print("-" * 40)
            for sector, exposure in sorted(sector_exposure.items()):
                pct = (exposure / state['total_equity'] * 100) if state['total_equity'] > 0 else 0
                limit_pct = self.limits.max_sector_pct * 100
                flag = " !LIMIT" if pct > limit_pct else ""
                print(f"{sector:<15} ${exposure:>11,.2f} {pct:>9.1f}%{flag}")

        print()
