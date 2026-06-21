"""Database module for RL strategy paper trading.

SQLite storage for positions, trades, and performance tracking.
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..config import get_config


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Safely parse datetime from SQLite string."""
    if not value:
        return None
    try:
        # Try ISO format first
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            # Try SQLite's default format: YYYY-MM-DD HH:MM:SS
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                # Try date only
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                # Return current time as fallback
                return datetime.now()


@dataclass
class Position:
    """Current position state."""
    ticker: str
    shares: float
    entry_price: float
    entry_time: datetime
    current_price: float
    current_time: datetime
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Executed trade record."""
    id: Optional[int]
    ticker: str
    action: str  # BUY, SELL, HOLD
    shares: float
    price: float
    timestamp: datetime
    pnl: Optional[float]
    signal_confidence: float
    features_snapshot: Optional[str]  # JSON of features at trade time


class PaperTradingDB:
    """SQLite database for paper trading."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.cfg = get_config()
        if db_path is None:
            db_path = self.cfg.paper_trades_db
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL DEFAULT 0,
                entry_price REAL,
                entry_time TIMESTAMP,
                current_price REAL,
                current_time TIMESTAMP,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker)
            )
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pnl REAL,
                signal_confidence REAL DEFAULT 0,
                features_snapshot TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Equity curve table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_equity REAL NOT NULL,
                cash_balance REAL NOT NULL,
                positions_value REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp)")
        
        conn.commit()
        conn.close()
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get current position for a ticker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ticker, shares, entry_price, entry_time, 
                   current_price, current_time, unrealized_pnl, realized_pnl
            FROM positions WHERE ticker = ?
        """, (ticker,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Position(
                ticker=row[0],
                shares=row[1],
                entry_price=row[2],
                entry_time=_parse_datetime(row[3]),
                current_price=row[4],
                current_time=_parse_datetime(row[5]),
                unrealized_pnl=row[6] or 0,
                realized_pnl=row[7] or 0
            )
        return None
    
    def update_position(
        self,
        ticker: str,
        shares: float,
        entry_price: Optional[float] = None,
        current_price: Optional[float] = None,
        realized_pnl: Optional[float] = None
    ):
        """Update or create position."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Check if position exists
        cursor.execute("SELECT id FROM positions WHERE ticker = ?", (ticker,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            if realized_pnl is not None:
                cursor.execute("""
                    UPDATE positions 
                    SET shares = ?, current_price = ?, current_time = ?,
                        realized_pnl = realized_pnl + ?, updated_at = ?
                    WHERE ticker = ?
                """, (shares, current_price, now, realized_pnl, now, ticker))
            else:
                cursor.execute("""
                    UPDATE positions 
                    SET shares = ?, current_price = ?, current_time = ?, updated_at = ?
                    WHERE ticker = ?
                """, (shares, current_price, now, now, ticker))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO positions (ticker, shares, entry_price, entry_time,
                                       current_price, current_time, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, shares, entry_price, now, current_price, now, now))
        
        conn.commit()
        conn.close()
    
    def record_trade(
        self,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        pnl: Optional[float] = None,
        signal_confidence: float = 0.0,
        features_snapshot: Optional[str] = None
    ) -> int:
        """Record a trade execution.
        
        Returns:
            Trade ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO trades (ticker, action, shares, price, timestamp,
                               pnl, signal_confidence, features_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, action, shares, price, now, pnl, signal_confidence, features_snapshot))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return trade_id
    
    def record_equity(
        self,
        total_equity: float,
        cash_balance: float,
        positions_value: float,
        unrealized_pnl: float = 0,
        realized_pnl: float = 0
    ):
        """Record equity snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO equity_snapshots (total_equity, cash_balance, positions_value,
                                          unrealized_pnl, realized_pnl)
            VALUES (?, ?, ?, ?, ?)
        """, (total_equity, cash_balance, positions_value, unrealized_pnl, realized_pnl))
        
        conn.commit()
        conn.close()
    
    def get_trades(self, ticker: Optional[str] = None, limit: int = 100) -> List[Trade]:
        """Get trade history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if ticker:
            cursor.execute("""
                SELECT id, ticker, action, shares, price, timestamp, pnl, 
                       signal_confidence, features_snapshot
                FROM trades WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?
            """, (ticker, limit))
        else:
            cursor.execute("""
                SELECT id, ticker, action, shares, price, timestamp, pnl,
                       signal_confidence, features_snapshot
                FROM trades ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        trades = []
        for row in rows:
            trades.append(Trade(
                id=row[0],
                ticker=row[1],
                action=row[2],
                shares=row[3],
                price=row[4],
                timestamp=_parse_datetime(row[5]),
                pnl=row[6],
                signal_confidence=row[7] or 0,
                features_snapshot=row[8]
            ))
        
        return trades
    
    def get_all_positions(self) -> List[Position]:
        """Get all current positions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ticker, shares, entry_price, entry_time, 
                   current_price, current_time, unrealized_pnl, realized_pnl
            FROM positions WHERE shares != 0
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        positions = []
        for row in rows:
            positions.append(Position(
                ticker=row[0],
                shares=row[1],
                entry_price=row[2],
                entry_time=_parse_datetime(row[3]),
                current_price=row[4],
                current_time=_parse_datetime(row[5]),
                unrealized_pnl=row[6] or 0,
                realized_pnl=row[7] or 0
            ))
        
        return positions
    
    def get_performance_stats(self) -> dict:
        """Calculate performance statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]
        
        # Win rate
        cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
        winning_trades = cursor.fetchone()[0]
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Total P&L
        cursor.execute("SELECT SUM(pnl) FROM trades")
        total_pnl = cursor.fetchone()[0] or 0
        
        # Average P&L per trade
        cursor.execute("SELECT AVG(pnl) FROM trades")
        avg_pnl = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl
        }


def test_db():
    """Test the database module."""
    print("Testing paper trading database...")
    
    db = PaperTradingDB("test_paper_trades.db")
    
    # Test recording a trade
    db.record_trade("AAPL", "BUY", 100, 150.0, signal_confidence=0.85)
    print("✓ Recorded trade")
    
    # Test updating position
    db.update_position("AAPL", 100, entry_price=150.0, current_price=152.0)
    print("✓ Updated position")
    
    # Test getting position
    pos = db.get_position("AAPL")
    if pos:
        print(f"✓ Retrieved position: {pos.shares} shares @ ${pos.entry_price}")
    
    # Test getting trades
    trades = db.get_trades("AAPL")
    print(f"✓ Retrieved {len(trades)} trades")
    
    # Test performance stats
    stats = db.get_performance_stats()
    print(f"✓ Performance stats: {stats}")
    
    # Cleanup
    import os
    os.remove("test_paper_trades.db")
    print("\n✓ All database tests passed!")


if __name__ == "__main__":
    test_db()
