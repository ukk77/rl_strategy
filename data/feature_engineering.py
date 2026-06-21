"""Feature engineering for RL agent.

Merges OHLCV data with sentiment and risk features into a unified feature vector.
"""
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from ..config import get_config


class FeatureEngineer:
    """Build feature vectors from multiple data sources."""
    
    def __init__(self):
        self.cfg = get_config()

    def load_ohlcv(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load hourly OHLCV data from parquet files.

        Args:
            ticker: Stock symbol

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        data_path = Path(self.cfg.market_data_path) / f"{ticker}.parquet"
        
        if not data_path.exists():
            print(f"Warning: No data file found at {data_path}")
            return None
        
        df = pd.read_parquet(data_path)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
        
        return df
    
    def fetch_sentiment_from_db(self, ticker: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch sentiment history from SQLite database.
        
        Args:
            ticker: Stock symbol
            limit: Number of records to fetch
            
        Returns:
            DataFrame with sentiment data or None if DB not found
        """
        db_path = Path(self.cfg.sentiment_db_path)
        
        if not db_path.exists():
            print(f"Warning: Sentiment DB not found at {db_path}")
            return None
        
        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT ticker, captured_at, avg_sentiment, overall_sentiment, 
                       confidence, total_articles
                FROM sentiment_snapshots
                WHERE ticker = ?
                ORDER BY captured_at DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(ticker, limit))
            conn.close()
            
            if not df.empty:
                df['captured_at'] = pd.to_datetime(df['captured_at'])
                df.set_index('captured_at', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching sentiment from DB: {e}")
            return None
    
    def fetch_sentiment_from_api(self, ticker: str) -> Optional[Dict]:
        """Fetch latest sentiment from API as fallback.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with sentiment data or None on failure
        """
        url = f"{self.cfg.sentiment_api_url}/api/history/{ticker}?limit=1"
        
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("snapshots") and len(data["snapshots"]) > 0:
                    return data["snapshots"][0]
        except Exception as e:
            print(f"Warning: Sentiment API call failed: {e}")
        
        return None
    
    def fetch_risk_from_db(self, ticker: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch risk history from SQLite database.
        
        Args:
            ticker: Stock symbol
            limit: Number of records to fetch
            
        Returns:
            DataFrame with risk data or None if DB not found
        """
        db_path = Path(self.cfg.risk_db_path)
        
        if not db_path.exists():
            print(f"Warning: Risk DB not found at {db_path}")
            return None
        
        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT ticker, captured_at, composite_risk_score, risk_bucket,
                       var_95_hist_1d AS var_95, beta, sharpe AS sharpe_ratio
                FROM risk_snapshots
                WHERE ticker = ?
                ORDER BY captured_at DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(ticker, limit))
            conn.close()
            
            if not df.empty:
                df['captured_at'] = pd.to_datetime(df['captured_at'])
                df.set_index('captured_at', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching risk from DB: {e}")
            return None
    
    def fetch_risk_from_api(self, ticker: str) -> Optional[Dict]:
        """Fetch latest risk from API as fallback.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with risk data or None on failure
        """
        url = f"{self.cfg.risk_api_url}/api/history/{ticker}?limit=1"
        
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("snapshots") and len(data["snapshots"]) > 0:
                    return data["snapshots"][0]
        except Exception as e:
            print(f"Warning: Risk API call failed: {e}")
        
        return None
    
    def build_features(
        self,
        ticker: str,
        ohlc: pd.DataFrame,
        sentiment: Optional[pd.DataFrame] = None,
        risk: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Build feature DataFrame by merging all data sources.
        
        Args:
            ticker: Stock symbol
            ohlc: OHLCV DataFrame
            sentiment: Optional sentiment DataFrame
            risk: Optional risk DataFrame
            
        Returns:
            DataFrame with all features merged
        """
        df = ohlc.copy()
        
        # Calculate price features (10 dimensions)
        # 1-2: Returns
        # Use lowercase column names from parquet files
        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        
        df['return_1h'] = df[close_col].pct_change(1)
        df['return_24h'] = df[close_col].pct_change(24)
        
        # 3-4: Volatility (rolling std)
        df['volatility_5h'] = df[close_col].pct_change().rolling(5).std()
        df['volatility_20h'] = df[close_col].pct_change().rolling(20).std()
        
        # 5-6: Volume ratios
        df['volume_ratio_5h'] = df[volume_col] / df[volume_col].rolling(5).mean()
        df['volume_ratio_20h'] = df[volume_col] / df[volume_col].rolling(20).mean()
        
        # 7-8: Price position within range
        df['high_low_range'] = (df[close_col] - df[low_col]) / (df[high_col] - df[low_col] + 1e-8)
        df['bb_position'] = self._calculate_bb_position(df[close_col])
        
        # 9-10: Trend indicators
        df['sma_ratio'] = df[close_col] / df[close_col].rolling(20).mean()
        df['momentum'] = df[close_col].diff(10) / df[close_col].shift(10)
        
        # Merge sentiment (3 dimensions)
        if sentiment is not None and not sentiment.empty:
            df = df.merge(
                sentiment[['avg_sentiment', 'overall_sentiment', 'confidence']],
                left_index=True,
                right_index=True,
                how='left'
            )
            # Convert sentiment direction to numeric
            df['sentiment_direction'] = df['overall_sentiment'].map({
                'positive': 1,
                'negative': -1,
                'neutral': 0
            }).fillna(0)
        else:
            # Fallback values
            df['avg_sentiment'] = 0.0
            df['confidence'] = 0.5
            df['sentiment_direction'] = 0
        
        # Merge risk (3 dimensions)
        if risk is not None and not risk.empty:
            df = df.merge(
                risk[['composite_risk_score', 'var_95', 'beta']],
                left_index=True,
                right_index=True,
                how='left'
            )
        else:
            # Fallback values
            df['composite_risk_score'] = 50.0
            df['var_95'] = 0.02
            df['beta'] = 1.0
        
        # Select feature columns (17 dimensions total)
        feature_cols = [
            'return_1h', 'return_24h',
            'volatility_5h', 'volatility_20h',
            'volume_ratio_5h', 'volume_ratio_20h',
            'high_low_range', 'bb_position',
            'sma_ratio', 'momentum',
            'avg_sentiment', 'confidence', 'sentiment_direction',
            'composite_risk_score', 'var_95', 'beta'
        ]
        
        # Fill NaN values
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Clip extreme values
        for col in feature_cols:
            df[col] = df[col].clip(-10, 10)
        
        return df[feature_cols]
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands (0-1 scale)."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (prices - lower) / (upper - lower + 1e-8)
    
    def get_latest_features(
        self,
        ticker: str,
        lookback: int = 100
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Get latest features and current state for a ticker.
        
        Args:
            ticker: Stock symbol
            lookback: Number of historical periods to include
            
        Returns:
            Tuple of (features_df, current_state) or (None, None) on failure
        """
        # Load data
        ohlc = self.load_ohlcv(ticker)
        if ohlc is None or ohlc.empty:
            return None, None
        
        sentiment = self.fetch_sentiment_from_db(ticker, limit=lookback)
        risk = self.fetch_risk_from_db(ticker, limit=lookback)
        
        # Build features
        features = self.build_features(ticker, ohlc, sentiment, risk)
        
        if features.empty:
            return None, None
        
        # Get latest state
        current_state = features.iloc[-1]
        
        return features, current_state


def test_feature_engineering():
    """Test the feature engineering pipeline."""
    print("Testing feature engineering...")
    
    engineer = FeatureEngineer()
    
    # Test with AAPL
    ticker = "AAPL"
    features, current = engineer.get_latest_features(ticker, lookback=50)
    
    if features is not None:
        print(f"✓ Successfully built features for {ticker}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Columns: {list(features.columns)}")
        print(f"  Current state shape: {current.shape}")
        print(f"\nSample features (last row):")
        print(current)
        return True
    else:
        print(f"✗ Failed to build features for {ticker}")
        return False


if __name__ == "__main__":
    test_feature_engineering()
