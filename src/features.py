import numpy as np
import pandas as pd

def compute_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Compute log returns."""
    return np.log(prices / prices.shift(periods))

def compute_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling volatility."""
    return returns.rolling(window).std()

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_features(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute all features for regime detection."""
    close = prices
    returns = compute_returns(close)

    features = pd.DataFrame({
        'return': returns,
        'volatility': compute_volatility(returns, window),
        'volatility_60': compute_volatility(returns, 60),
        'rsi': compute_rsi(close),
        'momentum': close / close.shift(window) - 1,
        'skewness': returns.rolling(window).skew(),
        'kurtosis': returns.rolling(window).kurt()
    })

    return features.dropna()