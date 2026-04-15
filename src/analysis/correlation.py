import pandas as pd

def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation matrix."""
    return returns.corr()

def compute_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise covariance matrix."""
    return returns.cov()

def compute_rolling_correlation(
    returns: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    window: int = 30
) -> pd.Series:
    """Compute rolling correlation between two tickers."""
    return returns[ticker1].rolling(window=window).corr(returns[ticker2])
