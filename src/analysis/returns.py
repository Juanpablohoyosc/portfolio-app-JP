import pandas as pd
import numpy as np

TRADING_DAYS = 252

def compute_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns."""
    return data.pct_change().dropna()

def compute_summary_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized summary statistics for each asset."""
    stats = pd.DataFrame(index=returns.columns)
    stats["Annualized Return"]     = returns.mean() * TRADING_DAYS
    stats["Annualized Volatility"] = returns.std() * np.sqrt(TRADING_DAYS)
    stats["Skewness"]              = returns.skew()
    stats["Kurtosis"]              = returns.kurt()
    stats["Min Daily Return"]      = returns.min()
    stats["Max Daily Return"]      = returns.max()
    return stats

def compute_wealth_index(returns: pd.DataFrame, initial: float = 10000) -> pd.DataFrame:
    """Compute cumulative wealth index starting from initial investment."""
    return (1 + returns).cumprod() * initial
