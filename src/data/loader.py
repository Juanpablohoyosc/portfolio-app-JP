import yfinance as yf
import pandas as pd
import streamlit as st

BENCHMARK = "^GSPC"
MIN_DATA_THRESHOLD = 0.95

@st.cache_data(ttl=3600)
def download_data(tickers: list, start: str, end: str):
    """
    Downloads adjusted closing prices for tickers + S&P 500 benchmark.
    Returns:
        data (pd.DataFrame): Clean price data including benchmark
        valid_tickers (list): Tickers that passed validation
        warnings (list): Warning messages for dropped/truncated tickers
    """
    warnings = []
    all_tickers = list(tickers) + [BENCHMARK]

    # --------------------------------
    # Download raw data
    # --------------------------------
    try:
        raw = yf.download(
            all_tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True
        )["Close"]
    except Exception as e:
        return None, [], [f"❌ Data download failed: {str(e)}"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    if raw.empty:
        return None, [], ["❌ No data returned. Check tickers and date range."]

    # --------------------------------
    # Validate each ticker
    # --------------------------------
    valid_tickers = []

    for ticker in tickers:
        if ticker not in raw.columns:
            warnings.append(f"❌ **{ticker}** could not be downloaded and was removed.")
            continue

        missing_pct = raw[ticker].isna().sum() / len(raw[ticker])

        if missing_pct > (1 - MIN_DATA_THRESHOLD):
            warnings.append(
                f"⚠️ **{ticker}** has {missing_pct:.1%} missing data and was removed."
            )
        else:
            valid_tickers.append(ticker)

    # --------------------------------
    # Keep valid tickers + benchmark
    # --------------------------------
    keep_cols = valid_tickers + [BENCHMARK]
    data = raw[[col for col in keep_cols if col in raw.columns]]

    # --------------------------------
    # Truncate to overlapping date range
    # --------------------------------
    original_len = len(data)
    data = data.dropna()
    truncated_len = len(data)

    if truncated_len < original_len:
        warnings.append(
            f"ℹ️ Data truncated to overlapping date range: "
            f"**{data.index[0].date()}** to **{data.index[-1].date()}** "
            f"({truncated_len} trading days)."
        )

    # --------------------------------
    # Final check — enough data?
    # --------------------------------
    if len(data) < 504:
        warnings.append(
            f"⚠️ Only {len(data)} trading days available after cleaning. "
            "Results may be unreliable."
        )

    return data, valid_tickers, warnings


def get_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns."""
    return data.pct_change().dropna()


def get_stock_returns(data: pd.DataFrame, valid_tickers: list) -> pd.DataFrame:
    """Returns only for valid stock tickers (excludes benchmark)."""
    returns = get_returns(data)
    return returns[valid_tickers]


def get_benchmark_returns(data: pd.DataFrame) -> pd.Series:
    """Returns only for S&P 500 benchmark."""
    returns = get_returns(data)
    return returns[BENCHMARK]