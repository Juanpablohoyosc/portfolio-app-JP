def clean_data(data):
    """Cleans the financial data by handling missing values."""
    return data.dropna()

def normalize_weights(weights):
    """Normalizes a list of weights to sum to 1."""
    total = sum(weights)
    if total == 0:
        return weights
    return [w / total for w in weights]

def handle_error(error_message):
    """Handles errors by logging or displaying an error message."""
    print(f"Error: {error_message}")  # Replace with appropriate logging in production

def validate_tickers(tickers):
    """Validates a list of ticker symbols."""
    valid_tickers = []
    for ticker in tickers:
        if isinstance(ticker, str) and len(ticker) > 0:
            valid_tickers.append(ticker)
    return valid_tickers if 3 <= len(valid_tickers) <= 10 else None