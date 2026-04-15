# Portfolio Analytics Application

## Overview
This project is an interactive web application built using Streamlit that allows users to construct and analyze equity portfolios in real time. The application provides various analytical features, including return calculations, risk analytics, correlation analysis, and portfolio optimizations.

## Features
- **User Inputs**: Enter stock ticker symbols, select date ranges, and specify portfolio preferences.
- **Data Retrieval**: Download adjusted closing prices for selected tickers and the S&P 500 benchmark using the yfinance library.
- **Return Computation**: Calculate annualized mean return, volatility, skewness, kurtosis, and daily return statistics.
- **Risk Analysis**: Analyze rolling volatility, drawdowns, and risk-adjusted metrics like Sharpe and Sortino ratios.
- **Correlation Analysis**: Generate correlation heatmaps and rolling correlations between selected stocks.
- **Portfolio Construction**: Create equal-weight portfolios, Global Minimum Variance (GMV) portfolios, and Maximum Sharpe Ratio portfolios.
- **Custom Portfolio Builder**: Manually set portfolio weights and dynamically update performance metrics.
- **Estimation Window Sensitivity**: Analyze how different lookback periods affect portfolio weights.

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd portfolio-app
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
streamlit run app.py
```

## Directory Structure
```
portfolio-app
├── src
│   ├── data
│   │   └── loader.py
│   ├── analysis
│   │   ├── returns.py
│   │   ├── risk.py
│   │   ├── correlation.py
│   │   └── optimization.py
│   ├── components
│   │   ├── charts.py
│   │   └── tables.py
│   └── utils
│       └── helpers.py
├── pages
│   ├── 01_returns.py
│   ├── 02_risk.py
│   ├── 03_correlation.py
│   ├── 04_portfolio.py
│   └── 05_sensitivity.py
├── app.py
├── requirements.txt
└── .gitignore
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.