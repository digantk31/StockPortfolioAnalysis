# Stock Portfolio Analysis, Optimization and Risk Management

A comprehensive Python-based portfolio management system for analyzing NIFTY 50 stocks, optimizing asset allocation, and measuring risk.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Historical Data Analysis**: Fetch and analyze 3 years of NIFTY 50 stock data
- **Return Calculations**: Daily, monthly, annual returns & CAGR
- **Risk Metrics**: Volatility, Beta, Sharpe Ratio, Sortino Ratio, Max Drawdown
- **Correlation Analysis**: Identify diversification opportunities
- **Portfolio Optimization**: Mean-Variance Optimization using Modern Portfolio Theory
- **Efficient Frontier**: Visualize optimal risk-return tradeoffs
- **Stress Testing**: VaR, CVaR, and historical crash simulations
- **Comprehensive Visualizations**: 9 professional charts

## Project Structure

```
StockPortfolioAnalysis/
├── config/
│   └── config.py              # Configuration settings
├── data/                      # Downloaded stock data
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py        # Stock data collection
│   ├── returns_analysis.py    # Return calculations
│   ├── risk_metrics.py        # Risk measurement
│   ├── correlation_analysis.py # Correlation & diversification
│   ├── portfolio_optimizer.py # Mean-Variance Optimization
│   ├── stress_testing.py      # Stress testing & drawdowns
│   └── visualizations.py      # Plotting functions
├── reports/                   # Generated reports & charts
├── main.py                    # Main execution script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Selected Stocks

| Stock | Sector |
|-------|--------|
| RELIANCE | Energy |
| TCS | IT Services |
| HDFCBANK | Banking |
| INFY | IT Services |
| HINDUNILVR | FMCG |
| ICICIBANK | Banking |
| BHARTIARTL | Telecom |
| ITC | FMCG |
| KOTAKBANK | Banking |
| LT | Infrastructure |

## Quick Start

### 1. Clone & Setup
Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/digantk31/StockPortfolioAnalysis.git
cd StockPortfolioAnalysis
```

### 2. Create Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Run the Project
You can run the full analysis pipeline using the main script:

```bash
python main.py
```

Or explore the interactive notebook:
```bash
jupyter notebook notebooks/portfolio_analysis_demo.ipynb
```

## Output

The analysis generates:

### Data Files
- `data/stock_prices.csv` - Historical adjusted close prices
- `data/benchmark_prices.csv` - NIFTY 50 index prices

### Report
- `reports/analysis_report.md` - Comprehensive analysis report

### Visualizations
1. `stock_prices.png` - Normalized price trends
2. `returns_distribution.png` - Daily returns histograms
3. `correlation_heatmap.png` - Stock correlation matrix
4. `efficient_frontier.png` - Optimal portfolio frontier
5. `portfolio_composition.png` - Weight allocation pie chart
6. `drawdown_analysis.png` - Portfolio drawdown over time
7. `cumulative_returns.png` - Portfolio vs benchmark returns
8. `risk_return_scatter.png` - Risk-return profile
9. `rolling_volatility.png` - 21-day rolling volatility

## Key Financial Concepts & Formulas

### 1. Daily Returns
How much the stock price changed from one day to the next.
- **Formula**: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
- *Simple English*: (Today's Price - Yesterday's Price) / Yesterday's Price

### 2. Compound Annual Growth Rate (CAGR)
The mean annual growth rate of an investment over a specified time period longer than one year.
- **Formula**: $CAGR = (\frac{P_{end}}{P_{start}})^{\frac{1}{n}} - 1$
- *Simple English*: How much your money grew per year on average, smoothing out the bumps.

### 3. Volatility (Risk)
A measure of how much the stock's returns swing around the average. High volatility means high risk.
- **Formula (Annualized)**: $\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$
- *Simple English*: The "nervousness" of the stock. 252 is the number of trading days in a year.

### 4. Sharpe Ratio
Measures return per unit of risk. The gold standard for comparing investments.
- **Formula**: $Sharpe = \frac{R_p - R_f}{\sigma_p}$
- *Where*:
  - $R_p$ = Portfolio Return
  - $R_f$ = Risk-Free Rate (e.g., Govt Bond Yield)
  - $\sigma_p$ = Portfolio Volatility
- *Simple English*: Is the extra risk worth the extra reward? Higher is better (above 1.0 is good).

### 5. Beta ($\beta$)
Measures how a stock moves compared to the market (NIFTY 50).
- **Formula**: $\beta = \frac{Cov(R_s, R_m)}{Var(R_m)}$
- *Simple English*:
  - $\beta = 1$: Moves exactly with the market.
  - $\beta > 1$: More volatile than the market (Aggressive).
  - $\beta < 1$: Less volatile than the market (Defensive).

### 6. Maximum Drawdown
The largest drop from a peak to a trough.
- **Formula**: $MDD = \frac{Trough Value - Peak Value}{Peak Value}$
- *Simple English*: What's the worst-case loss I could have suffered if I bought at the top and sold at the bottom?

### 7. Identifying Optimal Weights (Portfolio Optimization)
We use **Mean-Variance Optimization** to find the best mix of stocks.
- **Objective**: Maximize Sharpe Ratio
- **Math**: $\text{max} \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$
- *Where*: $w$ = weights, $\mu$ = expected returns, $\Sigma$ = covariance matrix.

## Learning Outcomes

This project demonstrates:
- Python for quantitative finance
- Portfolio performance evaluation
- Risk management techniques
- Diversification principles
- Mean-Variance Optimization
- Financial data visualization
- Real-world market behavior analysis

## License

MIT License - Feel free to use for educational purposes.

---

*Built with Python for data-driven investment decisions*
