"""
Main Execution Script for Stock Portfolio Analysis
Orchestrates all modules and generates comprehensive analysis report
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    STOCK_TICKERS, STOCK_SECTORS, RISK_FREE_RATE,
    START_DATE_STR, END_DATE_STR, REPORTS_DIR
)
from src.data_fetcher import DataFetcher
from src.returns_analysis import ReturnsAnalysis
from src.risk_metrics import RiskMetrics
from src.correlation_analysis import CorrelationAnalysis
from src.portfolio_optimizer import PortfolioOptimizer
from src.stress_testing import StressTesting
from src.visualizations import Visualizations


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def generate_markdown_report(results, filepath):
    """
    Generate comprehensive markdown report
    
    Args:
        results: Dict with all analysis results
        filepath: Path to save the report
    """
    report = f"""# Stock Portfolio Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of a diversified portfolio consisting of 
**{len(STOCK_TICKERS)} NIFTY 50 stocks** covering multiple sectors including Banking, IT Services, 
FMCG, Energy, Telecom, and Infrastructure.

**Analysis Period:** {START_DATE_STR} to {END_DATE_STR}

---

## 1. Portfolio Stocks

| Stock | Sector |
|-------|--------|
"""
    for stock in STOCK_TICKERS:
        sector = STOCK_SECTORS.get(stock, 'Other')
        report += f"| {stock.replace('.NS', '')} | {sector} |\n"
    
    # Returns Summary
    report += """
---

## 2. Return Analysis

### Individual Stock Performance
"""
    returns_summary = results['returns_summary']
    report += returns_summary.to_markdown() + "\n"
    
    # Risk Metrics
    report += """
---

## 3. Risk Metrics

### Individual Stock Risk Profile
"""
    risk_summary = results['risk_summary']
    report += risk_summary.to_markdown() + "\n"
    
    # Correlation
    report += """
---

## 4. Correlation Analysis

### Average Portfolio Correlation
"""
    report += f"**Average Correlation:** {results['avg_correlation']:.4f}\n\n"
    report += "### Most Correlated Pairs\n"
    report += results['highest_correlations'].to_markdown() + "\n\n"
    report += "### Least Correlated Pairs (Best for Diversification)\n"
    report += results['lowest_correlations'].to_markdown() + "\n"
    
    # Portfolio Optimization
    report += """
---

## 5. Portfolio Optimization Results

### Optimal Portfolios Comparison
"""
    opt_summary = results['optimization_summary']
    report += opt_summary.to_markdown() + "\n"
    
    report += """
### Maximum Sharpe Ratio Portfolio Weights
"""
    max_sharpe = results['max_sharpe_portfolio']
    for asset, weight in max_sharpe['weights'].items():
        if weight > 0.01:
            report += f"- **{asset.replace('.NS', '')}:** {weight*100:.2f}%\n"
    
    report += f"""
**Expected Return:** {max_sharpe['return']*100:.2f}%  
**Volatility:** {max_sharpe['volatility']*100:.2f}%  
**Sharpe Ratio:** {max_sharpe['sharpe_ratio']:.4f}
"""
    
    # Stress Testing
    report += """
---

## 6. Stress Testing & Risk Analysis

### Tail Risk Metrics
"""
    tail_risk = results['tail_risk_metrics']
    for metric, value in tail_risk.items():
        report += f"- **{metric}:** {value:.4f}\n"
    
    report += """
### Stress Scenario Analysis
"""
    stress_scenarios = results['stress_scenarios']
    report += stress_scenarios.to_markdown() + "\n"
    
    report += """
### Worst 5 Trading Days
"""
    report += results['worst_days'].to_markdown() + "\n"
    
    # Visualizations Reference
    report += f"""
---

## 7. Visualizations

The following charts have been generated in the `{REPORTS_DIR}/` directory:

1. **stock_prices.png** - Normalized stock price trends
2. **returns_distribution.png** - Daily returns distribution histograms
3. **correlation_heatmap.png** - Stock correlation matrix heatmap
4. **efficient_frontier.png** - Efficient frontier with optimal portfolios
5. **portfolio_composition.png** - Optimal portfolio weight allocation
6. **drawdown_analysis.png** - Portfolio drawdown over time
7. **cumulative_returns.png** - Cumulative portfolio vs benchmark returns
8. **risk_return_scatter.png** - Risk-return profile of individual stocks
9. **rolling_volatility.png** - 21-day rolling volatility

---

## 8. Key Insights

1. **Diversification:** The portfolio achieves diversification across {len(set(STOCK_SECTORS.values()))} sectors.

2. **Optimal Allocation:** The Maximum Sharpe Ratio portfolio suggests concentrating 
   investments in stocks with the best risk-adjusted returns.

3. **Risk Management:** VaR and CVaR metrics provide clear downside risk estimates for 
   portfolio stress testing.

4. **Correlation:** Low correlation pairs offer opportunities for further diversification.

---

## 9. Methodology

### Financial Formulas Used

- **CAGR:** $(P_{{end}}/P_{{start}})^{{1/n}} - 1$
- **Volatility:** $\\sigma = \\sqrt{{\\sum(r_i - \\bar{{r}})^2/(n-1)}}$
- **Beta:** $\\beta = Cov(r_i, r_m) / Var(r_m)$
- **Sharpe Ratio:** $(R_p - R_f) / \\sigma_p$
- **VaR (95%):** 5th percentile of return distribution
- **CVaR:** Mean of returns below VaR threshold

### Data Source
- Historical price data fetched from Yahoo Finance via `yfinance` library

---

*Report generated by Python Portfolio Analysis System*
"""
    
    # Save report
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {filepath}")


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("  STOCK PORTFOLIO ANALYSIS, OPTIMIZATION AND RISK MANAGEMENT")
    print("=" * 70)
    print(f"\nAnalysis Period: {START_DATE_STR} to {END_DATE_STR}")
    print(f"Risk-Free Rate: {RISK_FREE_RATE*100}%")
    print(f"Number of Stocks: {len(STOCK_TICKERS)}")
    
    # Initialize results dictionary
    results = {}
    
    # =========================================================================
    # 1. Data Collection
    # =========================================================================
    print_header("1. DATA COLLECTION")
    
    fetcher = DataFetcher()
    prices, benchmark = fetcher.get_all_data()
    
    print(f"\nData collected successfully!")
    print(f"Total trading days: {len(prices)}")
    
    if prices.empty:
        print("\nERROR: No stock data collected. Exiting analysis.")
        return results

    # =========================================================================
    # 2. Returns Analysis
    # =========================================================================
    print_header("2. RETURNS ANALYSIS")
    
    returns_analyzer = ReturnsAnalysis(prices)
    daily_returns = returns_analyzer.calculate_daily_returns()
    monthly_returns = returns_analyzer.calculate_monthly_returns()
    
    returns_summary = returns_analyzer.get_summary_table()
    results['returns_summary'] = returns_summary
    
    print("\nReturn Statistics:")
    print(returns_summary)
    
    cagr = returns_analyzer.calculate_cagr()
    print("\nCAGR by Stock:")
    for stock, value in cagr.items():
        print(f"  {stock.replace('.NS', '')}: {value*100:.2f}%")
    
    # =========================================================================
    # 3. Risk Metrics
    # =========================================================================
    print_header("3. RISK METRICS")
    
    benchmark_returns = benchmark.pct_change().dropna()
    risk_analyzer = RiskMetrics(daily_returns, benchmark_returns)
    
    risk_summary = risk_analyzer.get_risk_summary()
    results['risk_summary'] = risk_summary
    
    print("\nRisk Metrics Summary:")
    print(risk_summary)
    
    # =========================================================================
    # 4. Correlation Analysis
    # =========================================================================
    print_header("4. CORRELATION ANALYSIS")
    
    corr_analyzer = CorrelationAnalysis(daily_returns)
    correlation_matrix = corr_analyzer.calculate_correlation_matrix()
    
    results['correlation_matrix'] = correlation_matrix
    results['avg_correlation'] = corr_analyzer.calculate_average_correlation()
    results['highest_correlations'] = corr_analyzer.get_highest_correlations()
    results['lowest_correlations'] = corr_analyzer.get_lowest_correlations()
    
    print(f"\nAverage Correlation: {results['avg_correlation']:.4f}")
    print("\nHighest Correlations (Similar Stocks):")
    print(results['highest_correlations'])
    print("\nLowest Correlations (Best for Diversification):")
    print(results['lowest_correlations'])
    
    # =========================================================================
    # 5. Portfolio Optimization
    # =========================================================================
    print_header("5. PORTFOLIO OPTIMIZATION")
    
    optimizer = PortfolioOptimizer(daily_returns)
    
    # Maximum Sharpe Ratio Portfolio
    max_sharpe = optimizer.optimize_sharpe_ratio()
    results['max_sharpe_portfolio'] = max_sharpe
    
    print("\nMaximum Sharpe Ratio Portfolio:")
    print(f"  Expected Return: {max_sharpe['return']*100:.2f}%")
    print(f"  Volatility: {max_sharpe['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {max_sharpe['sharpe_ratio']:.4f}")
    print("\n  Optimal Weights:")
    for asset, weight in max_sharpe['weights'].items():
        if weight > 0.01:
            print(f"    {asset.replace('.NS', '')}: {weight*100:.2f}%")
    
    # Minimum Volatility Portfolio
    min_vol = optimizer.optimize_min_volatility()
    results['min_vol_portfolio'] = min_vol
    
    print("\nMinimum Volatility Portfolio:")
    print(f"  Expected Return: {min_vol['return']*100:.2f}%")
    print(f"  Volatility: {min_vol['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {min_vol['sharpe_ratio']:.4f}")
    
    # Equal Weight Portfolio
    equal_weight = optimizer.get_equal_weight_portfolio()
    results['equal_weight_portfolio'] = equal_weight
    
    print("\nEqual Weight Portfolio:")
    print(f"  Expected Return: {equal_weight['return']*100:.2f}%")
    print(f"  Volatility: {equal_weight['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {equal_weight['sharpe_ratio']:.4f}")
    
    # Optimization summary
    results['optimization_summary'] = optimizer.get_optimization_summary()
    
    # Generate Efficient Frontier
    print("\nGenerating Efficient Frontier...")
    frontier = optimizer.generate_efficient_frontier(50)
    random_portfolios = optimizer.generate_random_portfolios(3000)
    
    results['efficient_frontier'] = frontier
    results['random_portfolios'] = random_portfolios
    
    # =========================================================================
    # 6. Stress Testing
    # =========================================================================
    print_header("6. STRESS TESTING")
    
    optimal_weights = list(max_sharpe['weights'].values())
    stress_tester = StressTesting(daily_returns, optimal_weights)
    
    # Tail risk metrics
    tail_risk = stress_tester.calculate_tail_risk_metrics()
    results['tail_risk_metrics'] = tail_risk
    
    print("\nTail Risk Metrics:")
    for metric, value in tail_risk.items():
        print(f"  {metric}: {value:.4f}")
    
    # Stress scenarios
    stress_scenarios = stress_tester.run_stress_tests()
    results['stress_scenarios'] = stress_scenarios
    
    print("\nStress Scenario Analysis:")
    print(stress_scenarios)
    
    # Worst periods
    worst_days = stress_tester.calculate_worst_periods(5, 'D')
    results['worst_days'] = worst_days
    
    print("\nWorst 5 Trading Days:")
    print(worst_days)
    
    # Drawdown analysis
    drawdown = stress_tester.calculate_drawdown()
    results['drawdown'] = drawdown
    
    print(f"\nMaximum Drawdown: {stress_tester.calculate_max_drawdown()*100:.2f}%")
    
    dd_duration = stress_tester.calculate_drawdown_duration()
    print(f"Longest Drawdown Duration: {dd_duration['max_duration_days']} days")
    
    recovery = stress_tester.calculate_recovery_time()
    if recovery:
        print(f"Recovery Time from Max DD: {recovery} days")
    else:
        print("Recovery Time: Not yet recovered")
    
    # =========================================================================
    # 7. Generate Visualizations
    # =========================================================================
    print_header("7. GENERATING VISUALIZATIONS")
    
    viz = Visualizations()
    
    # Stock prices
    viz.plot_stock_prices(prices)
    
    # Returns distribution
    viz.plot_returns_distribution(daily_returns)
    
    # Correlation heatmap
    viz.plot_correlation_heatmap(correlation_matrix)
    
    # Efficient frontier
    viz.plot_efficient_frontier(
        frontier, 
        random_portfolios,
        max_sharpe,
        min_vol
    )
    
    # Portfolio composition
    viz.plot_portfolio_composition(
        max_sharpe['weights'],
        title="Maximum Sharpe Ratio Portfolio Composition"
    )
    
    # Drawdown
    viz.plot_drawdown(drawdown)
    
    # Cumulative returns
    portfolio_returns = (daily_returns * optimal_weights).sum(axis=1)
    aligned_benchmark = benchmark_returns.loc[portfolio_returns.index]
    viz.plot_cumulative_returns(portfolio_returns, aligned_benchmark)
    
    # Risk-return scatter
    viz.plot_risk_return_scatter(daily_returns)
    
    # Rolling volatility
    viz.plot_rolling_volatility(daily_returns)
    
    print("\nAll visualizations generated successfully!")
    
    # =========================================================================
    # 8. Generate Report
    # =========================================================================
    print_header("8. GENERATING REPORT")
    
    report_path = os.path.join(REPORTS_DIR, "analysis_report.md")
    generate_markdown_report(results, report_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("ANALYSIS COMPLETE")
    
    print(f"""
Summary:
--------
✓ Analyzed {len(STOCK_TICKERS)} stocks across {len(set(STOCK_SECTORS.values()))} sectors
✓ Calculated returns (Daily, Monthly, Annual, CAGR)
✓ Computed risk metrics (Volatility, Beta, Sharpe, Sortino)
✓ Generated correlation analysis
✓ Performed Mean-Variance Optimization
✓ Created Efficient Frontier
✓ Conducted stress testing (VaR, CVaR, Drawdowns)
✓ Generated 9 visualization charts
✓ Created comprehensive analysis report

Optimal Portfolio (Max Sharpe Ratio):
  Expected Return: {max_sharpe['return']*100:.2f}%
  Volatility: {max_sharpe['volatility']*100:.2f}%
  Sharpe Ratio: {max_sharpe['sharpe_ratio']:.4f}

Output Files:
  - Data: data/stock_prices.csv, data/benchmark_prices.csv
  - Report: {REPORTS_DIR}/analysis_report.md
  - Charts: {REPORTS_DIR}/*.png (9 files)
""")

    return results


if __name__ == "__main__":
    results = main()
