"""
Visualizations Module
Creates all charts and plots for portfolio analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import REPORTS_DIR, STOCK_SECTORS


class Visualizations:
    """
    Class to create portfolio analysis visualizations
    """
    
    def __init__(self, save_dir=None):
        """
        Initialize visualizations
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or REPORTS_DIR
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_stock_prices(self, prices, title="Stock Prices Over Time", filename="stock_prices.png"):
        """
        Plot normalized stock prices
        
        Args:
            prices: DataFrame with stock prices
            title: Plot title
            filename: Output filename
        """
        # Normalize prices to start at 100
        normalized = prices / prices.iloc[0] * 100
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for column in normalized.columns:
            ax.plot(normalized.index, normalized[column], label=column.replace('.NS', ''), linewidth=1.5)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Price (Starting = 100)', fontsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def plot_returns_distribution(self, returns, title="Daily Returns Distribution", filename="returns_distribution.png"):
        """
        Plot histogram of returns for each stock
        
        Args:
            returns: DataFrame with returns
            title: Plot title
            filename: Output filename
        """
        n_stocks = len(returns.columns)
        n_cols = 3
        n_rows = (n_stocks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()
        
        for i, column in enumerate(returns.columns):
            ax = axes[i]
            data = returns[column].dropna()
            
            ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax.axvline(x=data.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {data.mean()*100:.2f}%')
            
            ax.set_title(column.replace('.NS', ''), fontsize=12, fontweight='bold')
            ax.set_xlabel('Daily Return', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def plot_correlation_heatmap(self, correlation_matrix, title="Stock Correlation Matrix", filename="correlation_heatmap.png"):
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix: DataFrame with correlations
            title: Plot title
            filename: Output filename
        """
        # Clean column names
        corr = correlation_matrix.copy()
        corr.columns = [c.replace('.NS', '') for c in corr.columns]
        corr.index = [c.replace('.NS', '') for c in corr.index]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            vmin=-1,
            vmax=1,
            annot_kws={'size': 10}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def plot_efficient_frontier(self, frontier_df, random_portfolios=None, 
                                max_sharpe=None, min_vol=None,
                                title="Efficient Frontier", filename="efficient_frontier.png"):
        """
        Plot the efficient frontier
        
        Args:
            frontier_df: DataFrame with efficient frontier portfolios
            random_portfolios: DataFrame with random portfolios (optional)
            max_sharpe: Dict with max Sharpe portfolio info
            min_vol: Dict with min volatility portfolio info
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot random portfolios if provided
        if random_portfolios is not None:
            scatter = ax.scatter(
                random_portfolios['Volatility'] * 100,
                random_portfolios['Return'] * 100,
                c=random_portfolios['Sharpe Ratio'],
                cmap='RdYlGn',
                alpha=0.5,
                s=10
            )
            plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Plot efficient frontier
        ax.plot(
            frontier_df['Volatility'] * 100,
            frontier_df['Return'] * 100,
            'b-',
            linewidth=3,
            label='Efficient Frontier'
        )
        
        # Plot special portfolios
        if max_sharpe:
            ax.scatter(
                max_sharpe['volatility'] * 100,
                max_sharpe['return'] * 100,
                marker='*',
                color='gold',
                s=500,
                edgecolors='black',
                linewidths=1,
                label=f'Max Sharpe (SR: {max_sharpe["sharpe_ratio"]:.2f})',
                zorder=5
            )
        
        if min_vol:
            ax.scatter(
                min_vol['volatility'] * 100,
                min_vol['return'] * 100,
                marker='*',
                color='red',
                s=500,
                edgecolors='black',
                linewidths=1,
                label=f'Min Volatility',
                zorder=5
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def plot_portfolio_composition(self, weights, title="Optimal Portfolio Composition", filename="portfolio_composition.png"):
        """
        Plot portfolio weights as pie chart and bar chart
        
        Args:
            weights: Dict or Series with portfolio weights
            title: Plot title
            filename: Output filename
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        # Filter out very small weights
        weights = weights[weights > 0.01]
        
        # Clean names
        weights.index = [c.replace('.NS', '') for c in weights.index]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        explode = [0.02] * len(weights)
        
        ax1.pie(
            weights * 100,
            labels=weights.index,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=90
        )
        ax1.set_title('Weight Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.barh(weights.index, weights * 100, color=colors)
        ax2.set_xlabel('Weight (%)', fontsize=12)
        ax2.set_title('Portfolio Weights', fontsize=14, fontweight='bold')
        ax2.axvline(x=100/len(weights), color='red', linestyle='--', 
                    label=f'Equal Weight ({100/len(weights):.1f}%)')
        ax2.legend()
        
        # Add value labels
        for bar, val in zip(bars, weights * 100):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def plot_drawdown(self, drawdown, title="Portfolio Drawdown", filename="drawdown_analysis.png"):
        """
        Plot drawdown over time
        
        Args:
            drawdown: Series with drawdown values
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.fill_between(drawdown.index, drawdown * 100, 0, 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown * 100, 'r-', linewidth=1)
        
        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.scatter([max_dd_idx], [max_dd_val * 100], color='darkred', s=100, zorder=5)
        ax.annotate(f'Max DD: {max_dd_val*100:.1f}%', 
                   xy=(max_dd_idx, max_dd_val * 100),
                   xytext=(10, -30), textcoords='offset points',
                   fontsize=10, color='darkred',
                   arrowprops=dict(arrowstyle='->', color='darkred'))
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def plot_cumulative_returns(self, portfolio_returns, benchmark_returns=None,
                                title="Cumulative Returns", filename="cumulative_returns.png"):
        """
        Plot cumulative returns comparison
        
        Args:
            portfolio_returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns (optional)
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
        
        ax.plot(portfolio_cumulative.index, portfolio_cumulative * 100, 
               'b-', linewidth=2, label='Portfolio')
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            ax.plot(benchmark_cumulative.index, benchmark_cumulative * 100,
                   'gray', linewidth=2, linestyle='--', label='NIFTY 50 Benchmark', alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def plot_risk_return_scatter(self, returns_data, title="Risk-Return Profile", filename="risk_return_scatter.png"):
        """
        Plot risk vs return scatter for individual stocks
        
        Args:
            returns_data: DataFrame with returns
            title: Plot title
            filename: Output filename
        """
        # Calculate annualized metrics
        annual_returns = returns_data.mean() * 252 * 100
        annual_volatility = returns_data.std() * np.sqrt(252) * 100
        sharpe = (returns_data.mean() * 252 - 0.07) / (returns_data.std() * np.sqrt(252))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create colors based on sectors
        colors = []
        sectors = []
        for stock in returns_data.columns:
            sector = STOCK_SECTORS.get(stock, 'Other')
            sectors.append(sector)
        
        unique_sectors = list(set(sectors))
        color_map = dict(zip(unique_sectors, plt.cm.Set2(np.linspace(0, 1, len(unique_sectors)))))
        colors = [color_map[s] for s in sectors]
        
        scatter = ax.scatter(
            annual_volatility, 
            annual_returns, 
            c=sharpe,
            cmap='RdYlGn',
            s=200,
            alpha=0.8,
            edgecolors='black'
        )
        
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Add labels
        for i, stock in enumerate(returns_data.columns):
            ax.annotate(
                stock.replace('.NS', ''),
                (annual_volatility.iloc[i], annual_returns.iloc[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
    def plot_rolling_volatility(self, returns, window=21, title="Rolling Volatility (21-day)", 
                                filename="rolling_volatility.png"):
        """
        Plot rolling volatility
        
        Args:
            returns: DataFrame with returns
            window: Rolling window size
            title: Plot title
            filename: Output filename
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for column in rolling_vol.columns:
            ax.plot(rolling_vol.index, rolling_vol[column], 
                   label=column.replace('.NS', ''), linewidth=1, alpha=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


if __name__ == "__main__":
    # Test visualizations
    from data_fetcher import DataFetcher
    from returns_analysis import ReturnsAnalysis
    from correlation_analysis import CorrelationAnalysis
    
    fetcher = DataFetcher()
    prices, benchmark = fetcher.get_all_data()
    
    returns_analyzer = ReturnsAnalysis(prices)
    daily_returns = returns_analyzer.calculate_daily_returns()
    
    corr_analyzer = CorrelationAnalysis(daily_returns)
    
    viz = Visualizations()
    
    print("\nGenerating visualizations...")
    viz.plot_stock_prices(prices)
    viz.plot_returns_distribution(daily_returns)
    viz.plot_correlation_heatmap(corr_analyzer.calculate_correlation_matrix())
    viz.plot_risk_return_scatter(daily_returns)
    viz.plot_rolling_volatility(daily_returns)
    
    print("\nAll visualizations saved!")
