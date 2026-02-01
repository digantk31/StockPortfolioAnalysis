"""
Stress Testing Module
Performs stress testing, drawdown analysis, VaR and CVaR calculations
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TRADING_DAYS, STRESS_SCENARIOS, VAR_CONFIDENCE_LEVELS


class StressTesting:
    """
    Class to perform stress testing and risk analysis
    """
    
    def __init__(self, returns_data, weights=None):
        """
        Initialize with returns data and portfolio weights
        
        Args:
            returns_data: DataFrame with daily returns
            weights: Portfolio weights (default: equal weighted)
        """
        self.returns = returns_data
        self.n_assets = len(returns_data.columns)
        self.weights = weights if weights is not None else np.array([1/self.n_assets] * self.n_assets)
        
        # Calculate portfolio returns
        self.portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
    def calculate_var(self, confidence_level=0.95, method='historical'):
        """
        Calculate Value at Risk
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
            
        Returns:
            VaR value (positive number representing potential loss)
        """
        if method == 'historical':
            # Historical VaR
            var = -np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            from scipy.stats import norm
            mean = self.portfolio_returns.mean()
            std = self.portfolio_returns.std()
            z_score = norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)
        else:
            raise ValueError("Method must be 'historical' or 'parametric'")
        
        return var
    
    def calculate_cvar(self, confidence_level=0.95):
        """
        Calculate Conditional VaR (Expected Shortfall)
        
        This is the expected loss given that we're in the worst (1-confidence) percentile
        
        Args:
            confidence_level: Confidence level
            
        Returns:
            CVaR value (positive number representing expected tail loss)
        """
        var = self.calculate_var(confidence_level, method='historical')
        
        # Get returns below VaR threshold
        tail_returns = self.portfolio_returns[self.portfolio_returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        return -tail_returns.mean()
    
    def calculate_drawdown(self):
        """
        Calculate drawdown series
        
        Returns:
            Series with drawdown values
        """
        # Calculate cumulative returns
        cumulative = (1 + self.portfolio_returns).cumprod()
        
        # Rolling maximum
        running_max = cumulative.cummax()
        
        # Drawdown
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown
    
    def calculate_max_drawdown(self):
        """
        Calculate maximum drawdown
        
        Returns:
            Maximum drawdown value (negative number)
        """
        drawdown = self.calculate_drawdown()
        return drawdown.min()
    
    def calculate_drawdown_duration(self):
        """
        Calculate drawdown duration statistics
        
        Returns:
            Dict with duration statistics
        """
        drawdown = self.calculate_drawdown()
        
        # Find periods of drawdown
        is_drawdown = drawdown < 0
        
        # Calculate duration of each drawdown period
        drawdown_periods = []
        current_duration = 0
        
        for i, in_dd in enumerate(is_drawdown):
            if in_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            drawdown_periods.append(current_duration)
        
        if len(drawdown_periods) == 0:
            return {'max_duration': 0, 'avg_duration': 0, 'count': 0}
        
        return {
            'max_duration_days': max(drawdown_periods),
            'avg_duration_days': np.mean(drawdown_periods),
            'count': len(drawdown_periods)
        }
    
    def calculate_recovery_time(self):
        """
        Calculate time to recover from maximum drawdown
        
        Returns:
            Recovery time in trading days
        """
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        # Find peak before max drawdown
        max_dd_idx = drawdown.idxmin()
        
        # Find when we recovered (next time cumulative >= previous peak)
        peak_value = running_max.loc[max_dd_idx]
        
        # Check if we recovered
        post_dd = cumulative.loc[max_dd_idx:]
        recovery_mask = post_dd >= peak_value
        
        if recovery_mask.any():
            recovery_idx = recovery_mask.idxmax()
            recovery_days = (recovery_idx - max_dd_idx).days
            return recovery_days
        else:
            return None  # Not yet recovered
    
    def simulate_stress_scenario(self, scenario_return):
        """
        Simulate portfolio impact under a stress scenario
        
        Args:
            scenario_return: Market return in the scenario (e.g., -0.30 for -30%)
            
        Returns:
            Portfolio impact
        """
        # Simple approximation: use beta to estimate portfolio impact
        # For a more sophisticated approach, we'd use historical stress periods
        
        # Calculate average beta of portfolio
        portfolio_impact = scenario_return * np.mean(self.weights)
        
        return portfolio_impact
    
    def run_stress_tests(self, scenarios=None):
        """
        Run all stress test scenarios
        
        Args:
            scenarios: Dict of scenario name -> market return
            
        Returns:
            DataFrame with stress test results
        """
        scenarios = scenarios or STRESS_SCENARIOS
        
        results = []
        for scenario_name, market_return in scenarios.items():
            # Estimate portfolio impact using historical beta
            # This is a simplified approach
            portfolio_impact = market_return * 1.0  # Assuming beta of 1 for simplicity
            
            results.append({
                'Scenario': scenario_name,
                'Market Impact (%)': market_return * 100,
                'Portfolio Impact (%)': portfolio_impact * 100,
                'Portfolio Value (assuming 100)': 100 * (1 + portfolio_impact)
            })
        
        return pd.DataFrame(results)
    
    def calculate_tail_risk_metrics(self):
        """
        Calculate various tail risk metrics
        
        Returns:
            Dict with tail risk metrics
        """
        return {
            'VaR 95% (daily %)': self.calculate_var(0.95) * 100,
            'VaR 99% (daily %)': self.calculate_var(0.99) * 100,
            'CVaR 95% (daily %)': self.calculate_cvar(0.95) * 100,
            'CVaR 99% (daily %)': self.calculate_cvar(0.99) * 100,
            'Max Drawdown (%)': self.calculate_max_drawdown() * 100,
            'Skewness': self.portfolio_returns.skew(),
            'Kurtosis': self.portfolio_returns.kurtosis(),
        }
    
    def calculate_worst_periods(self, n=5, period='D'):
        """
        Find the worst performing periods
        
        Args:
            n: Number of worst periods to return
            period: Period type ('D' for daily, 'M' for monthly)
            
        Returns:
            DataFrame with worst periods
        """
        if period == 'M':
            returns = self.portfolio_returns.resample('ME').apply(
                lambda x: (1 + x).prod() - 1
            )
        else:
            returns = self.portfolio_returns
        
        worst = returns.nsmallest(n)
        return pd.DataFrame({
            'Date': worst.index,
            'Return (%)': worst.values * 100
        })
    
    def calculate_best_periods(self, n=5, period='D'):
        """
        Find the best performing periods
        
        Args:
            n: Number of best periods to return
            period: Period type
            
        Returns:
            DataFrame with best periods
        """
        if period == 'M':
            returns = self.portfolio_returns.resample('ME').apply(
                lambda x: (1 + x).prod() - 1
            )
        else:
            returns = self.portfolio_returns
        
        best = returns.nlargest(n)
        return pd.DataFrame({
            'Date': best.index,
            'Return (%)': best.values * 100
        })
    
    def get_stress_test_summary(self):
        """
        Get comprehensive stress test summary
        
        Returns:
            Dict with all stress test results
        """
        dd_duration = self.calculate_drawdown_duration()
        recovery = self.calculate_recovery_time()
        
        return {
            'tail_risk_metrics': self.calculate_tail_risk_metrics(),
            'stress_scenarios': self.run_stress_tests(),
            'worst_5_days': self.calculate_worst_periods(5, 'D'),
            'worst_5_months': self.calculate_worst_periods(5, 'M'),
            'drawdown_duration': dd_duration,
            'recovery_time_days': recovery if recovery else 'Not yet recovered'
        }


if __name__ == "__main__":
    # Test stress testing
    from data_fetcher import DataFetcher
    from returns_analysis import ReturnsAnalysis
    
    fetcher = DataFetcher()
    prices, _ = fetcher.get_all_data()
    
    returns_analyzer = ReturnsAnalysis(prices)
    daily_returns = returns_analyzer.calculate_daily_returns()
    
    stress_tester = StressTesting(daily_returns)
    
    print("\n" + "="*60)
    print("Stress Testing Results")
    print("="*60)
    
    print("\nTail Risk Metrics:")
    for metric, value in stress_tester.calculate_tail_risk_metrics().items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nStress Scenarios:")
    print(stress_tester.run_stress_tests())
    
    print("\nWorst 5 Days:")
    print(stress_tester.calculate_worst_periods(5, 'D'))
