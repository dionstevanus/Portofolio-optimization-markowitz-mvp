import pandas as pd
import numpy as np
from scipy.optimize import minimize

class MarkowitzOptimizer:
    """
    Implements Markowitz Portfolio Optimization.
    """
    
    def __init__(self, returns, risk_free_rate, num_portfolios=5000):
        """
        Initialize the MarkowitzOptimizer with returns data.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame containing daily returns for all assets
        risk_free_rate : float
            Annual risk-free rate (decimal)
        num_portfolios : int
            Number of random portfolios to generate for Monte Carlo simulation
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.num_portfolios = num_portfolios
        self.daily_rf = risk_free_rate / 252  # Daily risk-free rate
        self.num_assets = len(returns.columns)
        self.cov_matrix = returns.cov() * 252  # Annualized covariance matrix
        self.mean_returns = returns.mean() * 252  # Annualized mean returns
    
    def random_portfolios(self):
        """
        Generate random portfolio weights and calculate their returns, volatilities, and Sharpe ratios.
        
        Returns:
        --------
        tuple
            Arrays of returns, volatilities, Sharpe ratios, and weights for random portfolios
        """
        results = np.zeros((3, self.num_portfolios))
        weights_record = np.zeros((self.num_portfolios, self.num_assets))
        
        for i in range(self.num_portfolios):
            # Generate random weights
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            weights_record[i] = weights
            
            # Calculate portfolio return and volatility
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Calculate Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Store results
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
            
        return results[0], results[1], results[2], weights_record
    
    def minimize_volatility(self, target_return):
        """
        Find the portfolio with minimum volatility for a given target return.
        
        Parameters:
        -----------
        target_return : float
            Target portfolio return
            
        Returns:
        --------
        numpy.ndarray
            Optimal portfolio weights
        """
        num_assets = len(self.mean_returns)
        args = (self.cov_matrix,)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target_return})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        result = minimize(self.portfolio_volatility, 
                          num_assets * [1. / num_assets], 
                          args=args,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
                          
        return result['x']
    
    def portfolio_volatility(self, weights, cov_matrix):
        """
        Calculate portfolio volatility.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
        cov_matrix : numpy.ndarray
            Covariance matrix of returns
            
        Returns:
        --------
        float
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def max_sharpe_ratio(self):
        """
        Find the portfolio with maximum Sharpe ratio.
        
        Returns:
        --------
        numpy.ndarray
            Optimal portfolio weights
        """
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix, self.risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        result = minimize(self.negative_sharpe_ratio,
                          num_assets * [1. / num_assets],
                          args=args,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
                          
        return result['x']
    
    def negative_sharpe_ratio(self, weights, mean_returns, cov_matrix, risk_free_rate):
        """
        Calculate negative Sharpe ratio (for minimization).
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
        mean_returns : numpy.ndarray
            Mean returns of assets
        cov_matrix : numpy.ndarray
            Covariance matrix of returns
        risk_free_rate : float
            Risk-free rate
            
        Returns:
        --------
        float
            Negative Sharpe ratio
        """
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return -sharpe_ratio
    
    def efficient_frontier(self, target_returns):
        """
        Calculate the efficient frontier for a range of target returns.
        
        Parameters:
        -----------
        target_returns : numpy.ndarray
            Array of target returns
            
        Returns:
        --------
        numpy.ndarray
            Array of volatilities corresponding to target returns
        """
        volatilities = []
        
        for target in target_returns:
            weights = self.minimize_volatility(target)
            volatility = self.portfolio_volatility(weights, self.cov_matrix)
            volatilities.append(volatility)
            
        return np.array(volatilities)
    
    def optimize(self):
        """
        Perform portfolio optimization.
        
        Returns:
        --------
        dict
            Dictionary containing optimization results
        """
        # Generate random portfolios
        returns, volatilities, sharpe_ratios, weights = self.random_portfolios()
        
        # Find portfolio with maximum Sharpe ratio
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_return = returns[max_sharpe_idx]
        max_sharpe_volatility = volatilities[max_sharpe_idx]
        max_sharpe_weights = weights[max_sharpe_idx]
        max_sharpe = sharpe_ratios[max_sharpe_idx]
        
        # Find portfolio with minimum volatility
        min_vol_idx = np.argmin(volatilities)
        min_vol_return = returns[min_vol_idx]
        min_vol_volatility = volatilities[min_vol_idx]
        min_vol_weights = weights[min_vol_idx]
        min_vol_sharpe = sharpe_ratios[min_vol_idx]
        
        # Calculate optimal portfolio weights
        optimal_weights = self.max_sharpe_ratio()
        
        # Calculate performance metrics for optimal portfolio
        optimal_return = np.sum(self.mean_returns * optimal_weights)
        optimal_volatility = self.portfolio_volatility(optimal_weights, self.cov_matrix)
        optimal_sharpe = (optimal_return - self.risk_free_rate) / optimal_volatility
        
        # Return results
        return {
            'returns': returns,
            'volatilities': volatilities,
            'sharpe_ratios': sharpe_ratios,
            'weights': weights,
            'max_sharpe_return': max_sharpe_return,
            'max_sharpe_volatility': max_sharpe_volatility,
            'max_sharpe_weights': max_sharpe_weights,
            'max_sharpe': max_sharpe,
            'min_vol_return': min_vol_return,
            'min_vol_volatility': min_vol_volatility,
            'min_vol_weights': min_vol_weights,
            'min_vol_sharpe': min_vol_sharpe,
            'optimal_weights': optimal_weights,
            'optimal_return': optimal_return,
            'optimal_volatility': optimal_volatility,
            'optimal_sharpe': optimal_sharpe
        }
