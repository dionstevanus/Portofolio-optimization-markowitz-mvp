import numpy as np
import pandas as pd
import random

class SimplifiedRLOptimizer:
    """
    Implements a simplified version of Reinforcement Learning for portfolio optimization.
    This is a simpler implementation that doesn't require gym or stable-baselines3.
    """
    
    def __init__(self, returns, training_episodes=100, window_size=30):
        """
        Initialize the RLOptimizer.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame containing daily returns for assets
        training_episodes : int
            Number of episodes to train the RL agent
        window_size : int
            Number of days to include in the state
        """
        self.returns = returns
        self.training_episodes = training_episodes
        self.window_size = window_size
        self.num_assets = len(returns.columns)
        self.asset_names = returns.columns
        self.cov_matrix = returns.cov() * 252  # Annualized covariance
        self.mean_returns = returns.mean() * 252  # Annualized mean returns
    
    def _generate_weights(self):
        """
        Generate random portfolio weights that sum to 1.
        
        Returns:
        --------
        numpy.ndarray
            Random weights
        """
        weights = np.random.random(self.num_assets)
        return weights / np.sum(weights)
    
    def _calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        tuple
            (return, volatility, sharpe_ratio)
        """
        # Portfolio return
        portfolio_return = np.dot(self.mean_returns, weights)
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def _compute_reward(self, weights):
        """
        Compute reward for weights based on Sharpe ratio and diversification.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        float
            Reward value
        """
        portfolio_return, portfolio_volatility, sharpe_ratio = self._calculate_portfolio_metrics(weights)
        
        # Diversification penalty (Herfindahl-Hirschman Index)
        concentration = np.sum(np.square(weights))
        diversification_component = -concentration * 5  # Penalize concentration
        
        # Return-based reward
        return_component = sharpe_ratio * 10
        
        return return_component + diversification_component
    
    def _simulate_backtest(self, weights):
        """
        Simulate a backtest for given weights.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        list
            List of portfolio values over time
        """
        portfolio_values = [1.0]
        
        # Calculate daily portfolio returns
        portfolio_returns = self.returns.dot(weights)
        
        # Calculate cumulative portfolio value
        for ret in portfolio_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        return portfolio_values
    
    def optimize(self):
        """
        Perform portfolio optimization using simplified RL approach.
        
        Returns:
        --------
        dict
            Optimization results
        """
        best_weights = None
        best_reward = float('-inf')
        best_sharpe = 0
        best_return = 0
        best_volatility = 0
        
        # Simple RL approach: try many random weights and choose the best
        for _ in range(self.training_episodes * 50):
            # Generate random weights
            weights = self._generate_weights()
            
            # Calculate reward
            reward = self._compute_reward(weights)
            
            # Update best weights if reward is better
            if reward > best_reward:
                best_reward = reward
                best_weights = weights
                best_return, best_volatility, best_sharpe = self._calculate_portfolio_metrics(weights)
        
        # Create results dictionary
        results = {
            'optimal_weights': pd.Series(best_weights, index=self.asset_names),
            'expected_return': best_return,
            'expected_volatility': best_volatility,
            'sharpe_ratio': best_sharpe
        }
        
        return results

# Alias to keep compatibility with existing code
RLOptimizer = SimplifiedRLOptimizer
