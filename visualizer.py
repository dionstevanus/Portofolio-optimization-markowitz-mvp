import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

class Visualizer:
    """
    Handles visualization for portfolio optimization.
    """
    
    def __init__(self):
        """
        Initialize the Visualizer.
        """
        pass
    
    def plot_efficient_frontier(self, returns, volatilities, optimal_return, optimal_volatility, sharpe_ratios):
        """
        Plot the efficient frontier with the optimal portfolio.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Array of portfolio returns
        volatilities : numpy.ndarray
            Array of portfolio volatilities
        optimal_return : float
            Return of the optimal portfolio
        optimal_volatility : float
            Volatility of the optimal portfolio
        sharpe_ratios : numpy.ndarray
            Array of Sharpe ratios
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the efficient frontier plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot of random portfolios colored by Sharpe ratio
        scatter = ax.scatter(
            volatilities, 
            returns * 100, 
            c=sharpe_ratios, 
            cmap='viridis', 
            alpha=0.5, 
            s=10
        )
        
        # Mark optimal portfolio
        ax.scatter(
            optimal_volatility, 
            optimal_return * 100, 
            color='red', 
            marker='*', 
            s=300, 
            label='Optimal Portfolio'
        )
        
        # Add colorbar to show Sharpe ratio scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        # Set labels and title
        ax.set_xlabel('Annualized Volatility (%)')
        ax.set_ylabel('Annualized Return (%)')
        ax.set_title('Efficient Frontier')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        return fig
    
    def plot_allocation(self, weights, asset_labels):
        """
        Plot the portfolio allocation.
        
        Parameters:
        -----------
        weights : numpy.ndarray or pandas.Series
            Portfolio weights
        asset_labels : list
            Asset names
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the allocation plot and weight table
        """
        # Create figure with two subplots (2 rows, 1 column)
        fig = plt.figure(figsize=(12, 10))
        
        # Define grids for pie chart and table
        ax_pie = plt.subplot2grid((2, 1), (0, 0))
        ax_table = plt.subplot2grid((2, 1), (1, 0))
        
        # Convert weights to a Series if it's not already
        if not isinstance(weights, pd.Series):
            weights = pd.Series(weights, index=asset_labels)
        
        # Sort weights in descending order
        weights_sorted = weights.sort_values(ascending=False)
        
        # Define maximum number of segments to show in pie chart
        # If more than this, group smaller ones as "Others"
        max_segments = 10
        
        if len(weights_sorted) > max_segments:
            # Keep top N weights and group the rest as "Others"
            top_weights = weights_sorted.iloc[:max_segments-1]
            others_weight = weights_sorted.iloc[max_segments-1:].sum()
            
            # Create a new Series with top weights and "Others"
            pie_weights = pd.concat([top_weights, pd.Series({"Others": others_weight})])
            
            # Create pie chart for allocation with limited segments
            wedges, texts, autotexts = ax_pie.pie(
                pie_weights, 
                labels=pie_weights.index, 
                autopct='%1.1f%%',
                startangle=90, 
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Add a note about grouped assets
            ax_pie.text(0, -1.2, f"* 'Others' represents {len(weights_sorted) - max_segments + 1} additional assets with smaller weights",
                     fontsize=9, ha='center')
        else:
            # If we have fewer segments than the threshold, show all
            wedges, texts, autotexts = ax_pie.pie(
                weights_sorted, 
                labels=weights_sorted.index, 
                autopct='%1.1f%%',
                startangle=90, 
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
        
        # Enhance text appearance
        for text in texts:
            text.set_fontsize(10)
        
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
        
        ax_pie.set_title('Portfolio Allocation')
        ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Create table with all weights
        # Format weights as percentages with 2 decimal places
        weight_percentages = [f"{w*100:.2f}%" for w in weights_sorted]
        
        # Hide table axes
        ax_table.axis('tight')
        ax_table.axis('off')
        
        # Create table with colored cells based on weight values
        cell_colors = []
        for w in weights_sorted:
            # Color gradient: darker for higher weights
            intensity = min(0.8, w * 1.5)  # Scale, max at 0.8
            cell_colors.append([(1-intensity, 1-intensity/2, 1-intensity)])
        
        table = ax_table.table(
            cellText=[[f"{w*100:.2f}%"] for w in weights_sorted],
            rowLabels=weights_sorted.index,
            colLabels=["Weight (%)"],
            loc='center',
            cellLoc='center',
            cellColours=cell_colors
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Adjust table scale
        
        ax_table.set_title('Complete Asset Allocation List', pad=20)
        
        plt.tight_layout()
        
        return fig
    
    def plot_performance_comparison(self, dates, markowitz_performance, rl_performance, equal_performance):
        """
        Plot performance comparison between different strategies.
        
        Parameters:
        -----------
        dates : pandas.DatetimeIndex
            Dates for the performance data
        markowitz_performance : pandas.Series
            Cumulative performance of Markowitz portfolio
        rl_performance : pandas.Series
            Cumulative performance of RL portfolio
        equal_performance : pandas.Series
            Cumulative performance of equal-weight portfolio
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the performance comparison plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot performances
        ax.plot(dates, markowitz_performance, label='Markowitz Portfolio', linewidth=2)
        ax.plot(dates, rl_performance, label='RL Portfolio', linewidth=2)
        ax.plot(dates, equal_performance, label='Equal-Weight Portfolio', linewidth=2, linestyle='--')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value (Starting at $1)')
        ax.set_title('Portfolio Performance Comparison')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_matrix(self, corr_matrix):
        """
        Plot the correlation matrix as a heatmap.
        
        Parameters:
        -----------
        corr_matrix : pandas.DataFrame
            Correlation matrix of asset returns
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the correlation heatmap
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Correlation')
        
        # Set ticks and labels
        ticks = np.arange(len(corr_matrix.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add title
        ax.set_title('Correlation Matrix')
        
        # Loop over data to add text annotations
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center', 
                        color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                        fontsize=8)
        
        plt.tight_layout()
        
        return fig
