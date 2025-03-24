import pandas as pd
import numpy as np
import yfinance as yf
import datetime

class DataProcessor:
    """
    Handles data collection and preprocessing for portfolio optimization.
    """
    
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize the DataProcessor with stock tickers and date range.
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        start_date : datetime.date
            Start date for historical data
        end_date : datetime.date
            End date for historical data
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.returns = None
    
    def get_stock_data(self):
        """
        Fetch historical stock data from Yahoo Finance.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing adjusted close prices for all tickers
        """
        try:
            # Convert dates to string format for yfinance
            start_str = self.start_date.strftime('%Y-%m-%d')
            end_str = self.end_date.strftime('%Y-%m-%d')
            
            print(f"Fetching data for tickers: {self.tickers}")
            print(f"Date range: {start_str} to {end_str}")
            
            # Ensure date range is valid and start date is before end date
            if self.start_date >= self.end_date:
                raise Exception("Start date must be before end date")
            
            # Ensure date range is not too short (at least 30 days)
            if (self.end_date - self.start_date).days < 30:
                self.start_date = self.end_date - datetime.timedelta(days=30)
                start_str = self.start_date.strftime('%Y-%m-%d')
                print(f"Adjusted start date to ensure minimum 30-day period: {start_str}")
            
            # Try fetching each ticker individually to identify problematic tickers
            valid_tickers = []
            for ticker in self.tickers:
                try:
                    single_data = yf.download(ticker, start=start_str, end=end_str, progress=False)
                    if not single_data.empty:
                        valid_tickers.append(ticker)
                    else:
                        print(f"Warning: No data found for {ticker}")
                except Exception as e:
                    print(f"Error fetching {ticker}: {str(e)}")
            
            if not valid_tickers:
                raise Exception("No valid tickers found. Please check your ticker symbols.")
            
            print(f"Valid tickers: {valid_tickers}")
            
            # Fetch data for valid tickers
            data = yf.download(valid_tickers, start=start_str, end=end_str)
            
            # Check if data is empty
            if data.empty:
                raise Exception("No data found for the selected stocks in the given date range")
            
            # Extract adjusted close prices - handle different versions of yfinance
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-level columns (multiple tickers)
                    if 'Adj Close' in data.columns.levels[0]:
                        self.stock_data = data['Adj Close']
                    else:
                        self.stock_data = data['Close']
                else:
                    # Single level columns (one ticker)
                    if 'Adj Close' in data.columns:
                        self.stock_data = data['Adj Close']
                    else:
                        self.stock_data = data['Close']
            except Exception as e:
                print(f"Error extracting price data: {str(e)}")
                # Fallback approach
                try:
                    self.stock_data = data.xs('Adj Close', level=0, axis=1, drop_level=False)
                except:
                    try:
                        self.stock_data = data.xs('Close', level=0, axis=1, drop_level=False)
                    except Exception as e2:
                        print(f"Fallback extraction failed: {str(e2)}")
                        # Last resort: try to use whatever data we have
                        self.stock_data = data
            
            # Handle case where only one ticker is provided
            if len(valid_tickers) == 1:
                if isinstance(self.stock_data, pd.Series):
                    self.stock_data = pd.DataFrame(self.stock_data)
                    self.stock_data.columns = valid_tickers
                elif len(self.stock_data.shape) == 2 and self.stock_data.shape[1] == 1:
                    # Ensure column name matches ticker
                    self.stock_data.columns = valid_tickers
            
            # Show data shape and first few rows for debugging
            print(f"Data shape: {self.stock_data.shape}")
            print("First few rows:")
            print(self.stock_data.head())
            
            # Forward fill missing values (for non-trading days)
            self.stock_data = self.stock_data.ffill()
            
            # Drop any remaining NaN values
            self.stock_data = self.stock_data.dropna()
            
            # Update the tickers list to only include valid ones
            self.tickers = valid_tickers
            
            return self.stock_data
            
        except Exception as e:
            print(f"Failed to fetch stock data: {str(e)}")
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def calculate_returns(self):
        """
        Calculate daily returns from adjusted close prices.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing daily returns for all tickers
        """
        if self.stock_data is None:
            self.get_stock_data()
        
        # Calculate daily returns
        self.returns = self.stock_data.pct_change().dropna()
        
        return self.returns
    
    def get_covariance_matrix(self):
        """
        Calculate the covariance matrix of returns.
        
        Returns:
        --------
        pandas.DataFrame
            Covariance matrix of returns
        """
        if self.returns is None:
            self.calculate_returns()
        
        return self.returns.cov()
    
    def get_correlation_matrix(self):
        """
        Calculate the correlation matrix of returns.
        
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix of returns
        """
        if self.returns is None:
            self.calculate_returns()
        
        return self.returns.corr()
    
    def get_annualized_returns(self):
        """
        Calculate annualized returns.
        
        Returns:
        --------
        pandas.Series
            Annualized returns for each ticker
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Assuming 252 trading days in a year
        return self.returns.mean() * 252
    
    def get_annualized_volatility(self):
        """
        Calculate annualized volatility.
        
        Returns:
        --------
        pandas.Series
            Annualized volatility for each ticker
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Assuming 252 trading days in a year
        return self.returns.std() * np.sqrt(252)
