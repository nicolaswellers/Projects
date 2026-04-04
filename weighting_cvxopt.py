import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cvx
'''
FORM TO DEFINE BASKET:

name_basket =  Basket('name', weights = [0.2, 0.2, 0.2, 0.2, 0.2], tickers = ['A', 'B', 'C', 'D', 'E'], benchmark_ticker)

FORM TO DEFINE PORTFOLIO:

name_portfolio = Portfolio('name').add_basket(name_basket)

HOW TO DEFINE TIMEFRAME AND GET RETURNS AND VOLATILITY:

name_portfolio.get_basket_vols_and_rets(timeframe_start_date, timeframe_end_date)

HOW TO GET BETAS:

name_portfolio.get_basket_beta(timeframe_start_date, timeframe_end_date)

HOW TO OPTIMIZE WEIGHTS:

optimized_weights = Optimizer(name_portfolio).optimize_weights(timeframe_start_date, timeframe_end_date)

'''

class Basket:
    
    def __init__(self, name, weights, tickers,bechmark_ticker):
        if len(tickers) != len(weights):
            raise ValueError("Tickers and weights must be the same length!")
        if not np.isclose(sum(weights), 1):
            raise ValueError("Weights must sum to 1!")
        self.name = name
        self.holdings = pd.DataFrame({'Ticker': tickers, 'Weight': weights}).set_index('Ticker')
        self.benchmark_ticker = bechmark_ticker
        
        
    def get_holdings(self):
        return self.holdings
    
    
    def get_vol_and_ret(self, start_date, end_date):
        tickers = self.holdings.index.tolist()

        # 1. Download without selecting columns yet
        data = yf.download(tickers, start=start_date, end=end_date, interval='1d', progress=False)
    
        # 2. Check for the multi-index columns
        if 'Adj Close' in data.columns:
            data = data['Adj Close']
        elif 'Close' in data.columns:
            data = data['Close']
        else:
            # If yfinance returned a flat DataFrame (happens with 1 ticker)
            # We don't need to do anything, but let's ensure it's a DataFrame
            pass

        # 3. Handle the single-ticker edge case
        # If only one ticker is downloaded, 'returns' might become a Series. 
        # We need it to stay a DataFrame so 'returns.columns' exists.
        
        returns = data.pct_change().dropna()
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()
    
        # 4. Alignment and Math
        aligned_weights = self.holdings.loc[returns.columns, 'Weight'].values
        cov_matrix = returns.cov()
    
        basket_volatility = np.sqrt(aligned_weights.T @ cov_matrix @ aligned_weights)
        basket_returns = returns.dot(aligned_weights)
        basket_returns_annualised = basket_returns.mean() * 252 
        basket_volatility_annualised = basket_volatility * np.sqrt(252)
    
        return basket_returns, basket_volatility, basket_returns_annualised, basket_volatility_annualised
    
    def get_basket_bret(self, start_date, end_date, benchmark_ticker):
        
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, interval='1d', progress=False)
        
        if 'Adj Close' in benchmark_data.columns:
            benchmark_prices = benchmark_data['Adj Close']
        elif 'Close' in benchmark_data.columns:
            benchmark_prices = benchmark_data['Close']
        else:
            raise ValueError("Benchmark data does not contain 'Adj Close' or 'Close' columns.")
        
        benchmark_returns = benchmark_prices.pct_change().dropna()
        
        return benchmark_returns
    
    def __del__ (self):
        print(f"Basket '{self.name}' is being deleted.")
        
        
class Portfolio:
    def __init__(self, name="Portfolio"):
        self.name = name
        self.baskets = []
    
    def add_basket(self, basket):
        if not isinstance(basket, Basket):
            raise ValueError("Only Basket instances can be added!")
        self.baskets.append(basket)
        return self
   
    def get_vol_ret_beta(self, start_date, end_date):
        rows = []
        for basket in self.baskets:
            ret, vol, ret_annualised, vol_annualised = basket.get_vol_and_ret(start_date, end_date)
            b_ret = basket.get_basket_bret(start_date, end_date, basket.benchmark_ticker)
            combined = pd.concat([ret, b_ret], axis=1).dropna()
            covariance_matrix = np.cov(combined.iloc[:,0], combined.iloc[:,1])
            beta = covariance_matrix[0][1] / np.var(combined.iloc[:,1])
            rows.append({'Basket': basket.name, 'Return': ret,'Return Annualised': ret_annualised, 'Volatility': vol, 'Volatility Annualised': vol_annualised, 'Beta': beta})
        return pd.DataFrame(rows).set_index('Basket')
    
    def del_basket(self, basket_name):
        self.baskets = [b for b in self.baskets if b.name.lower() != basket_name]
        return self
        

class Optimizer:
    def __init__(self, portfolio, risk_aversion=0):
        if not isinstance(portfolio, Portfolio):
            raise ValueError("Optimizer requires a Portfolio instance!")
        self.portfolio = portfolio
        self.risk_aversion = risk_aversion

    def optimized_weights(self, start_date, end_date):

        if not self.portfolio.baskets:
            raise ValueError("Portfolio has no baskets to optimize!")
        
        baskets = self.portfolio.get_vol_ret_beta(start_date, end_date)
        returns = baskets['Return Annualised'].values
        beta = baskets['Beta'].values
        
        
        # Define optimization variables
        n = len(self.portfolio.baskets)
        w = cvx.Variable(n)
        
        # Define the objective (maximize returns)
        
        objective = cvx.Maximize(returns @ w - self.risk_aversion * beta @ w)  # Adjusted for beta
        
        # Define constraints (weights sum to 1 and are non-negative)
        constraints = [cvx.sum(w) == 1, w<=0.20, w >= 0.05]
        
        # Solve the optimization problem
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        if w.value is None:
            raise ValueError("Optimization Infeasible: You likely have fewer than 5 baskets.")
        
        optimized_weights = pd.Series(w.value, index=baskets.index, name="Optimized_Weight")
        return optimized_weights

        
    
    
# 1. Tech Giants (Software & Platform)
tech_tickers = ['MSFT', 'AAPL', 'GOOGL', 'META', 'ADBE']
tech_weights = [0.35, 0.30, 0.15, 0.15, 0.05]

# 2. Semiconductors (High Beta / Growth) - NEW
semi_tickers = ['NVDA', 'AVGO', 'AMD', 'TXN', 'MU']
semi_weights = [0.40, 0.25, 0.15, 0.10, 0.10]

# 3. Financials (Banking & Payments)
fin_tickers = ['JPM', 'V', 'MA', 'BAC', 'GS']
fin_weights = [0.30, 0.25, 0.20, 0.15, 0.10]

# 4. Healthcare (Pharma & Managed Care)
health_tickers = ['UNH', 'LLY', 'JNJ', 'ABBV', 'MRK']
health_weights = [0.25, 0.25, 0.20, 0.15, 0.15]

# 5. Consumer Staples (Defensive)
staples_tickers = ['PG', 'COST', 'WMT', 'KO', 'PEP']
staples_weights = [0.25, 0.20, 0.20, 0.20, 0.15]

# 6. Energy (Oil & Gas)
energy_tickers = ['XOM', 'CVX', 'COP', 'EOG', 'MPC']
energy_weights = [0.40, 0.30, 0.10, 0.10, 0.10]

# 7. Utilities (Infrastructure)
util_tickers = ['NEE', 'SO', 'DUK', 'SRE', 'AEP']
util_weights = [0.30, 0.20, 0.20, 0.15, 0.15]

# 8. Consumer Discretionary (Retail & Auto)
disc_tickers = ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE']
disc_weights = [0.40, 0.20, 0.15, 0.15, 0.10]

# Initialize Baskets
# Setup Benchmark
benchmark = 'SPY'

# Create Basket Instances
baskets_list = [
    Basket("Tech", tech_weights, tech_tickers, benchmark),
    Basket("Semis", semi_weights, semi_tickers, benchmark),
    Basket("Financials", fin_weights, fin_tickers, benchmark),
    Basket("Healthcare", health_weights, health_tickers, benchmark),
    Basket("Staples", staples_weights, staples_tickers, benchmark),
    Basket("Energy", energy_weights, energy_tickers, benchmark),
    Basket("Utilities", util_weights, util_tickers, benchmark),
    Basket("Discretionary", disc_weights, disc_tickers, benchmark)
]

# Initialize Portfolio
aachen_capital = Portfolio("AACHEN_V2")
for b in baskets_list:
    aachen_capital.add_basket(b)

# Run Optimization (Lookback: 2021 to 2026)
start_date = '2021-01-01'
end_date = '2026-01-01'

try:
    optimizer = Optimizer(aachen_capital)
    final_weights = optimizer.optimized_weights(start_date, end_date)
    
    print("--- Optimized Weights Across Baskets ---")
    print(final_weights.round(4))
    
except Exception as e:
    print(f"Error: {e}")



            

    
    