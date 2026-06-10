from engine import PortfolioManager, Basket
from analysis import Analysis
import IPython.display as display
from visuals import Visuals
import pandas as pd
import numpy as np
from barra_factor import BarraFactor
from data import DataLoaderYfinance, DataProcessor
from optimiser import Optimiser

baskets = [
    # --- Semiconductors ---
    Basket("Semiconductors", [0.5, 0.5],
           ['SMH', 'AIXA.DE'], 'SPY'),


    Basket("Utilities", [1.0], ['XLU'], 'SPY'),

    Basket("Gold", [1.0], ['GLD'], 'SPY'),
    
    
    # --- Defence ---
    Basket("Defence EU", [1],
           ['EUAD'], 'SPY'),
    
    
    Basket("Hyperliquid Strategies", [1],
           ['PURR'], 'SPY'),
    
    Basket("Copper", [1],
           ['COPX'], 'SPY'),
]
    


RIC_MAP = {
    'AAPL': 'AAPL.O', 'MSFT': 'MSFT.O', 'NVDA': 'NVDA.O', 'GOOGL': 'GOOGL.O',
    'AMZN': 'AMZN.O', 'META': 'META.O', 'COST': 'COST.O', 'QCOM': 'QCOM.O',
    'TXN':  'TXN.O',  'ADI':  'ADI.O',  'AMAT': 'AMAT.O',
    'JPM':  'JPM.N',  'BAC':  'BAC.N',  'GS':   'GS.N',   'MS':   'MS.N',
    'WFC':  'WFC.N',  'C':    'C.N',    'JNJ':  'JNJ.N',  'UNH':  'UNH.N',
    'ABBV': 'ABBV.N', 'MRK':  'MRK.N',  'LLY':  'LLY.N',  'TMO':  'TMO.N',
    'PG':   'PG.N',   'KO':   'KO.N',   'PEP':  'PEP.N',  'WMT':  'WMT.N',
    'CL':   'CL.N',   'NEE':  'NEE.N',  'DUK':  'DUK.N',  'SO':   'SO.N',
    'D':    'D.N',    'AEP':  'AEP.N',  'SRE':  'SRE.N',  'XOM':  'XOM.N',
    'CVX':  'CVX.N',  'COP':  'COP.N',  'SLB':  'SLB.N',  'EOG':  'EOG.N',
    'MPC':  'MPC.N',  'GE':   'GE.N',   'CAT':  'CAT.N',  'HON':  'HON.N',
    'MMM':  'MMM.N',  'UPS':  'UPS.N',  'LMT':  'LMT.N',
}

NASDAQ = ['AAPL.O', 'MSFT.O', 'NVDA.O', 'GOOGL.O', 'AMZN.O', 'META.O',
          'COST.O', 'QCOM.O', 'TXN.O', 'ADI.O', 'AMAT.O']
NYSE   = ['JPM.N', 'BAC.N', 'GS.N', 'MS.N', 'WFC.N', 'C.N', 'JNJ.N',
          'UNH.N', 'ABBV.N', 'MRK.N', 'LLY.N', 'TMO.N', 'PG.N', 'KO.N',
          'PEP.N', 'WMT.N', 'CL.N', 'NEE.N', 'DUK.N', 'SO.N', 'D.N',
          'AEP.N', 'SRE.N', 'XOM.N', 'CVX.N', 'COP.N', 'SLB.N', 'EOG.N',
          'MPC.N', 'GE.N', 'CAT.N', 'HON.N', 'MMM.N', 'UPS.N', 'LMT.N']
'''
data = DataLoaderLSEG()
data.open_session_lseg()

universe     = list(set(ticker for basket in baskets for ticker in basket.tickers))
ric_universe = [RIC_MAP.get(t, t) for t in universe]

df = data.get_barra_fundamentals_lseg(universe=ric_universe, nasdaq=NASDAQ, nyse=NYSE)

processor = DataProcessor()
df = processor.process_lseg(df)

print("\n--- DATA CHECK ---")
print(f"DataFrame Shape: {df.shape}")
print("Columns:", df.columns.tolist())
print("------------------\n")

barra = BarraFactor(df)

print("Calculating factor returns...")
factor_returns = barra.calculate_factor_returns()

print("Calculating risk matrix...")
history = barra.factor_return_history()
barra.calculate_ewma_risk_matrix(history, halflife=60)

attribution = barra.match_to_baskets(baskets)
risk_metrics = barra.risk_results(baskets)

for basket in baskets:
    if basket.name in attribution:
        print(f"\n--- {basket.name.upper()} ---")
        pred_ret = attribution[basket.name]['Total_Return']
        print(f"Factor-Predicted Return: {pred_ret:.6f}")
        sys_vol = risk_metrics[basket.name]['Systematic_Vol']
        print(f"Annualized Systematic Vol: {sys_vol:.2%}")
        print("Top 3 Risk Drivers:")
        risk_contrib = risk_metrics[basket.name]['Risk_Contribution']
        top_risks = risk_contrib.abs().sort_values(ascending=False).head(3)
        for factor in top_risks.index:
            print(f"  > {factor}: {risk_contrib[factor]:.6f}")
    else:
        print(f"\n--- {basket.name} skipped (No matching data found) ---")'''
        
        
Manager = PortfolioManager(baskets, start_date="2020-01-01", end_date="2026-01-01")
basket_stats = Manager.calculate_basket_metrics()

result = Optimiser.get_weights(basket_stats, max_beta=1.0, risk_aversion=0.5)

weights       = result['weights']
final_beta    = result['max_beta']
final_ra      = result['risk_aversion']
final_max_w   = result['max_weight']
attempts      = result['attempts']

# Per-basket breakdown
df = pd.DataFrame({
    'Basket':     [b['Name']       for b in basket_stats],
    'Weight':     weights,
    'Ann Return': [b['Ann Return'] for b in basket_stats],
    'Beta':       [b['Beta']       for b in basket_stats],
}).sort_values('Weight', ascending=False).reset_index(drop=True)

# Portfolio-level
port_return = float(np.dot(weights, [b['Ann Return'] for b in basket_stats]))
port_beta   = float(np.dot(weights, [b['Beta']       for b in basket_stats]))

print("\n=== Optimised Portfolio ===")
print(df.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

print("\n--- Realised constraints ---")
print(f"Final risk aversion:   {final_ra:.2f}")
print(f"Final max beta cap:    {final_beta:.2f}")
print(f"Final max weight cap:  {final_max_w:.2f}")
print(f"Attempts to converge:  {attempts}")

print("\n--- Portfolio metrics ---")
print(f"Expected return: {port_return:.2%}")
print(f"Realised beta:   {port_beta:.3f}")
print(f"Sum of weights:  {weights.sum():.4f}")