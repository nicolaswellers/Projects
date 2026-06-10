'''partial corr: this measures the correlation of residuals between securities

Idea: market beta is very prevalent, so we try to remove and control for this
We take the residuals of the regression of each security on SPY, and then we measure the correlation of these residuals

x and y are the securities we are investigating,
control is the underlying factor we are trying to eliminate

'''

import numpy as np
import pandas as pd
import statsmodels.api as sm

class PartialCorr:
    def __init__ (self, x, y, control):
        self.x = x
        self.y = y
        self.control = control
        
    def log_rets(self):
        x_rets = np.log(self.x / self.x.shift(1)).dropna()
        y_rets = np.log(self.y / self.y.shift(1)).dropna()
        control_rets = np.log(self.control / self.control.shift(1)).dropna()
        return x_rets, y_rets, control_rets
    
        
    def standardise(self, x_rets, y_rets, control_rets):
        x_rets_z = (x_rets - x_rets.mean()) / x_rets.std()
        y_rets_z = (y_rets - y_rets.mean()) / y_rets.std()
        control_rets_z = (control_rets - control_rets.mean()) / control_rets.std()
        return x_rets_z, y_rets_z, control_rets_z
    
    def regress(self, x_rets_z, y_rets_z, control_rets_z):
        x_model = sm.OLS(x_rets_z, sm.add_constant(control_rets_z)).fit()
        y_model = sm.OLS(y_rets_z, sm.add_constant(control_rets_z)).fit()
        return x_model.resid, y_model.resid
    
    def ewma_corr(self, x_resid, y_resid, span=60):
        x_ewma = x_resid.ewm(span=span).mean()
        y_ewma = y_resid.ewm(span=span).mean()
        corr = x_ewma.corr(y_ewma)
        return corr
    
    def compute_partial_corr(self):
        x_rets, y_rets, control_rets = self.log_rets()
        x_rets_z, y_rets_z, control_rets_z = self.standardise(x_rets, y_rets, control_rets)
        x_resid, y_resid = self.regress(x_rets_z, y_rets_z, control_rets_z)
        partial_corr = self.ewma_corr(x_resid, y_resid)
        return partial_corr
    
        
