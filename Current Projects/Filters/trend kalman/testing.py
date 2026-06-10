'''this is the testing and evaluation document

I am evaluating trend filters according to:
1. accuracy of trend detection (compared to HP filter) 
2. lag (how quickly does it respond to changes in trend, time to react to HP filter)
3. noise robustness (how well does it perform in the presence of noise)
4. parameter stability (how sensitive is it to changes in parameters)


1. will be evaluated on square error
2. will be evalauted by cross correlation between HP change and filter change
3. noise is inherent in S and P, I will penalise for false positives in regard to HP filter
4. here i will use grid search ------ STILL IN PROGRESS


I can also compare and evaluate the performance of other benchmarks to this to compare with my kalman filters
'''
import numpy as np
class Testing:
    def RSE(self, filter_trend, hp_trend, filter_slope, hp_slope):
        trend_error = np.sum((filter_trend - hp_trend)**2)
        slope_error = np.sum((filter_slope - hp_slope)**2)
        return {"trend_RSE": trend_error, "slope_RSE": slope_error}
    
    def lag(self, filter_slope, hp_slope):
        # cross correlation to find lag
        correlation = np.correlate(filter_slope, hp_slope, mode='full')
        lag = correlation.argmax() - (len(filter_slope) - 1)
        return lag
    
    def false_positives(self, filter_slope, hp_slope):
        # count false positives where filter detects a change but HP does not
        false_positives = np.sum((filter_slope != 0) & (hp_slope == 0))
        return false_positives
    
    