'''Trend extraction kalman:
2D (position and velocity)

input is a price series, 
'''

import numpy as np
import pandas as pd
from pykalman import KalmanFilter

class TwoDKalman:

    def __init__(self):
        
        # State transition matrix
        self.transition_matrix = np.array([
            [1, 1],
            [0, 1]
        ])

        # Observation matrix
        self.observation_matrix = np.array([
            [1, 0]
        ])

        self.kf = KalmanFilter(
            transition_matrices=self.transition_matrix,
            observation_matrices=self.observation_matrix,
            initial_state_mean=[0, 0],
            observation_covariance=1,
            transition_covariance=0.01 * np.eye(2)
        )

    def filter(self, prices):

        state_means, state_covs = self.kf.filter(prices.values)

        trend = pd.Series(
            state_means[:, 0],
            index=prices.index
        )

        slope = pd.Series(
            state_means[:, 1],
            index=prices.index
        )

        return trend, slope
    
    