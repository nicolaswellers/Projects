'''adaptive kalman filter for trend extraction

dynamically updates Q (process noise covariance) and R (observation noise covariance)


input: price series, volatility series, stress series

stress series = 0.5*abs(credit_shock) + 0.5*abs(liquidity_shock), take from stress and liquidity filter
volatility series = GARCH estimate
price series = SPY close or IWM for example
'''

import numpy as np
import pandas as pd

class AdaptiveKalmanFilter:

    def __init__(self):

        # state: [level, slope]
        self.x = np.array([[0.0],
                           [0.0]])

        self.P = np.eye(2)

        self.F = np.array([
            [1, 1],
            [0, 1]
        ])

        self.H = np.array([[1, 0]])

        self.I = np.eye(2)

    def filter(self, prices, vol, stress):

        levels = []
        slopes = []

        for t in range(len(prices)):

            y = prices.iloc[t]

            # --------------------------------
            # Adaptive covariances
            # --------------------------------

            R = np.array([[1 + vol.iloc[t]]])

            q_scale = 0.01 * (1 + stress.iloc[t])

            Q = q_scale * np.eye(2)

            # --------------------------------
            # Predict
            # --------------------------------

            x_pred = self.F @ self.x

            P_pred = self.F @ self.P @ self.F.T + Q

            # --------------------------------
            # Innovation
            # --------------------------------

            y_pred = self.H @ x_pred

            e = y - y_pred[0, 0]

            S = self.H @ P_pred @ self.H.T + R

            # --------------------------------
            # Kalman Gain
            # --------------------------------

            K = P_pred @ self.H.T @ np.linalg.inv(S)

            # --------------------------------
            # Update
            # --------------------------------

            self.x = x_pred + K * e

            self.P = (self.I - K @ self.H) @ P_pred

            levels.append(self.x[0, 0])
            slopes.append(self.x[1, 0])

        return (
            pd.Series(levels, index=prices.index),
            pd.Series(slopes, index=prices.index)
        )