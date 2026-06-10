'''unscented kalman filter (non-linear)

inputs are price series, vol series, stress series, same as adaptive kalman'''

import numpy as np
import pandas as pd

from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints


class AdaptiveUKF:

    def __init__(
        self,
        alpha=0.1,
        beta=2.0,
        kappa=0.0,
        base_process_var=0.01,
        base_observation_var=1.0
    ):

        self.base_process_var = base_process_var
        self.base_observation_var = base_observation_var

        # ----------------------------------------
        # Sigma points
        # ----------------------------------------

        self.points = MerweScaledSigmaPoints(
            n=2,
            alpha=alpha,
            beta=beta,
            kappa=kappa
        )

        # ----------------------------------------
        # Build UKF
        # ----------------------------------------

        self.ukf = UnscentedKalmanFilter(
            dim_x=2,
            dim_z=1,
            dt=1.0,
            fx=self.fx,
            hx=self.hx,
            points=self.points
        )

        # ----------------------------------------
        # Initial state
        # ----------------------------------------

        self.ukf.x = np.array([
            0.0,   # trend level
            0.0    # trend slope
        ])

        # Initial covariance
        self.ukf.P *= 1.0

        # Initial process covariance
        self.ukf.Q = self.base_process_var * np.eye(2)

        # Initial observation covariance
        self.ukf.R = np.array([
            [self.base_observation_var]
        ])

    # ==================================================
    # Nonlinear State Transition Function
    # ==================================================

    def fx(self, x, dt):

        level = x[0]
        slope = x[1]

        # ----------------------------------------
        # Nonlinear trend dynamics
        # ----------------------------------------

        new_level = level + slope * dt

        # nonlinear slope persistence
        new_slope = (
            0.95 * slope
            - 0.05 * slope**3
        )

        return np.array([
            new_level,
            new_slope
        ])

    # ==================================================
    # Observation Function
    # ==================================================

    def hx(self, x):

        # observed price = latent level + noise
        return np.array([
            x[0]
        ])

    # ==================================================
    # Main Filter
    # ==================================================

    def filter(
        self,
        prices,
        volatility=None,
        stress=None
    ):

        prices = pd.Series(prices)

        # ----------------------------------------
        # Defaults
        # ----------------------------------------

        if volatility is None:
            volatility = pd.Series(
                np.ones(len(prices)),
                index=prices.index
            )

        if stress is None:
            stress = pd.Series(
                np.zeros(len(prices)),
                index=prices.index
            )

        # initialise state from first observation so filter doesn't diverge
        self.ukf.x = np.array([float(prices.iloc[0]), 0.0])
        self.ukf.P = np.diag([float(prices.iloc[0]) ** 2 * 0.01, 1.0])

        levels = []
        slopes = []
        residuals = []

        # ==================================================
        # Filtering Loop
        # ==================================================

        for t in range(len(prices)):

            y = prices.iloc[t]

            vol_t = float(volatility.iloc[t])
            stress_t = float(stress.iloc[t])

            # ----------------------------------------
            # Adaptive Observation Covariance
            # ----------------------------------------

            self.ukf.R = np.array([
                [self.base_observation_var * (1 + vol_t)]
            ])

            # ----------------------------------------
            # Adaptive Process Covariance
            # ----------------------------------------

            q_scale = self.base_process_var * (1 + stress_t)
            self.ukf.Q = q_scale * np.eye(2)

            # ----------------------------------------
            # Enforce P positive-definite via eigenvalue clipping
            # ----------------------------------------

            P_sym = (self.ukf.P + self.ukf.P.T) / 2
            eigvals, eigvecs = np.linalg.eigh(P_sym)
            eigvals = np.maximum(eigvals, 1e-4)
            self.ukf.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # ----------------------------------------
            # Predict
            # ----------------------------------------

            self.ukf.predict()

            # ----------------------------------------
            # Update
            # ----------------------------------------

            self.ukf.update(y)

            # ----------------------------------------
            # Store Outputs
            # ----------------------------------------

            levels.append(self.ukf.x[0])
            slopes.append(self.ukf.x[1])

            residual = y - self.ukf.x[0]
            residuals.append(residual)

        # ==================================================
        # Output DataFrame
        # ==================================================

        out = pd.DataFrame(index=prices.index)

        out["trend"] = levels
        out["slope"] = slopes
        out["acceleration"] = out["slope"].diff()
        out["residual"] = residuals

        return out