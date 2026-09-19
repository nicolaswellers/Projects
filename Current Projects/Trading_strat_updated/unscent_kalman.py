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
        '''Walk-forward filter: the value reported for date t is the
        *predicted* (a priori) state -- built from the posterior at t-1
        propagated forward through fx, before today's observation is folded
        in. That means the output at t depends only on information through
        t-1, never on today's own price. Today's price is only incorporated
        afterward, via update(), so it can inform the prediction for t+1.

        The very first date has no t-1 to predict from (the state has to be
        seeded from that date's own price), so it carries no walk-forward
        output and is reported as NaN.

        Prices are normalised to their own first value before filtering, so
        every series starts at 1.0 regardless of whether the instrument
        trades at $5 or $5,000 -- base_process_var/base_observation_var are
        fixed absolute constants, so without this a high-priced stock's
        ordinary day-to-day moves dwarf those variances and the filter's
        covariance (and the cubic slope term in fx()) can diverge. "trend",
        "slope" and "residual" are therefore reported as fractions of the
        starting price, not raw price units.
        '''

        prices = pd.Series(prices)
        p0 = float(prices.iloc[0])
        if p0 <= 0:
            raise ValueError(f"cannot normalise: first price is non-positive ({p0})")
        prices = prices / p0

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

        # seed state from the first observation -- unavoidable cold start,
        # since there is no earlier data to predict it from
        self.ukf.x = np.array([float(prices.iloc[0]), 0.0])
        self.ukf.P = np.diag([float(prices.iloc[0]) ** 2 * 0.01, 1.0])
        self.ukf.R = np.array([[self.base_observation_var * (1 + float(volatility.iloc[0]))]])
        self.ukf.Q = self.base_process_var * (1 + float(stress.iloc[0])) * np.eye(2)
        self.ukf.update(prices.iloc[0])

        levels    = [np.nan]
        slopes    = [np.nan]
        residuals = [np.nan]

        # ==================================================
        # Filtering Loop
        # ==================================================

        for t in range(1, len(prices)):

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
            # Predict: state at t from data through t-1 only
            # ----------------------------------------

            self.ukf.predict()

            # ----------------------------------------
            # Store Outputs (the a priori / predicted state -- NOT the
            # post-update state below, which would already know today's y)
            # ----------------------------------------

            level_prior = self.ukf.x[0]
            slope_prior = self.ukf.x[1]

            levels.append(level_prior)
            slopes.append(slope_prior)
            residuals.append(y - level_prior)

            # ----------------------------------------
            # Update: fold today's price in for *tomorrow's* predict only
            # ----------------------------------------

            self.ukf.update(y)

        # ==================================================
        # Output DataFrame
        # ==================================================

        out = pd.DataFrame(index=prices.index)

        out["trend"] = levels
        out["slope"] = slopes
        out["acceleration"] = out["slope"].diff()
        out["residual"] = residuals

        return out