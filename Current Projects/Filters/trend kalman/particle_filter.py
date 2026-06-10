'''particle filter - distribution of particles through Monte-Carlo estimation

Particle Filter for Latent Trend Extraction
-------------------------------------------

Purpose:
- Nonlinear/non-Gaussian latent trend estimation
- Robust during crashes and regime shifts
- Alternative to Kalman/UKF

input price series, volatility series GARCH
stress index from liquidity and credit filters

'''

import numpy as np
import pandas as pd
from scipy.stats import norm


class ParticleFilter:

    def __init__(
        self,
        n_particles=1000,
        process_noise_level=0.5,
        process_noise_slope=0.05,
        observation_noise=1.0,
        random_state=42
    ):

        np.random.seed(random_state)

        self.n_particles = n_particles

        # process noise
        self.process_noise_level = process_noise_level
        self.process_noise_slope = process_noise_slope

        # observation noise
        self.observation_noise = observation_noise

        # particles:
        # [level, slope]
        self.particles = None

        # weights
        self.weights = np.ones(self.n_particles) / self.n_particles

    # ======================================================
    # Initialize Particles
    # ======================================================

    def initialize(self, initial_price):

        self.particles = np.zeros((self.n_particles, 2))

        # level
        self.particles[:, 0] = (
            initial_price
            + np.random.randn(self.n_particles)
        )

        # slope
        self.particles[:, 1] = (
            0.01 * np.random.randn(self.n_particles)
        )

    # ======================================================
    # State Transition
    # ======================================================

    def predict(self):

        level = self.particles[:, 0]
        slope = self.particles[:, 1]

        # --------------------------------------------
        # Nonlinear dynamics
        # --------------------------------------------

        new_level = (
            level
            + slope
            + self.process_noise_level
            * np.random.randn(self.n_particles)
        )

        # nonlinear slope persistence
        new_slope = (
            0.98 * slope
            - 0.02 * slope**3
            + self.process_noise_slope
            * np.random.randn(self.n_particles)
        )

        self.particles[:, 0] = new_level
        self.particles[:, 1] = new_slope

    # ======================================================
    # Observation Update
    # ======================================================

    def update(self, observation):

        predicted_obs = self.particles[:, 0]

        # likelihood under Gaussian observation noise
        likelihoods = norm.pdf(
            observation,
            loc=predicted_obs,
            scale=self.observation_noise
        )

        # update weights
        self.weights *= likelihoods

        # numerical stability
        self.weights += 1e-300

        self.weights /= np.sum(self.weights)

    # ======================================================
    # Effective Particle Count
    # ======================================================

    def effective_n(self):

        return 1.0 / np.sum(self.weights**2)

    # ======================================================
    # Resampling
    # ======================================================

    def resample(self):

        cumulative_sum = np.cumsum(self.weights)

        cumulative_sum[-1] = 1.0

        indexes = np.searchsorted(
            cumulative_sum,
            np.random.rand(self.n_particles)
        )

        self.particles = self.particles[indexes]

        self.weights.fill(1.0 / self.n_particles)

    # ======================================================
    # Estimate State
    # ======================================================

    def estimate(self):

        mean = np.average(
            self.particles,
            weights=self.weights,
            axis=0
        )

        return mean

    # ======================================================
    # Main Filtering Loop
    # ======================================================

    def filter(
        self,
        prices,
        adaptive_volatility=None,
        adaptive_stress=None
    ):

        prices = pd.Series(prices)

        # optional adaptive inputs
        if adaptive_volatility is None:
            adaptive_volatility = pd.Series(
                np.ones(len(prices)),
                index=prices.index
            )

        if adaptive_stress is None:
            adaptive_stress = pd.Series(
                np.zeros(len(prices)),
                index=prices.index
            )

        # initialize
        self.initialize(prices.iloc[0])

        trend_estimates = []
        slope_estimates = []
        residuals = []

        # ==================================================
        # Sequential Monte Carlo Loop
        # ==================================================

        for t in range(len(prices)):

            y = prices.iloc[t]

            vol_t = float(adaptive_volatility.iloc[t])
            stress_t = float(adaptive_stress.iloc[t])

            # ------------------------------------------
            # Adaptive Noise Scaling
            # ------------------------------------------

            self.process_noise_level = (
                0.5 * (1 + stress_t)
            )

            self.process_noise_slope = (
                0.05 * (1 + stress_t)
            )

            self.observation_noise = (
                1.0 * (1 + vol_t)
            )

            # ------------------------------------------
            # Predict
            # ------------------------------------------

            self.predict()

            # ------------------------------------------
            # Update
            # ------------------------------------------

            self.update(y)

            # ------------------------------------------
            # Resample if particle degeneracy
            # ------------------------------------------

            if self.effective_n() < self.n_particles / 2:
                self.resample()

            # ------------------------------------------
            # Estimate State
            # ------------------------------------------

            state = self.estimate()

            trend = state[0]
            slope = state[1]

            trend_estimates.append(trend)
            slope_estimates.append(slope)

            residuals.append(y - trend)

        # ==================================================
        # Output DataFrame
        # ==================================================

        out = pd.DataFrame(index=prices.index)

        out["trend"] = trend_estimates
        out["slope"] = slope_estimates
        out["acceleration"] = out["slope"].diff()
        out["residual"] = residuals

        return out


