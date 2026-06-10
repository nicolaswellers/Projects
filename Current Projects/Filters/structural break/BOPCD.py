'''BOPCD
this estimates the probability of a changepoint

we start by looking at how long we have been in the current regime, considering all data

P(r_t | x_1:t) 

The model updates at every step:

A) Growth: P(r_t = r_{t-1} + 1)
B) Changepoint: P(r_t = 0)

so the total probability is given by:

P(r_t,x_t) = growth (no break) and cp (break)

We want to predict the likelihood of a new point:

P(x_t | current regime)

in the growth case:

P (r_t = r +1) = P(r_{t-1} = r) * P(x_t | current regime) * (1 - hazard)

in the changepoint case:

P(r_t = 0) = sum_{r} P(r_{t-1} = r) * P(x_t | current regime) * hazard

where the hazard is the expected regime length
1/50 for short-term, 1/150 for medium-term, 1/300 for long-term

I want to consider fat tails, so I will use a Student-t likelihood approximation

student-t likelihood is given by:

P(x_t | current regime) = (1 + (x_t - mu)^2 / (nu * var))^(-(nu + 1) / 2)

we continue to iterate this model each day, for several inputs, including:

credit spreads
liquidity shock
vol regime
correlation breakdown

'''


import numpy as np
import pandas as pd

class StudentTBOCPD:
    """
    BOCPD with Student-t likelihood approximation via
    adaptive variance (scale mixture behavior)
    """

    def __init__(self, hazard=1/150, nu=5):
        self.hazard = hazard
        self.nu = nu  # degrees of freedom (fat tail control)
        self.reset()

    def reset(self):
        self.R = np.array([1.0])

        # regime parameters
        self.mu = 0.0
        self.kappa = 1.0
        self.alpha = 1.0
        self.beta = 1.0

        self.cp_probs = []

    # -------------------------
    # Student-t predictive step
    # -------------------------
    def predictive(self):
        mean = self.mu

        # key difference: heavier tails via inflated variance
        scale = self.beta / (self.alpha * self.kappa)
        var = scale * (self.nu / (self.nu - 2 + 1e-9))  # Student-t variance inflation

        return mean, var

    # -------------------------
    # Update posterior params
    # -------------------------
    def update(self, x):
        self.kappa += 1
        delta = x - self.mu
        self.mu += delta / self.kappa

        self.alpha += 0.5
        self.beta += 0.5 * delta**2

    # -------------------------
    # One step update
    # -------------------------
    def step(self, x):
        x = float(x)

        mean, var = self.predictive()

        # Student-t like likelihood (robust tail)
        likelihood = (1 + (x - mean)**2 / (self.nu * var)) ** (-(self.nu + 1) / 2)

        # growth probabilities
        growth = self.R * likelihood * (1 - self.hazard)

        # changepoint probability
        cp = np.sum(self.R * likelihood * self.hazard)

        # update run length distribution
        new_R = np.zeros(len(growth) + 1)
        new_R[0] = cp
        new_R[1:] = growth

        self.R = new_R / np.sum(new_R)

        # update regime params (assume no changepoint)
        self.update(x)

        self.cp_probs.append(self.R[0])

        return self.R[0]  # return changepoint probability