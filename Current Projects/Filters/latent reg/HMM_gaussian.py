'''HMM to determine latent regimes, inputs from all other filters


MAYBE USE PCA ON FILTERS TO DECIDE WHAT HMM INPUTS ARE BEST?'''
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class GaussianHMMRegimeModel:
  

    def __init__(self, n_states, random_state):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None

   

    def build_features(self, *args):
        """
        Inputs:
        z scores from filters aka features
        """

        df = pd.DataFrame({
            "put features here": args
        }).dropna()

        
        
        features = [
            "here are your features"
        ]

        df = df.dropna()

        self.feature_df = df
        return df

    # ----------------------------
    # Fit HMM
    # ----------------------------

    def fit(self):
        X = self.feature_df.values

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=500,
            random_state=self.random_state
        )

        self.model.fit(X)

        self.feature_df["state"] = self.model.predict(X)

        return self.model, self.feature_df


    def regime_summary(self):
        if self.model is None:
            raise ValueError("Model not fitted yet.")

        return self.feature_df.groupby("state").mean()

    def transition_matrix(self):
        if self.model is None:
            raise ValueError("Model not fitted yet.")

        return pd.DataFrame(self.model.transmat_)