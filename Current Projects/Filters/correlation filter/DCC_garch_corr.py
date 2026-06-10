''''DCC framework:

I have already developed several GARCH and HAR systems that I will connect to this DCC 

input to DCC: (z_t)

z_t = r_t/sigma_t where r_t is return, sigma_t is garch volatility estimate

ouput: R_t (correlation matrix of entered z signals, N x N)

Q_t is the evolving covariance matrix of z_t, which we can use to derive R_t (correlation matrix)

Q_t = (1 - a - b) * S + a * (z_{t-1} z_{t-1}^T) + b * Q_{t-1}

a = reaction to new shocks
b = persistence of old shocks

S = unconditional covariance of z_t (long-run average correlation) 

(z_{t-1} z_{t-1}^T) = shock term/ short_term co-movement

Q_{t-1} = persistence term/ long-term co-movement

Normalisation step to get into wanted form: 

R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2} 

R_ij,t = Q_i,j,t / sqrt(Q_i,i,t * Q_j,j,t)

which is analogous to how we get correlation from covariance 

corr_ij = cov_ij / (sigma_i * sigma_j)


'''
import numpy as np
class DCC:
    def __init__(self, a=0.02, b=0.97):
        self.a = a
        self.b = b
        self.z = None
        self.N = None
        self.S = None
        self.Q = None
        self.R_history = [] # i want to store the history of R_t for analysis and plotting
    
    def set_inputs(self,z):
        self.z = np.asarray(z)
        T,N = self.z.shape
        self.N = N
        self.S = np.zeros((N, N))
        
        for t in range(T):
            z_t = self.z[t].reshape(-1, 1) # column vector
            self.S += (z_t @ z_t.T) 
        
        self.S /= T
        self.Q = self.S.copy() # initial Q is set to S
        
        
    def fit(self):

        T, N = self.z.shape

        self.Q = self.S.copy()
        self.R_history = []

        for t in range(1, T):

            z_prev = self.z[t-1].reshape(-1, 1)

            # 1. shock term
            shock = z_prev @ z_prev.T

            # 2. DCC update
            self.Q = (
                (1 - self.a - self.b) * self.S
                + self.a * shock
                + self.b * self.Q
            )

            # 3. normalize to correlation matrix R_t
            diag = np.sqrt(np.diag(self.Q))
            R_t = self.Q / np.outer(diag, diag)

            np.fill_diagonal(R_t, 1.0)

            self.R_history.append(R_t)

        return np.array(self.R_history)
    
    