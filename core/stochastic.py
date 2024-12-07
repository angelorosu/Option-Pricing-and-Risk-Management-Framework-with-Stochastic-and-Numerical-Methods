import numpy as np
from scipy.stats import gamma
class StochasticModel:
    '''Base Class for all other models to use'''
    def __init__(self,S0,T,steps,paths):
        self.S0 = S0
        self.T = T
        self.steps = steps
        self.dt = T/steps
        self.paths = paths
        self.X = None  # Log returns 
        self.S = None  # Simulated prices

    def generate_path(self):
        raise NotImplementedError("This method defined in each subclass for the models")
    

class GBMModel(StochasticModel):
    """
    Simulate price paths using Geometric Brownian Motion.

    Returns:
        np.ndarray: Simulated prices of shape (paths, steps+1).
    """
    def __init__(self,S0,mu,sigma,T,steps,paths):
        super().__init__(S0,T,steps,paths)
        self.mu = mu
        self.sigma = sigma

    def generate_path(self):
        '''Simulate teh GBM price paths'''
        dX = (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.random.randn(self.paths, self.steps) * np.sqrt(self.dt)

        # Accumulate the increments
        self.X = np.hstack([np.zeros((self.paths, 1)), np.cumsum(dX, axis=1)])

        # Transform to geometric Brownian motion
        self.S = self.S0 * np.exp(self.X)

        return self.S
    
class HestonModel(StochasticModel):
    '''
    Simulate price paths using Heston Model

    returns:
    np.ndarray: simulated prices of shape(paths,steps+1)
    '''

    def __init__(self, S0, T, steps, paths,v0,kappa,theta,rho,mu,sigma):
        super().__init__(S0, T, steps, paths)
        self.v0 = v0
        self.kappa = kappa 
        self.theta = theta
        self.rho = rho
        self.mu = mu
        self.sigma = sigma

    def generate_path(self):

        cov_matrix = np.array([[1,self.rho],[self.rho,1]]) #correlation of the two stochastic processes
        L = np.linalg.cholesky(cov_matrix)

        # initialising paths
        S = np.zeros((self.paths,self.steps+1))
        v = np.zeros((self.paths,self.steps+1))
        S[:,0] = self.S0
        v[:,0] = self.v0

        for i in range(1,self.steps+1):
            # generate correlated random numbers
            Z = np.random.randn(self.paths,2)
            dW = np.dot(Z,L.T)* np.sqrt(self.dt)

            # Variance process (CIR dynamics)
            v_prev = np.maximum(v[:, i-1], 0)
            v[:, i] = np.maximum(
                v_prev + self.kappa * (self.theta - v_prev) * self.dt + self.sigma * np.sqrt(v_prev) * dW[:, 1],
                0
            )
            
            # Asset price process (multiplicative form)
            S[:, i] = S[:, i-1] * np.exp(
                (self.mu - 0.5 * v_prev) * self.dt + np.sqrt(v_prev) * dW[:, 0]
            )

        return S,v



class VarianceGammaModel(StochasticModel):
    def __init__(self, S0, T, steps, paths,mu,sigma,kappa):
        super().__init__(S0, T, steps, paths)
        self.sigma = sigma
        self.kappa = kappa
        self.mu = mu

    def generate_path(self):
        # Compute increments of the gamma process
        dG = gamma.rvs(a=self.dt / self.kappa, scale=self.kappa, size=(self.steps, self.paths))

        # Compute the increments of the ABM on the gamma random clock
        dX = self.mu * dG + self.sigma * np.random.randn(self.steps, self.paths) * np.sqrt(dG)

        # Accumulate the increments
        X = np.vstack([np.zeros(self.paths), np.cumsum(dX, axis=0)])

        # Transform to price paths
        S = self.S0 * np.exp(X)

        # Transpose to ensure shape (paths, steps + 1)
        return S.T
