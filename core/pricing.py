import numpy as np
from scipy.stats import norm
class OptionPricing:
    '''Base class for Monte Carlo'''
    
    def __init__(self,S0,K,r,T):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T

    def price(self):
        raise NotImplementedError("Subclasses implement this method")
    

class BlackScholesPricing(OptionPricing):
    """Black Scholes for European Option Pricing"""

    def __init__(self, S0, K, r, T ,sigma,option_type='call'):
        super().__init__(S0, K, r, T)
        self.sigma = sigma
        self.option_type = option_type

    def price(self):
        d1 = (np.log(self.S0/self.K))+ (self.r + (0.5 * self.sigma**2)*self.T) 
        d2 = d1 - self.sigma*(np.sqrt(self.T))

        if self.option_type == 'call':
            return self.S0 * norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        elif self.option_type =='put':
            return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        else:
            raise ValueError('Invalid option type!')
        
class MonteCarloPricing(OptionPricing):
    def __init__(self, S0, K, r, T,paths,steps,option_type='call'):
        super().__init__(S0, K, r, T)
        self.paths = paths
        self.steps = steps
        self.option_type = option_type
    
    def price(self,price_paths):
        S_T = price_paths[:,-1]

        if self.option_type == 'call':
            payoffs = np.maximum(S_T-self.K,0)
        elif self.option_type =='put':
            payoffs = np.maximum(self.K - S_T,0)
        else:
            raise ValueError('Invalid option type')
        
        option_price = np.exp(-self.r*self.T) * np.mean(payoffs)
        return option_price

