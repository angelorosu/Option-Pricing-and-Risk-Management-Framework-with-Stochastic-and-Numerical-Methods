import numpy as np

class VaRCalculator:
    """
    A class to calculate Value-at-Risk (VaR) for a portfolio.
    """
    def __init__(self, returns, alpha=0.05):
        """
        Initialize the VaRCalculator with portfolio returns and confidence level.

        Parameters:
        - returns (array-like): Portfolio returns or losses.
        - alpha (float): Confidence level for VaR (default is 5%).
        """
        self.returns = np.array(returns)
        self.alpha = alpha

    def calculate_var(self):
        """
        Calculate the Value-at-Risk (VaR) at the specified confidence level.

        Returns:
        - float: The VaR value.
        """
        # Sort returns in ascending order
        sorted_returns = np.sort(self.returns)

        # Find the index corresponding to the alpha quantile
        index = int(self.alpha * len(sorted_returns))

        # Return the VaR value
        return sorted_returns[index]
