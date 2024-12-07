import numpy as np

class VasicekModel:
    def __init__(self, r0, a, b, sigma, T, steps, paths):
        self.r0 = r0
        self.a = a
        self.b = b
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.paths = paths
        self.dt = T / steps

    def simulate(self):
        rates = np.zeros((self.paths, self.steps + 1))
        rates[:, 0] = self.r0
        for t in range(1, self.steps + 1):
            dW = np.sqrt(self.dt) * np.random.randn(self.paths)
            rates[:, t] = rates[:, t-1] + self.a * (self.b - rates[:, t-1]) * self.dt + self.sigma * dW
        return rates


class CIRModel:
    def __init__(self, r0, a, b, sigma, T, steps, paths):
        self.r0 = r0
        self.a = a
        self.b = b
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.paths = paths
        self.dt = T / steps

    def simulate(self):
        rates = np.zeros((self.paths, self.steps + 1))
        rates[:, 0] = self.r0
        for t in range(1, self.steps + 1):
            dW = np.sqrt(self.dt) * np.random.randn(self.paths)
            rates[:, t] = np.maximum(
                rates[:, t-1] + self.a * (self.b - rates[:, t-1]) * self.dt +
                self.sigma * np.sqrt(np.maximum(rates[:, t-1], 0)) * dW,
                0
            )
        return rates



class BondPricing:
    def __init__(self, rates):
        """
        Initialize the bond pricing system with the simulated rates.
        Parameters:
        - rates (np.ndarray): Simulated interest rates of shape (paths, steps)
        """
        self.rates = rates

    def price_zero_coupon_bond(self, maturity):
        """
        Price a zero-coupon bond maturing at 'maturity'.
        Parameters:
        - maturity (float): The time (in years) at which the bond matures.
        Returns:
        - float: Average price of the zero-coupon bond.
        """
        n_steps = int(maturity * self.rates.shape[1] / self.rates.shape[1])
        discount_factors = np.exp(-np.cumsum(self.rates[:, :n_steps], axis=1) * (maturity / n_steps))
        bond_price = np.mean(discount_factors[:, -1])
        return bond_price

    def price_coupon_bond(self, cashflows, times):
        """
        Price a coupon-bearing bond with given cashflows and times.
        Parameters:
        - cashflows (list): List of cashflows (coupon payments) at each payment time.
        - times (list): List of times at which cashflows are paid.
        Returns:
        - float: Price of the coupon-bearing bond.
        """
        bond_price = 0
        for cashflow, time in zip(cashflows, times):
            n_steps = int(time * self.rates.shape[1] / self.rates.shape[1])
            discount_factors = np.exp(-np.cumsum(self.rates[:, :n_steps], axis=1) * (time / n_steps))
            bond_price += cashflow * np.mean(discount_factors[:, -1])
        return bond_price


class YieldCurve:
    def __init__(self, rates):
        """
        Initialize the yield curve system with simulated interest rates.
        Parameters:
        - rates (np.ndarray): Simulated interest rates of shape (paths, steps)
        """
        self.rates = rates

    def generate_yield_curve(self, maturities):
        """
        Generate a yield curve for multiple maturities.
        Parameters:
        - maturities (list): List of times (in years) for which to compute yields.
        Returns:
        - list: List of yields for each maturity.
        """
        yields = []
        for maturity in maturities:
            n_steps = int(maturity * self.rates.shape[1] / self.rates.shape[1])
            discount_factors = np.exp(-np.cumsum(self.rates[:, :n_steps], axis=1) * (maturity / n_steps))
            bond_price = np.mean(discount_factors[:, -1])
            yield_to_maturity = -np.log(bond_price) / maturity
            yields.append(yield_to_maturity)
        return maturities, yields