import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.stochastic import GBMModel
from core.stochastic import HestonModel

# Add the root project directory to the Python path


def test_gbm_generate_paths():
    # PARAMETERS
    S0 = 100
    T = 1
    steps = 252
    paths = 1000
    mu = 0.05
    sigma = 0.2 

    # Initialise and generate paths
    gbm = GBMModel(S0,mu,sigma,T,steps,paths)
    price_paths = gbm.generate_path()

    # Test 1: Correct shape
    assert price_paths.shape == (paths, steps + 1), "Shape of GBM paths is incorrect"

    # Test 2: Initial prices match S0
    assert np.all(price_paths[:, 0] == S0), "Initial prices do not match S0"

    # Test 3: All prices are non-negative
    assert np.all(price_paths >= 0), "GBM paths contain negative prices"

     # Extract final prices
    final_prices = price_paths[:, -1]

    # Theoretical mean and variance
    theoretical_mean = S0 * np.exp(mu * T)
    theoretical_variance = (S0**2) * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)

    # Empirical mean and variance
    empirical_mean = np.mean(final_prices)
    empirical_variance = np.var(final_prices)

    # Check mean and variance
    assert np.isclose(empirical_mean, theoretical_mean, rtol=0.01), "GBM mean does not match theoretical mean"
    assert np.isclose(empirical_variance, theoretical_variance, rtol=0.05), "GBM variance does not match theoretical variance"


def test_heston_generate_paths():

    # PARAMETERS
    S0 = 100       # Initial price
    v0 = 0.04      # Initial variance
    kappa = 2.0    # Speed of mean reversion
    theta = 0.04   # Long-run variance
    rho = -0.7     # Correlation between Brownian motions
    sigma = 0.2    # Volatility of variance
    mu = 0.05      # Drift
    T = 1          # Time to maturity
    steps = 252    # Time steps
    paths = 1000   # Number of simulation paths

    # Generate Heston paths
    heston = HestonModel(S0, T, steps, paths, v0, kappa, theta, rho, mu, sigma)
    price_paths, variance_paths = heston.generate_path()

    # TEST 1: Shape of paths
    assert price_paths.shape == (paths, steps + 1), "Shape of Heston price paths is incorrect"
    assert variance_paths.shape == (paths, steps + 1), "Shape of Heston variance paths is incorrect"

    # TEST 2: Initial prices and variances
    assert np.all(price_paths[:, 0] == S0), "Initial prices do not match S0"
    assert np.all(variance_paths[:, 0] == v0), "Initial variances do not match v0"

    # TEST 3: Non-negativity
    assert np.all(price_paths >= 0), "Heston price paths contain negative prices"
    assert np.all(variance_paths >= 0), "Heston variance paths contain negative variances"

    # TEST 4: Statistical properties of variance
    # Check if the average variance approaches the long-run mean theta
    average_variance = np.mean(variance_paths[:, -1])
    assert np.isclose(average_variance, theta, rtol=0.1), "Average variance does not approach theta"

    # TEST 5: Correlation
    # Correlation between dW_S and dW_v
    log_returns = np.diff(np.log(price_paths), axis=1)
    variance_changes = np.diff(variance_paths, axis=1)

    empirical_corr = np.corrcoef(log_returns.flatten(), variance_changes.flatten())[0, 1]
    assert np.isclose(empirical_corr, rho, rtol=0.1), "Empirical correlation does not match input rho"

def test_variance_gamma_generate_paths():
    import numpy as np
    from core.stochastic import VarianceGammaModel

    # PARAMETERS
    S0 = 100        # Initial price
    mu = 0.05       # Drift
    sigma = 0.2     # Volatility
    kappa = 0.1     # Scale parameter of the Gamma process
    T = 1           # Time to maturity
    steps = 252     # Time steps
    paths = 1000    # Number of simulation paths

    # Generate Variance Gamma paths
    vg = VarianceGammaModel(S0, T, steps, paths, mu, sigma, kappa)
    price_paths = vg.generate_path()

    # TEST 1: Shape of paths
    assert price_paths.shape == (paths, steps + 1), "Shape of Variance Gamma paths is incorrect"

    # TEST 2: Initial prices
    assert np.all(price_paths[:, 0] == S0), "Initial prices do not match S0"

    # TEST 3: Non-negativity
    assert np.all(price_paths >= 0), "Variance Gamma paths contain negative prices"

    # TEST 4: Statistical properties of final prices
    final_prices = price_paths[:, -1]
    empirical_mean = np.mean(final_prices)
    empirical_variance = np.var(final_prices)

    # Check if mean and variance are reasonable
    # (These depend on VG parameters, so exact theoretical values are hard to validate)
    assert empirical_mean > S0, "Empirical mean is unexpectedly low"
    assert empirical_variance > 0, "Empirical variance is unexpectedly low"
