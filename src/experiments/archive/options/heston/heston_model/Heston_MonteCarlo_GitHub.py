#https://calebmigosi.medium.com/build-the-heston-model-from-scratch-in-python-part-iii-monte-carlo-pricing-5009c1ba7d29#6d49

# implementation of MC
import numpy as np

def MCHeston(St, K, T,r,
             sigma, kappa, theta, volvol, rho,
             iterations, timeStepsPerYear):
    timesteps = int(T * timeStepsPerYear)
    dt = 1 / timeStepsPerYear
    # Define the containers to hold values of St and Vt
    S_t = np.zeros((timesteps+1, iterations))
    V_t = np.zeros((timesteps+1, iterations))
    # Assign first value of all Vt to sigma
    V_t[0, :] = sigma
    S_t[0, :] = St
    # Use Cholesky decomposition to
    means = [0, 0]
    stdevs = [1 / 3, 1 / 3]
    covs = [[stdevs[0] ** 2, stdevs[0] * stdevs[1] * rho],
            [stdevs[0] * stdevs[1] * rho, stdevs[1] ** 2]]
    Z = np.random.multivariate_normal(means,
                                      covs, (iterations, timesteps)).T
    Z1 = Z[0]
    Z2 = Z[1]
    for i in range(1, timesteps):
        # Use Z2 to calculate Vt
        V_t[i, :] = np.maximum(V_t[i - 1, :] +
                               kappa * (theta - V_t[i - 1, :]) * dt +
                               volvol * np.sqrt(V_t[i - 1, :] * dt) * Z2[i, :], 0)

        # Use all V_t calculated to find the value of S_t
        S_t[i, :] = S_t[i - 1, :] + r * S_t[i, :] * dt +  np.sqrt(V_t[i, :] * dt) * S_t[i - 1, :] * Z1[i, :]

    return np.exp(-r * T) * np.mean(S_t[timesteps - 1, :] - K)

S0 = 248.0    # Initial stock price
K = 250.0     # Strike price
r = 0.03      # Risk-free rate
T = 1.0       # Time to maturity
kappa = 3.0   # Mean reversion rate
theta = 0.05  # Long-term average volatility
sigma = 0.3   # Volatility of volatility
rho = -0.6    # Correlation coefficient
v0 = 0.05     # Initial volatility

num_paths = 10000  # Number of Monte Carlo paths
num_steps = 252  # Number of time steps (daily)


price = MCHeston(S0, K, T, r, kappa, theta, sigma, rho, v0, num_paths, num_steps)

# Output results
print(f"Simulated European Call Option Price: {price}")