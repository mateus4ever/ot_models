import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Tesla EOD data ---
df = pd.read_csv("tesla_eod.csv", parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)
prices = df["Close"].values
log_returns = np.log(prices[1:] / prices[:-1])

# --- 2. Estimate basic parameters from real data ---
mu = np.mean(log_returns)
v0 = np.var(log_returns)
kappa = 2.0       # speed of mean reversion
theta = v0        # long-term variance
sigma = 0.5       # volatility of volatility
rho = -0.7        # correlation
dt = 1.0          # 1 trading day
n_paths = 1000
n_days = 100

# --- 3. Simulate Heston paths ---
def simulate_heston_paths(S0, v0, mu, kappa, theta, sigma, rho, T, N, M):
    dt = T / N
    S = np.zeros((M, N+1))
    v = np.zeros((M, N+1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, N + 1):
        z1 = np.random.normal(size=M)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=M)
        v[:, t] = np.abs(v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + sigma * np.sqrt(v[:, t-1] * dt) * z2)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t] * dt) * z1)
    return S

S0 = prices[-1]
simulated_paths = simulate_heston_paths(S0, v0, mu, kappa, theta, sigma, rho, T=n_days, N=n_days, M=n_paths)

# --- 4. Analyze real vs simulated returns ---
real_returns = pd.Series(log_returns[-n_days:])
mean_simulated_path = np.mean(simulated_paths, axis=0)
simulated_returns = np.log(mean_simulated_path[1:] / mean_simulated_path[:-1])
simulated_returns = pd.Series(simulated_returns)

# --- 5. Print stats ---
print("✅ Real returns stats:")
print(f"Mean: {real_returns.mean():.6f}, Std: {real_returns.std():.6f}")

print("\n✅ Simulated (mean path) returns stats:")
print(f"Mean: {simulated_returns.mean():.6f}, Std: {simulated_returns.std():.6f}")

# --- 6. Plot comparison ---
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(real_returns, bins=50, alpha=0.6, label="Real")
plt.hist(simulated_returns, bins=50, alpha=0.6, label="Simulated")
plt.title("Histogram of Log Returns")
plt.xlabel("Log Return")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(real_returns.values, label="Real Returns")
plt.plot(simulated_returns.values, label="Simulated Returns (mean path)")
plt.title("Time Series of Log Returns")
plt.xlabel("Day")
plt.ylabel("Log Return")
plt.legend()

plt.tight_layout()
plt.show()
