import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Data ---
df = pd.read_csv("tesla_eod.csv", parse_dates=["Datetime"], index_col="Datetime")
df = df.sort_index()
df = df.loc["2023-07-07":"2025-07-07"]
prices = df["Close"]

# --- 2. Calculate Log Returns ---
log_returns = np.log(prices / prices.shift(1)).dropna()
n_days = 50

# --- 3. Heston Model Simulation ---
def simulate_heston(S0, v0, r, kappa, theta, sigma, rho, T, N, M):
    dt = T / N
    prices = np.zeros((M, N+1))
    variances = np.zeros((M, N+1))
    prices[:, 0] = S0
    variances[:, 0] = v0

    for t in range(1, N+1):
        z1 = np.random.normal(size=M)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=M)
        vt_prev = variances[:, t-1]
        vt = np.maximum(vt_prev + kappa * (theta - vt_prev) * dt + sigma * np.sqrt(vt_prev * dt) * z1, 0)
        st = prices[:, t-1] * np.exp((r - 0.5 * vt_prev) * dt + np.sqrt(vt_prev * dt) * z2)
        variances[:, t] = vt
        prices[:, t] = st
    return prices

# --- 4. Estimate Parameters ---
returns_mean = log_returns.mean()
returns_var = log_returns.var()

S0 = prices.iloc[-1]
v0 = returns_var
r = 0.0
kappa = 1.0
theta = returns_var
sigma = 0.1
rho = -0.7
T = n_days / 252
N = n_days
M = 1000

simulated_paths = simulate_heston(S0, v0, r, kappa, theta, sigma, rho, T, N, M)

# --- 5. Compare Real vs Simulated Returns ---
max_days = len(log_returns)
if n_days > max_days:
    print(f"⚠️ Reducing n_days from {n_days} to available max of {max_days}")
    n_days = max_days

real_returns = pd.Series(log_returns[-n_days:].values)
mean_simulated_path = np.mean(simulated_paths, axis=0)
simulated_returns = np.log(mean_simulated_path[1:n_days+1] / mean_simulated_path[:n_days])
simulated_returns = pd.Series(simulated_returns)

# --- 6. Plot Comparison ---
plt.figure(figsize=(12, 5))
plt.plot(real_returns.values, label="Real Returns")
plt.plot(simulated_returns.values, label="Simulated Returns (Heston)", alpha=0.7)
plt.title("Real vs Simulated Log Returns")
plt.legend()
plt.grid(True)
plt.show()

# --- 7. Cumulative Difference Plot ---
diff = simulated_returns.values - real_returns.values
cumulative_diff = np.cumsum(diff)

print("Initial price S0:", S0)
print("Initial variance v0:", v0)
print("Mean daily return:", returns_mean)
print("Variance of returns:", returns_var)
print("Mean return (annualized):", returns_mean * 252)
print("Volatility (annualized):", np.sqrt(returns_var) * np.sqrt(252))

plt.figure(figsize=(12, 4))
plt.plot(cumulative_diff, label="Cumulative Difference (Sim - Real)")
plt.title("Cumulative Difference over Time")
plt.axhline(0, color='gray', linestyle='--')
plt.grid(True)
plt.legend()
plt.show()
