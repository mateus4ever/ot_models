# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Number of iterations in the simulation
n = 10000000

# Generate random points within a square of side length 2 centered at the origin
x = np.random.uniform(low=-1, high=1, size=n)
y = np.random.uniform(low=-1, high=1, size=n)

# Identify points inside the unit circle using Pythagoras's theorem
inside = np.sqrt(x**2 + y**2) <= 1

# Estimate Pi using the ratio of points inside the circle to total points
pi_estimate = 4.0 * sum(inside) / n

# Calculate percentage error compared to the actual value of Pi
error = abs((pi_estimate - np.pi) / np.pi) * 100

# Display results
print(f"After {n} simulations, our estimate of Pi is {pi_estimate} with an error of {round(error, 2)}%")

# Points outside the circle
outside = np.invert(inside)

# Plot the Monte Carlo simulation
plt.figure(figsize=(6,6))
plt.plot(x[inside], y[inside], 'b.', markersize=1, label='Inside Circle')
plt.plot(x[outside], y[outside], 'r.', markersize=1, label='Outside Circle')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Estimating Pi using Monte Carlo Simulation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.axis('square')
plt.show()
