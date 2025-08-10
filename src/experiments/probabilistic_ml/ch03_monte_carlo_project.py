import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random as npr

# Number of trials/simulations
n = 10000

# Define project parameters
fte = npr.uniform(low=1, high=5, size=n)  # Full-time equivalent persons on the team
effort = npr.uniform(low=240, high=480, size=n)  # Effort in person-days
price = npr.uniform(low=100, high=200, size=n)  # Price based on market research
units = npr.normal(loc=1000, scale=500, size=n)  # Units sold
discount_rate = npr.uniform(low=0.06, high=0.10, size=n)  # Discount rate

daily_rate = 400  # Cost per day
technology_charges = 500  # Technology-related costs
overhead_charges = 200  # Overhead costs
tax_rate = 0.15  # Tax rate

# Calculate project costs and revenues
labor_costs = effort * daily_rate
technology_costs = fte * technology_charges
overhead_costs = fte * overhead_charges
revenues = price * units

duration = effort / fte  # Duration in days

# Compute free cash flow
free_cash_flow = (revenues - labor_costs - technology_costs - overhead_costs) * (1 - tax_rate)

# Compute Net Present Value (NPV)
npv = free_cash_flow / (1 + discount_rate)

# Convert numpy arrays to pandas DataFrames for easier analysis
NPV = pd.DataFrame(npv, columns=['NPV'])
Duration = pd.DataFrame(duration, columns=['Days'])

# Plot histogram of NPV distribution
plt.figure(figsize=(10, 5))
plt.hist(NPV['NPV'], bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Project NPV')
plt.xlabel('Project NPV')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Display descriptive statistics of NPV
print(NPV.describe().round(2))

# Compute and display probability of positive NPV
success_probability = sum(NPV['NPV'] > 0) / n * 100
print(f'There is a {round(success_probability, 2)}% probability that the project will have a positive NPV.')

# Plot histogram of project duration distribution
plt.figure(figsize=(10, 5))
plt.hist(Duration['Days'], bins=50, color='green', alpha=0.7, edgecolor='black')
plt.title('Distribution of Project Duration')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Display descriptive statistics of project duration
print(Duration.describe().round(2))
