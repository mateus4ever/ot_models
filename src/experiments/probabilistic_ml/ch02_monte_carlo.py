import random
import matplotlib.pyplot as plt

# Number of iterations in the simulation
number_of_iterations = [10, 100, 1000, 10000]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

for i, num_iterations in enumerate(number_of_iterations):
    # Lists to store results of all iterations
    stay_results = []
    switch_results = []

    # Simulation loop
    for _ in range(num_iterations):
        doors = ['door 1', 'door 2', 'door 3']

        # Randomly select a door to place the car
        car_door = random.choice(doors)

        # You select a door at random
        your_door = random.choice(doors)

        # Monty selects a door that does not have the car and is not your choice
        monty_door = next(door for door in doors if door != car_door and door != your_door)

        # The door that remains after Monty opens one
        switch_door = next(door for door in doors if door != monty_door and door != your_door)

        # Append results
        stay_results.append(your_door == car_door)
        switch_results.append(switch_door == car_door)

    # Compute probabilities
    probability_staying = sum(stay_results) / num_iterations
    probability_switching = sum(switch_results) / num_iterations

    # Select subplot
    ax = axs[i // 2, i % 2]

    # Plot the probabilities as a bar graph
    ax.bar(['Stay', 'Switch'], [probability_staying, probability_switching], color=['blue', 'green'], alpha=0.7)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Probability of Winning')
    ax.set_title(f'After {num_iterations} Simulations')
    ax.set_ylim([0, 1])

    # Add probability values on the bars
    ax.text(0, probability_staying + 0.05, f'{probability_staying:.2f}', ha='center', va='bottom', fontsize=10)
    ax.text(1, probability_switching + 0.05, f'{probability_switching:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
