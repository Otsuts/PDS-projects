# import matplotlib.pyplot as plt

# # learning_rates = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
# # small_model_accuracy = [26.77, 25.57, 21.71, 20.90, 20.78, 18.46]
# # big_model_accuracy = [32.87, 29.10, 28.22, 27.89, 27.66, 26.35]

# learning_rates = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
# small_model_accuracy = [49.29, 48.54, 48.07, 47.29, 47.02, 42.90]
# big_model_accuracy = [36.16, 39.96, 40.31, 42.25, 42.46, 41.25]

# plt.figure(figsize=(10, 6))
# plt.plot(learning_rates, small_model_accuracy, marker='o', linestyle='-', label='Small Model')
# plt.plot(learning_rates, big_model_accuracy, marker='o', linestyle='-', label='Big Model')
# plt.xscale('log')
# plt.xlabel('Learning Rate (log scale)')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy of Semantic Relatedness Models')
# plt.legend()
# plt.grid(True)
# plt.savefig('../results/sememb')


import matplotlib.pyplot as plt
import numpy as np

# example data
learning_rates = [5e-4, 1e-4, 5e-5, 1e-5]
num_samples = [1, 400, 1000]
accuracies = np.array([
    [19.85, 65.91, 69.62],
    [33.44, 69.72, 69.87],
    [29.74, 69.87, 69.88],
    [21.22, 69.71, 69.38],
])

fig, ax = plt.subplots()

# using a colormap to represent different learning rates
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(learning_rates)))

for i, lr in enumerate(learning_rates):
    ax.plot(num_samples, accuracies[i], marker='o', linestyle='-', color=colors[i], label=f'LR: {lr}')

ax.set_xlabel('Number of Samples')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy of Synthetic Model for different Learning Rates and Sample Numbers')
ax.legend()
ax.grid(True)

plt.savefig('synthetic.png')
