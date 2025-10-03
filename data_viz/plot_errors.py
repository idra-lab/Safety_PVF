import matplotlib.pyplot as plt
import numpy as np


state_size = 11 

data_log_extrema = np.load(f'errors/error_extrema_size_{state_size}_log_learning_True.npy')
data_normal_extrema = np.load(f'errors/error_extrema_size_{state_size}_log_learning_False.npy')
data_log_general = np.load(f'errors/error_normal_size_{state_size}_log_learning_True.npy')
data_normal_general = np.load(f'errors/error_normal_size_{state_size}_log_learning_False.npy')

# Create a figure with 1 row and 2 columns of subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
# First subplot
axes[0].plot(data_log_extrema, label='log_learning', color='blue')
axes[0].set_title('Extrema errors log_learning')
axes[0].legend()
axes[0].grid(True)

# Second subplot
axes[1].plot(data_normal_extrema, label='normal_learning', color='red')
axes[1].set_title('Extrema errors normal_learning')
axes[1].legend()
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
plt.close()

# Create a figure with 1 row and 2 columns of subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
# First subplot
axes[0].plot(data_log_general, label='log_learning', color='blue')
axes[0].set_title('General error log_learning')
axes[0].legend()
axes[0].grid(True)

# Second subplot
axes[1].plot(data_normal_general, label='normal_learning', color='red')
axes[1].set_title('General error normal_learning')
axes[1].legend()
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
plt.close()

