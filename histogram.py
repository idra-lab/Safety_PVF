import numpy as np
from system_dyn import x_min,x_max
import matplotlib.pyplot as plt
from tqdm import tqdm

state_size = 9
bins = 5

data_pairs = np.load(f'data/data_pairs_state_size{state_size}.npy')
bins = np.linspace(x_min, x_max, bins)
hist = np.zeros(bins.shape[0] - 1)

for i in tqdm(range(data_pairs.shape[0])):
    for j in range(bins.shape[0] - 1):
        if (bins[j] <= data_pairs[i,:state_size]).all() and (bins[j+1] >= data_pairs[i,:state_size]).all():
            # print(f'+{j}')
            hist[j] +=1

print(f'bins {bins}, hist{hist}')

plt.figure()
plt.grid(True)

plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge', color='green', edgecolor='black')

# Adding labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram all components in same bin')

# Show the plot
plt.show()
plt.close()


failures = np.where(data_pairs[:,-1] == True)[0]

pass

