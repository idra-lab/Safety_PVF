from system_dyn import d_max, state_size
import numpy as np

noise = np.zeros((100,2000,state_size))

for i in range(100):
    for j in range(2000):
        noise[i,j] = np.random.normal(0,d_max,state_size)
np.save('noises_sequence.npy', np.array(noise))
