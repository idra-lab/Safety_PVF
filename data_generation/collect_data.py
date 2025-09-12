import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_min,x_max,state_size

episodes = 500000
length = 200

data = np.zeros((episodes,length,state_size+2))
succ=0
fails=0

print(f'State size : {state_size}')

for i in tqdm(range(episodes)):
    x = np.random.uniform(x_min,x_max,state_size)
    data[i,0] = np.hstack((x,[((r(x) > 0).all() and (c(x) >= 0).all()), (c(x) < 0).any()]))

    reached_length = 0
    reached_old = 0
    if not(data[i,0][-2]):
        for j in range(length-1):
            x_dyn = x_next(x,pi(x),gen_noise())
            reached_indx = np.where((r(x_dyn) > 0) & (c(x_dyn) >= 0))[0] 
            reached_length = reached_indx.shape[0]
            x = x_dyn
            if reached_length > 0:
                if reached_length > reached_old:              
                    x_reached = np.copy(x)
                    x[reached_indx] = x_reached[reached_indx]
                    reached_old = reached_length
                else:
                    x[reached_indx] = data[i,j,reached_indx]
            
            # print(f'Reached {reached_indx}')
            data[i,j+1] = np.hstack((x, [((r(x) > 0).all() and (c(x) >= 0).all()), (c(x) < 0).any()]))
            if ((r(x) > 0).all() and (c(x) >= 0).all()):
                succ += 1
                # print('SUCC')
                break
            if (c(x) < 0).any():
                # print(f'x < -2   {x}')
                fails += 1
                break
            if (x > x_max).any() or (x < x_min).any():
                data[i,j+1] = np.hstack((x, [False,True]))
                fails += 1
                break


print(f'Successes {succ} , failures {fails}')
np.save(f'training_data_interrupted_n{state_size}.npy',data)



