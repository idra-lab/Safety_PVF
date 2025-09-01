import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_min,x_max

episodes = 100000
length = 2000

data = np.zeros((episodes,length,3))
succ=0
fails=0
for i in tqdm(range(episodes)):
    x = np.random.uniform(x_min,x_max)
    data[i,0] = [x,((r(x) > 0) and (c(x) >= 0)), c(x) < 0]
    for j in range(length-1):
        x = x_next(x,pi(x),gen_noise())
        data[i,j+1] = [x, ((r(x) > 0) and (c(x) >= 0)), c(x) < 0]
        # if ((r(x) > 0) and (c(x) >= 0)):
        #     succ += 1
        #     break
        # if c(x) < 0:
        #     fails += 1
        #     break
        if x > x_max or x < x_min:
            data[i,j+1] = [x, False,True]
            break


print(f'Successes {succ} , failures {fails}')
np.save('training_data_not_interrupted.npy',data)



