import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_min,x_max,state_size,failure_max,target_max

episodes = 300_000
length = 200

data = np.zeros((episodes,length,state_size+2))
succ=0
fails=0

print(f'State size : {state_size}')

def r(x):
    # r = np.zeros(x.shape[0])
    r = (-1.5<=x)*(-1>= x)
    return r

def c(x):
    c = (x <= x_min)|(x >= x_max)
    return c

failure_max = - 2
target_max = -1
target_min = -1.5

def x_next(x,u,d):
    x=np.array(x)
    x_next = 1.01*x + 0.01 * (u + d)
    reached_states = np.where((failure_max <= x) & (x >= target_min) & (x <= target_max))[0]
    x_next[reached_states] = x[reached_states]
    return x_next

for i in tqdm(range(episodes)):
    # x = np.random.uniform(x_min,x_max,state_size)
    x = np.array([np.random.uniform(x_min,x_max)]*state_size)

    data[i,0] = np.hstack((x,[(r(x)*(~c(x))).all(), (c(x)).any()]))
    if ((r(x)*(~c(x))).all()):
        data[i,1] = np.hstack((x,[True, False]))
        succ+=1
        continue
    if not(data[i,0][-2]):
        for j in range(length-1):
            x = x_next(x,pi(x),gen_noise())
            
            data[i,j+1] = np.hstack((x,[(r(x)*(~c(x))).all(), (c(x)).any()]))
            if ((r(x)*(~c(x))).all()):
                succ += 1
                # print('SUCC')
                break
            if (c(x)).any():
                # print(f'x < -2.1   {x}')
                fails += 1
                break

# counter_succ=0
# data_tmp = np.zeros((3*fails,length,state_size+2))

# while counter_succ < 3*fails:
#     # print(f'Counter {counter}')
#     if succ % 1000 == 0:
#         print(f'Succ {succ}, fails {fails}, ratio {succ/fails}')
#     x = np.random.uniform(failure_max,target_max,state_size)

#     data_tmp[counter_succ,0] = np.hstack((x,[(r(x)*c(x)).all(), (c(x) < 0).any()]))
#     # if ((r(x) > 0).all() and (c(x) >= 0).all()):
#     #     data_tmp[counter_succ,1] = np.hstack((x,[True, False]))
#     #     succ+=1
#     #     counter_succ += 1

#     #     continue
    
#     reached_length = 0
#     reached_old = 0
    
#     for j in range(1):
#         x_dyn = x_next(x,pi(x),gen_noise())
#         reached_indx = np.where((r(x_dyn) > 0) & (c(x_dyn) >= 0))[0] 
#         reached_length = reached_indx.shape[0]
#         x = x_dyn
#         if reached_length > 0:
#             if reached_length > reached_old:              
#                 x_reached = np.copy(x)
#                 x[reached_indx] = x_reached[reached_indx]
#                 reached_old = reached_length
#             else:
#                 x[reached_indx] = data_tmp[counter_succ,j,reached_indx]
        
#         # print(f'Reached {reached_indx}')
#         if ((r(x) > 0).all() and (c(x) >= 0).all()):
#             succ += 1
#             data_tmp[counter_succ,j+1] = np.hstack((x, [(r(x)*c(x)).all(), (c(x) < 0).any()]))
#             counter_succ += 1
#             # print('SUCC')
#             break
#         if (c(x) < 0).any():
#             # print(f'x < -2   {x} fails')
#             # fails += 1
#             break
#         if (x > x_max).any() or (x < x_min).any():
#             # data_tmp[0,j+1] = np.hstack((x, [False,True]))
#             # fails += 1
#             break
        
# data = np.vstack((data,data_tmp))

# failures_sx = 20000
# data_sx = np.zeros((failures_sx,length,state_size+2))
# for j in range(failures_sx):
#     x = np.random.uniform(x_min,failure_max,state_size)
#     data_sx[j,0] = np.hstack((x,[(r(x)*c(x)).all(), (c(x) < 0).any()]))
#     data_sx[j,1] = np.hstack((x,[(r(x)*c(x)).all(), (c(x) < 0).any()]))

# data = np.vstack((data,data_sx))

print(f'Successes {succ} , failures {fails}, shape {data.shape}')
np.save(f'data/training_data_interrupted_different_n{state_size}.npy',data)



