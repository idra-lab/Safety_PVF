import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_max,x_min,grid_max,grid_min,state_size

width = 10
height = 5

grid_state_size = 1000
state_grid = np.linspace(grid_min,grid_max,grid_state_size+1)

# TD learning to learn probabilistic V
V_prob = np.zeros((grid_state_size + 1,)*state_size)
lr_TD = 0.1

V_prob_ground_truth = np.load("V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy")
state_grid_ground_truth = np.load("grid.npy")

def TD_step(x):
    state_size = x.shape[0]
    index_x = np.zeros(state_size)
    x_n = x
    for i in range(state_size):
        index_x[i] = np.argmin(np.abs(state_grid - x_n[i]))

    if not((c(x) < 0).any()): 
        for _ in range(1):
            x_n = x_next(x_n,pi(x),gen_noise())
        index_x_n = np.zeros(state_size)
        
        for i in range(state_size):
            index_x_n[i] = np.argmin(np.abs(state_grid - x_n[i]))
        
        if (r(x_n) > 0).all() and (c(x_n) > 0).all():   # when x <2 in target but not safe
            V_target = 1
        elif (c(x_n) < 0).any():
            V_target = 0
        else:
            V_target = V_prob[index_x_n.astype(int)]
        if (x > x_max).any() or (x < x_min).any():
            V_target = 0
    else:
        V_target = 0
    V_prob[index_x.astype(int)] += lr_TD * (V_target - V_prob[index_x.astype(int)])
    return x_n
     

#x_th = -1.78
# generate episodes and do TD learning
n_episodes = 500000
episode_length = 2000

for i in tqdm(range(n_episodes), desc='TD learning'):
    x =  np.random.uniform(x_min,x_max,state_size)
    # x = np.random.uniform(1.2,2)
    # print(f'x_0: {x}')
    #x = 1.1
    for _ in range(episode_length):
        x = TD_step(x)
        if (r(x) > 0).all() and (c(x) > 0).all():   # when x <2 in target but not safe
            break
        if (c(x) < 0).any():
            break

    # print(f'x_f {x}')
    if i%1000==0:
        plt.figure(figsize=(width, height))
        plt.grid(True)
        plt.gca().set_aspect('equal')
        plt.plot(state_grid, V_prob, 'blue', label="V_prob")
        plt.plot(state_grid_ground_truth, np.power(V_prob_ground_truth,state_size), 'red', label="V_prob_pow")
        plt.xlabel('State Grid')
        plt.ylabel('V_prob')
        plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=10)
        plt.legend()
        plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max} noise: {distr}')
        plt.savefig(f'plots/V_prob_n_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.png')
        # plt.show()
        plt.close()

    if i%10000==0:
        np.save(f'V_prob_n{state_size}_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.npy',V_prob)
        np.save(f'grid_n{state_size}.npy',state_grid)

np.save(f'V_prob_n_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.npy',V_prob)
np.save('grid_n.npy',state_grid)
