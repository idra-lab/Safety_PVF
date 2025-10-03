import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_max,x_min,grid_max,grid_min

width = 10
height = 5

def r(x):
    # r = np.zeros(x.shape[0])
    r = (-1.5<=x)*(-1>= x)
    return r

def c(x):
    c = (x <= x_min)|(x >= x_max)
    return c


grid_state_size = 1000
state_grid = np.linspace(grid_min,grid_max,grid_state_size+1)

V_table = np.zeros((grid_state_size+1))
V_table_old = np.zeros((grid_state_size+1))
# TD learning to learn probabilistic V
V_prob = np.zeros((grid_state_size + 1))
lr_TD = 0.1

def TD_step(x):
    index_x = np.argmin(np.abs(state_grid - x))
    x_n = x
    if r(x) * (~c(x)):
        V_target = 1
    else:
        if not(c(x)): 
            for _ in range(1):
                x_n = x_next(x_n,pi(np.array([x])),gen_noise())
            index_x_n = np.argmin(np.abs(state_grid - x_n))
            x_n = np.array([state_grid[index_x_n]])
            if r(x_n)*(~c(x_n)):   # when x <2 in target but not safe
                V_target = 1
            elif c(x_n):
                V_target = 0
            else:
                V_target = V_prob[index_x_n]
        else:
            V_target = 0
    V_prob[index_x] += lr_TD * (V_target - V_prob[index_x])
    return x_n
     

#x_th = -1.78
# generate episodes and do TD learning
n_episodes = 500000
episode_length = 200

for i in tqdm(range(n_episodes), desc='TD learning'):
    x =  np.array([np.random.choice(state_grid)])
    # x = np.random.uniform(1.2,2)
    # print(f'x_0: {x}')
    #x = 1.1
    for _ in range(episode_length):
        x = TD_step(x)
        # if r(x) > 0 and c(x) > 0:   # when x <2 in target but not safe
        #     break
        # if c(x) < 0:
        #     break

    # print(f'x_f {x}')
    if i%1000==0:
        plt.figure(figsize=(width, height))
        plt.grid(True)
        plt.gca().set_aspect('equal')
        plt.plot(state_grid, V_prob, 'black', label="V_prob")
        plt.xlabel('State Grid')
        plt.ylabel('V_prob')
        plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=10)
        plt.legend()
        plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max} noise: {distr}')
        plt.savefig(f'plots/V_prob_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}_different_problem.png')
        # plt.show()
        plt.close()

    if i%10000==0:
        np.save(f'V_prob_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}_different_problem.npy',V_prob)
        np.save('grid.npy',state_grid)

np.save('grid.npy',state_grid)
