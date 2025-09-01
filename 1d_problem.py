import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_max,x_min,grid_max,grid_min

width = 10
height = 5


# def x_next(x,u,d):
#     return 1.05*x + 0.05 * (u + d)


# def pi(x):
#     if x > x_th:
#         return -u_max
#     else:
#         return u_max
    
# def phi(x):
#     if x >= x_th:
#         return d_max
#     else:
#         return -d_max

# def c(x):
#     return np.max((np.min((x+2,10)),-10))

# def r(x):
#     return np.max((np.min((-(x+1),10)),-10))

def get_value_gamma(x0):   # new policy of them
    i=1
    x=x0
    max_iter = 1000
    r_gamma = np.zeros(max_iter)
    c_gamma = np.ones(max_iter)*100
    g_gamma = np.zeros(max_iter)
    while True:
        x_t=x_next(x,pi(x),phi(x)) 
        # print(f'x_t: {x_t}')
        r_gamma[i-1] = (gamma**i)*r(x_t)
        c_gamma[i-1] = (gamma**i)*c(x_t)
        g_gamma[i-1] = np.min((r_gamma[i-1],np.min(c_gamma[:i+1])))
        # print(g_gamma[i-1])

        x=x_t
        # if r_gamma[i-1] > 0 and c_gamma[i-1] > 0:
        #     return r_gamma[i-1]
        # elif c_gamma[i-1] < 0:
        #     return c_gamma[i-1]
        if i>=max_iter:
            print('Max iter')
            print(f'x_0 = {x0} x_t: {x_t}')
            return np.max(g_gamma)
        i+=1

def get_value_old(x0):   # old policy of them
    i=1
    x=x0
    max_iter = 1000
    r_ = np.zeros(max_iter)
    c_ = np.zeros(max_iter)
    g_ = np.zeros(max_iter)
    while True:
        x_t=x_next(x,pi(x),phi(x)) 
        r_[i-1] = r(x_t)
        c_[i-1] = c(x_t)
        g_[i-1] = np.min((r_[i-1],np.min(c_[:i])))
        x=x_t
        # if r_gamma[i-1] > 0 and c_gamma[i-1] > 0:
        #     return r_gamma[i-1]
        # elif c_gamma[i-1] < 0:
        #     return c_gamma[i-1]
        if i>=max_iter:
            print('Max iter')
            return np.max(g_)
        i+=1



grid_state_size = 1000
state_grid = np.linspace(grid_min,grid_max,grid_state_size+1)

V_table = np.zeros((grid_state_size+1))
V_table_old = np.zeros((grid_state_size+1))
# TD learning to learn probabilistic V
V_prob = np.ones((grid_state_size + 1)) *0.5
lr_TD = 0.1

def TD_step(x):
    index_x = np.argmin(np.abs(state_grid - x))
    x_n = x
    # if r(x) <= 0 and c(x_n) > 0:
    if not(c(x) < 0): 
        for _ in range(1):
            x_n = x_next(x_n,pi(x),gen_noise())
        index_x_n = np.argmin(np.abs(state_grid - x_n))
        x_n = state_grid[index_x_n]
        # else:
        #     x_n = x
        if r(x_n) > 0 and c(x_n) > 0:   # when x <2 in target but not safe
            V_target = 1
        elif c(x_n) < 0:
            V_target = 0
        else:
            V_target = V_prob[index_x_n]
        if x > x_max or x < x_min:
            V_target = 0
    else:
        V_target = 0
    V_prob[index_x] += lr_TD * (V_target - V_prob[index_x])
    return x_n
     

#x_th = -1.78
# generate episodes and do TD learning
n_episodes = 500000
episode_length = 2000

for i in tqdm(range(n_episodes), desc='TD learning'):
    x =  np.random.choice(state_grid)
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
        plt.savefig(f'plots/V_prob_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.png')
        # plt.show()
        plt.close()

    if i%10000==0:
        np.save(f'V_prob_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.npy',V_prob)
        np.save('grid.npy',state_grid)

    # print(f'x_f: {x}')

    # x =  np.random.uniform(1.4,2)
    # # print(f'x_0: {x}')
    # #x = 1.1
    # for _ in range(episode_length):
    #     x = TD_step(x)
    #     if r(x) > 0 and c(x) > 0:   # when x <2 in target but not safe
    #         break
    #     if c(x) < 0:
    #         break
# generate and plot the V_table for many x_th for the policies

# for j in range(grid_state_size + 1):
#     x_th = state_grid[j]
x_th = -1.5
for i in range(grid_state_size + 1):
    V_table[i] = get_value_gamma(state_grid[i])
    V_table_old[i] = get_value_old(state_grid[i])


plt.figure(figsize=(width, height))
# First subplot (left) for V_table - plot all data
plt.plot(state_grid, V_table, 'r', label="V_table")
plt.grid(True)
plt.gca().set_aspect('equal')
plt.xlabel('State Grid')
plt.ylabel('V_table')
plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=8)

plt.legend()
plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max}')
plt.savefig(f'plots/V_gamma_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.png')
plt.close()

plt.figure(figsize=(width, height))
# Second subplot (right) for V_table_old - plot all data
plt.plot(state_grid, V_table_old, 'b', label="V_table_old")
plt.gca().set_aspect('equal')
plt.grid(True)
plt.xlabel('State Grid')
plt.ylabel('V_table_old')
plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=8)
plt.legend()
plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max}')
plt.savefig(f'plots/V_old_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.png')

plt.close()

np.save(f'V_prob_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.npy',V_prob)
np.save(f'V_gamma_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.npy',V_table)
np.save('grid.npy',state_grid)
