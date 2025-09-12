import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

n=2
d_max = 4
u_max = 1
x_th = -1.5
x_min = -2.1
x_max = 2.1

V_prob = np.load('V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy')
state_grid = np.load('grid_n1.npy')

def pi(x):
    u = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] > x_th:
            u[i] = -u_max
        else:
            u[i] = u_max
    return u

def x_next(x,u):
    return 1.01*x + 0.01*u + 0.01 * np.random.normal(0,d_max,n)

def simulation(x0):
    x = x0
    for j in range(2000):
        for i in range(n):
            succ_flag = True
            x[i] = 1.01*x[i] + 0.01*(pi(x)[i] + np.random.normal(0,d_max,n)[i]) 
            if (-2 <= x[i]) and (x[i] < -1):
                succ_flag = True
            else:
                succ_flag = False
            if (-2 > x[i]) or (x_max<x[i]):
                # print(j)
                return False
        if succ_flag:
            return True
        
def simulation2(x0):
    x = x0
    for j in range(2000):
        x = 1.01*x + 0.01*(pi(x) + np.random.normal(0,d_max,n)) 
        if (-2 <= x).all() and (x < -1).all():
           return True
        if (-2 > x).any() or (x_max<x).any():
            print(f'Failed x: {x}')
            return False
    

state_indx = 600
x_0 = state_grid[state_indx] * np.ones(n)
print(f'x_0 {x_0} ------ V_prob {V_prob[state_indx]} ------ V_prob_joint {V_prob[state_indx]**n}')

n_sim = 100

succ = 0
V_stat = np.zeros(state_grid.shape[0])
for kk in tqdm(range(state_grid.shape[0])):
    x_0 = state_grid[kk] * np.ones(n)
    print(f'x_0 {x_0} ------ V_prob {V_prob[kk]} ------ V_prob_joint {V_prob[kk]**n}')
    succ=0
    for _ in range(n_sim):
        x_0 = state_grid[kk] * np.ones(n)
        if(simulation2(x_0)):
            succ += 1
    V_stat[kk] = succ/n_sim
    print(f'Success percentage {(succ/n_sim)*100} %')

plt.figure(figsize=(10, 5))
plt.grid(True)
plt.gca().set_aspect('equal')
plt.plot(state_grid, V_stat, 'blue', label="V_stat")
plt.plot(state_grid, V_prob**n, 'red', label="V_prob_TD^state_size")
plt.plot(state_grid, V_prob, 'green', label="V_prob_TD")
plt.xlabel('State Grid')
plt.ylabel('V_prob')
plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=10)
plt.legend()
plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max}')
plt.show()
plt.close()