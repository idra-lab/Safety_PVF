import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline,make_interp_spline,make_smoothing_spline
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,x_th,u_max,d_max,gamma,distr,x_max,x_min,state_size

# V_gamma = np.load('V_gamma_x_th:-1.5_u_max:_1_d_max:_2.npy')
V_prob = np.load('V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy')
state_grid = np.load('grid_n1.npy')

noises = np.load('noises_sequence.npy')

tests_for_state = 1000
steps = 2000
V_stat = np.zeros(state_grid.shape)
for i in tqdm(range(state_grid.shape[0]-1,-1,-10)):
    succ = 0
    failures = 0
    prob_estimate = V_prob[i]
    print(f'x_0: {state_grid[i]} -- V_prob: {max(0,min(1,prob_estimate))**state_size}')
    for ll in range(tests_for_state):
        steps_num = []
        x=np.ones(state_size)*np.copy(state_grid[i])
        if (r(x) > 0).all() and (c(x) >= 0).all():
                succ+=1
                continue  
        if (c(x) < 0).any():
                failures += 1
                continue
        for jj in range(steps):
            u = pi(x)
            # d= noises[ll,jj]
            x_dyn=x_next(x,u,gen_noise())
            reached_indx = np.where((r(x_dyn) > 0) == True)[0]
            x = x_dyn
            x[reached_indx] = -1.5
            # print(f'x: {x}  u: {u}, d: {d}')
            if (c(x) < 0).any():
                failures += 1
                break
            if (r(x) > 0).all() and (c(x) >= 0).all():
                succ+=1
                break      
            if (x < x_min).any() or (x > x_max).any():
                failures += 1
                break      
        steps_num.append(jj+1)
        if succ==0 and ll > 70:
            print('Skipped')
            break
    print(f'Target reachings {100*succ/tests_for_state} % --- fails {100*failures/tests_for_state} % mean_number_of_steps {np.mean(steps_num)}')
    V_stat[i] = succ/tests_for_state

np.save(f'V_stat_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_n{state_size}.npy',V_stat)


plt.figure(figsize=(10, 5))
plt.grid(True)
plt.gca().set_aspect('equal')
plt.plot(state_grid, V_stat, 'blue', label="V_stat")
plt.plot(state_grid, V_prob**state_size, 'red', label="V_prob_TD^state_size")
plt.plot(state_grid, V_prob, 'green', label="V_prob_TD")
plt.xlabel('State Grid')
plt.ylabel('V_prob')
plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=10)
plt.legend()
plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max} distr: {distr}')
plt.savefig(f'plots/V_stat_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}_n{state_size}.png')
plt.show()
plt.close()

np.random.seed(0)


V_stat = np.zeros(state_grid.shape)
for i in tqdm(range(state_grid.shape[0]-1,-1,-10)):
    succ = 0
    failures = 0
    prob_estimate = V_prob[i]
    print(f'x_0: {state_grid[i]} -- V_prob: {max(0,min(1,prob_estimate))**state_size}')
    for ll in range(tests_for_state):
        partial_succ=0
        partial_fails = 0
        for kk in range(state_size):
            steps_num = []
            x=np.copy([state_grid[i]])
            # x = np.ones(state_size) * np.copy(state_grid[i])
            if (r(x) > 0).all() and (c(x) >= 0).all():
                    partial_succ+=1
                    continue  
            if (c(x) < 0).any():
                    partial_fails += 1
                    continue
            for jj in range(steps):
                u = pi(x)
                # d = noises[ll,jj,kk]
                x=x_next(x,u,gen_noise[kk])
                # print(f'x: {x}  u: {u}, d: {d}')

                if (c(x) < 0).any():
                    partial_fails += 1
                    break
                if (r(x) > 0).all() and (c(x) >= 0).all():
                    partial_succ+=1
                    break      
                if (x < x_min).any() or (x > x_max).any():
                    partial_fails += 1
                    break      
            steps_num.append(jj+1)
            if partial_succ==0 and ll > 70:
                # print('Skipped')
                break
            if partial_fails>0:
                failures += 1
                break
        if partial_fails == 0 and partial_succ > 0:
            succ += 1 
            
                
        print(f'Target reachings {100*succ/tests_for_state} % --- fails {100*failures/tests_for_state} % mean_number_of_steps {np.mean(steps_num)}')
        V_stat[i] = succ/tests_for_state

np.save(f'V_stat_multi_single_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_n{state_size}.npy',V_stat)


plt.figure(figsize=(10, 5))
plt.grid(True)
plt.gca().set_aspect('equal')
plt.plot(state_grid, V_stat, 'blue', label="V_stat")
plt.plot(state_grid, V_prob**state_size, 'red', label="V_prob_TD^state_size")
plt.plot(state_grid, V_prob, 'green', label="V_prob_TD")
plt.xlabel('State Grid')
plt.ylabel('V_prob')
plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=10)
plt.legend()
plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max} distr: {distr}')
plt.savefig(f'plots/V_stat_multi_single_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}_n{state_size}.png')
plt.show()
plt.close()