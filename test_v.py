import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline,make_interp_spline,make_smoothing_spline
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,x_th,u_max,d_max,gamma,distr,x_max,x_min

# V_gamma = np.load('V_gamma_x_th:-1.5_u_max:_1_d_max:_2.npy')
V_prob = np.load('V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy')
state_grid = np.load('grid.npy')

# spline V_gamma
# V_gamma_spline = make_smoothing_spline(state_grid,V_gamma)

# spline V_prob
V_prob_spline = make_smoothing_spline(state_grid,V_prob)

# Generate finer points to plot the smooth spline
x_fine = np.linspace(state_grid.min(), state_grid.max(), 10000)
# gamma_points = V_gamma_spline(x_fine)
prob_points = V_prob_spline(x_fine)

# plt.figure()
# # Plot original points and spline
# plt.plot(x_fine, gamma_points, 'b.',label='gamma')
# plt.plot(x_fine, prob_points, 'r.',label='prob')
# plt.legend()
# plt.show()
 
# # sample points to start simulations
# x_0s=np.random.uniform(1.0, 1.1,100)

# tests_for_state = 100
# steps = 10000
# for i in range(x_0s.shape[0]):
#     succ = 0
#     failures = 0
#     prob_estimate = V_prob_spline(x_0s[i])
#     # if not(0.05<=prob_estimate <=0.95):
#     #     continue
#     print(f'x_0: {x_0s[i]} -- V_prob: {min(1,prob_estimate)}')
#     for _ in range(tests_for_state):
#         steps_num = []
#         x=x_0s[i]
#         for jj in range(steps):
#             x=x_next(x,pi(x),gen_noise())
#             if c(x) < 0:
#                 failures += 1
#                 break
#             if r(x) > 0:
#                 succ+=1
#                 break
#         steps_num.append(jj)
#     print(f'Target reachings {succ/tests_for_state} % --- fails {failures/tests_for_state} % mean_number_of_steps {np.mean(steps_num)}')
    
# statistical V(x)



tests_for_state = 100
steps = 2000
V_stat = np.zeros(state_grid.shape)
for i in tqdm(range(state_grid.shape[0]-1,-1,-10)):
    succ = 0
    failures = 0
    prob_estimate = V_prob[i]
    print(f'x_0: {state_grid[i]} -- V_prob: {max(0,min(1,prob_estimate))}')
    for ll in range(tests_for_state):
        steps_num = []
        x=state_grid[i]
        if r(x) > 0 and c(x) >= 0:
                succ+=1
                continue  
        if c(x) < 0:
                failures += 1
                continue
        for jj in range(steps):
            x=x_next(x,pi(x),gen_noise())
            if c(x) < 0:
                failures += 1
                break
            if r(x) > 0 and c(x) >= 0:
                succ+=1
                break      
            if x < x_min or x > x_max:
                failures += 1
                break      
        steps_num.append(jj+1)
        if succ==0 and ll > 70:
            print('Skipped')
            break
    print(f'Target reachings {100*succ/tests_for_state} % --- fails {100*failures/tests_for_state} % mean_number_of_steps {np.mean(steps_num)}')
    V_stat[i] = succ/tests_for_state

np.save(f'V_stat_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}.npy',V_stat)


plt.figure(figsize=(10, 5))
plt.grid(True)
plt.gca().set_aspect('equal')
plt.plot(state_grid, V_stat, 'blue', label="V_stat")
plt.plot(state_grid, V_prob, 'red', label="V_prob_TD")
plt.xlabel('State Grid')
plt.ylabel('V_prob')
plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=10)
plt.legend()
plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max} distr: {distr}')
plt.savefig(f'plots/V_stat_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.png')
# plt.show()
plt.close()