from system_dyn import state_size,x_next,pi,gen_noise,c,r,x_min,x_max,d_max
import numpy as np

noises = np.load('noises_sequence.npy')

V_prob = np.load('V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy')
state_grid = np.load('grid_n1.npy')

state_indx = 600
succ = 0
failures = 0
x_0 = state_grid[state_indx]
print(f'x_0: {state_grid[state_indx]} -- V_prob: {max(0,min(1,V_prob[state_indx]))**state_size}')

x_multi=np.ones(state_size)*np.copy(state_grid[state_indx])
x_single = np.copy(x_multi)
# for ll in range(2000):
#     u_multi = pi(x_multi)
#     d_multi = noises[ll:ll+state_size]
#     x_multi=x_next(x_multi,u_multi,d_multi)

#     u_single = np.zeros(state_size)
#     d_single = np.zeros(state_size)
#     for jj in range(state_size):
#         u_single[jj] = pi([x_single[jj]])
#         d_single[jj] = noises[ll:ll+state_size][jj]
#         x_single[jj] = x_next(x_single[jj],u_single[jj],d_single[jj])
    

#     if (c(x_multi) < 0).any():
#         print('Fail constraint multi')
#     if (r(x_multi) > 0).all() and (c(x_multi) >= 0).all():
#         print('Success multi')
#     if (x_multi < x_min).any() or (x_multi > x_max).any():
#         print('Fail boundaries multi')

#     if (c(x_single) < 0).any():
#         print('Fail constraint single')
#     if (r(x_single) > 0).all() and (c(x_single) >= 0).all():
#         print('Success single')
#     if (x_single < x_min).any() or (x_single > x_max).any():
#         print('Fail boundaries single')
    
    
#     print(f'x_multi: {x_multi} u_multi: {u_multi} d_multi: {d_multi} --- x_single: {x_single} u_single: {u_single}  d_single{d_single}')



k=100

np.random.seed(0)
for kk in range(k):
    # print(f'Sim {k}')
    x=np.ones(state_size)*np.copy(state_grid[state_indx])

    for ll in range(2000):
        # for jj in range(0,2000,state_size):
        u=pi(x)
        index = kk * 1000 + ll * state_size
        noise = noises[index:index+state_size]
        x=x_next(x,u,noise)
        if (c(x) < 0).any():
            failures += 1
            break
        if (r(x) > 0).all() and (c(x) >= 0).all():
            succ+=1
            break      
        if (x < x_min).any() or (x > x_max).any():
            failures += 1
            break     
    print(f'x_f: {x}   , u:{u} , d:{noise}')

print(f'Succ multi {succ}')
succ=0

np.random.seed(0)
for kk in range(k):
    # print(f'Sim {k}')
    
    for ll in range(state_size):
        partial_fails = 0
        partial_succ = 0
        x=np.copy(state_grid[state_indx])
        for __ in range(2000):
            # for jj in range(0,2000,state_size):
            u=pi(np.array([x]))
            # index = kk * 1000 + jj
            # noise = noises[index][0]
            noise = np.random.normal(0,d_max)
            x=x_next(x,u,noise)
            if (c(x) < 0).any():
                    partial_fails += 1
                    break
            if (r(x) > 0).all() and (c(x) >= 0).all():
                partial_succ+=1
                break      
            if (x < x_min).any() or (x > x_max).any():
                partial_fails += 1
                break
                
    print(f'x: {x}   , u:{u} , d:{noise}')
    if partial_fails == 0 and partial_succ>0:
        succ += 1 

# print(f'Succ single {succ}')
