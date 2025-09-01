import numpy as np

u_max = 1
d_max = 4
x_th = -1.5
gamma=0.99

x_min = -2.1
x_max = 2.2
grid_min = -2.1
grid_max = 3.0

distr = 'gaussian'

log_repr = False

def x_next(x,u,d):
    return 1.01*x + 0.01 * (u + d)

def pi(x):
    if x > x_th:
        return -u_max
    else:
        return u_max
    
def phi(x):
    if x >= x_th:
        return d_max
    else:
        return -d_max

def c(x):
    return np.max((np.min((x+2,10)),-10))

def r(x):
    return np.max((np.min((-(x+1),10)),-10))

# noise distribution
def gen_noise():
    if distr == 'uniform':
        return np.random.uniform(-d_max,d_max)
    elif distr == 'gaussian':
        return np.random.normal(0,d_max)
    
# settings_class
class DynSystem():
    def __init__(self):
        self.u_max = u_max
        self.d_max = d_max
        self.x_th = x_th
        self.gamma = gamma
        self.x_min = x_min
        self.x_max = x_max
        self.distr = distr
        self.log_repr = log_repr