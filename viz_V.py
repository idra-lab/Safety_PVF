import numpy as np
import matplotlib.pyplot as plt
from system_dyn import x_th, u_max, d_max
# d_max=2
# V_gamma = np.load("V_gamma_x_th:-1.5_u_max:_1_d_max:_0..npy")
V_prob = np.load("V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy")
state_grid = np.load("grid.npy")

# state_grid = np.linspace(-2.2,2.2,V_prob.shape[0])


plt.figure()
plt.grid(True)
# plt.gca().set_aspect("equal")
plt.plot(state_grid, np.ones(V_prob.shape) - V_prob, color="blue", marker=".", markersize=0.5,linewidth=0.5 ,label="V_prob")
plt.xlabel("x")
plt.yscale('log')
plt.ylabel("V_prob")
# plt.xticks(np.arange(np.min(state_grid), np.max(state_grid) + 0.2, 0.2), fontsize=10)
plt.legend()
plt.title(f"V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max}")
plt.savefig(f"plots/V_X.png")
plt.show()
plt.close()


