import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from net_learner import NeuralNetwork, RegressionNN
from tqdm import tqdm
from system_dyn import u_max,d_max,x_th,distr,System_conf
import os
import torch.nn as nn
import sys

# load data
if len(sys.argv) < 2:
    print('Use state size specified in the script')
    state_size = 10
else:
    state_size = int(sys.argv[1])
    print(f'Use state size {state_size}')
data = np.load(f"data/training_data_interrupted_different_n{state_size}.npy",mmap_mode='r')

print("Dataset loaded:", data.shape)

# data = data[:20000]
V_prob = np.load("data/V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy")
# V_prob = np.load("data/V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian_different_problem.npy")
state_grid = np.load("data/grid.npy")

dyn_settings = System_conf()
dyn_settings.state_size = state_size
dyn_settings.V_prob = V_prob
dyn_settings.state_grid = state_grid



def generate_pairs(data):
    """
    Execute on n_episodes x horizon x (states + reached + violated) data
    """
    pairs = []
    for i in range(0,data.shape[0],2):
        for j in range(data.shape[1] - 2):
            if (data[i, j, :] != np.zeros(data.shape[2])).any() and (
                data[i, j + 1, :] != np.zeros(data.shape[2])
            ).any():
                pairs.append(np.hstack((data[i, j, :], data[i, j + 1, :])))
    return np.array(pairs)

# compute ground truth for multi size case. Since each state is indipendent, the V for the multi state case is the product of V(x_i)
def V_prob_multi_state(x,V_prob,grid):
    V = 1
    for i in range(x.shape[0]):
        index_x = np.argmin(np.abs(grid - x[i]))
        V *= V_prob[index_x]
    return V

# success and failures array
reached, violated = data[:, :, -2], data[:, :, -1]

# remove a dimension in the data, to obtain only states
# norm_data_state = norm_data_state.reshape(-1, norm_data_state.shape[2])

if not(os.path.exists(f'data/data_pairs_state_size{state_size}.npy')):
    data_pairs_chunk = []
    # generate data in pairs
    step = int(data.shape[0]/1000)
    for jj in tqdm(range(0,data.shape[0],step)):
        data_chunk = data[jj:jj+step]
        data_pairs_chunk.append(generate_pairs(data_chunk))
    data_pairs = np.vstack(data_pairs_chunk)

    np.save(f'data/data_pairs_state_size{state_size}.npy',data_pairs)
else: data_pairs = np.load(f'data/data_pairs_state_size{state_size}.npy')

# plot state distribution
# Loop through each column and plot its histogram

# fig, axes = plt.subplots(1, sharey=True, tight_layout=True)
# for i in range(1):
# # We can set the number of bins with the *bins* keyword argument.
#     axes.hist(data_pairs[:,0], bins=50, edgecolor='black' , color='green', density=False)

# # Adjust layout
# plt.tight_layout()
# plt.show()


state_size = int(data_pairs[0].shape[0]/2 - 2)
print(f'State size: {state_size}')

mean = np.mean(data_pairs[:,np.r_[0:state_size, (state_size + 2):-2]],axis=(0))
dyn_settings.mean = 0
std = np.std(data_pairs[:,np.r_[0:state_size, (state_size + 2):-2]],axis=(0))
dyn_settings.std = 1 #std

print(f'mean and std of datasets: {mean} and {std}')
print(f'successes: {np.where(data_pairs[:,-2] == True)[0].shape[0]}, failures {np.where(data_pairs[:,-1] == True)[0].shape[0]}')
data_pairs_norm = np.copy(data_pairs)
print(f'Data shape {data_pairs_norm.shape}')
# data_pairs_norm[:,np.r_[0:state_size, (state_size + 2):-2]] = (data_pairs_norm[:,np.r_[0:state_size, (state_size + 2):-2]] - mean) / std
print(f'Number of pairs: {data_pairs_norm.shape[0]}')
reached = reached.reshape(-2)
violated = violated.reshape(-2)

print(f'Number of pairs with success {np.where(data_pairs[:,-2]==True)[0].shape[0]} , number of pairs with failure {np.where(data_pairs[:,-1]==True)[0].shape[0]}')
ratio = np.where(data_pairs[:,-1]==True)[0].shape[0]/np.where(data_pairs[:,-2]==True)[0].shape[0]
print(f'ratio failures/success = {ratio}, augment success to have ratio 1')

# if ratio > 10:
#     data_pairs = np.vstack((data_pairs,np.tile(data_pairs[np.where(data_pairs[:,-2]==True)[0]],(int(ratio-1),1))))
#     print(f'Number of pairs with success after rebalancing {np.where(data_pairs[:,-2]==True)[0].shape[0]} , number of pairs with failure {np.where(data_pairs[:,-1]==True)[0].shape[0]}')



# import sys
# sys.exit()
# # concatenate single states in pairs for TD learning
# even_length = get_even(int(norm_data_state.shape[0] / 2))
# data_pairs = norm_data_state.reshape((even_length, 2, norm_data_state.shape[1]))
# data_pairs = norm_data_state.reshape(
#     even_length, 2 * norm_data_state.shape[1]
# )  # n_states*2 x 28

# split dataset in train and test
train_size = int(0.99 * data_pairs_norm.shape[0])

x_train, x_test = data_pairs_norm[:train_size], data_pairs_norm[train_size:]

# x_train=x_train[:20000]
# transform to torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.Tensor(x_train).to(device)
# x_train = x_train[:int(x_train.shape[0]/10)]

class LogLoss(nn.Module):
    def __init__(self):
        super(LogLoss, self).__init__()
        print(f'Len {len(sys.argv)} argv  {sys.argv}')
        self.alpha = 1e-2 if len(sys.argv) < 3 else float(sys.argv[2])
        self.log_alpha = np.log10(self.alpha)
        self.log_1_m_alpha = np.log10(1-self.alpha)

    def log_sum(self,l1, l2):
        # compute log(x1+x2) given: l1=log(x1) and l2=log(x2)
        return (torch.max(l1, l2) + torch.log10(1+torch.pow(10,-torch.abs(l1-l2)))).detach()

    def forward(self, x, x_target):
        # l1 = (x + self.log_1_m_alpha).detach()
        # l2 = x_target + self.log_alpha

        loss = torch.nn.functional.mse_loss(torch.pow(10,x),torch.pow(10,x_target))  # Mean Squared Error 
        # loss = torch.nn.functional.smooth_l1_loss(x,self.log_sum(l1,l2),beta=0.6)
        return loss

activation_functions= [[nn.ReLU(),nn.Sigmoid()],[nn.ReLU(),nn.Identity()]]
if dyn_settings.log_repr:
    act = activation_functions[1]
else:
    act = activation_functions[0]

# network
input_size = state_size
nn_V_safe = NeuralNetwork(state_size, 256, 1, 2,log_representation=dyn_settings.log_repr,mean=mean,std=std, activation_hidden=act[0],activation_output=act[1]).to(device)
nn_target = NeuralNetwork(state_size, 256, 1, 2,log_representation=dyn_settings.log_repr,mean=mean,std=std, activation_hidden=act[0],activation_output=act[1]).to(device)
nn_target.load_state_dict(nn_V_safe.state_dict())
for param in nn_target.parameters():
    param.requires_grad = False

if dyn_settings.log_repr:
    loss_fn = LogLoss()
else:
    loss_fn = torch.nn.HuberLoss()
optimizer = torch.optim.Adam(
    nn_V_safe.parameters(),
    lr=5e-4,
    #weight_decay=2e-5,
    amsgrad=False,
)
batch_size = 1024
regressor = RegressionNN(batch_size, nn_V_safe,nn_target, loss_fn, optimizer, dyn_settings)

# training
epochs = 500
print("Start training\n")
train_evol = regressor.training(x_train, epochs)
print("Training completed\n")

# print('Evaluate the model')
# rmse_train, rel_err = regressor.testing(x_train_val, y_train_val)
# print(f'RMSE on Training data: {rmse_train:.5f}')
# print(f'Maximum error wrt training data: {torch.max(rel_err).item():.5f}')


# Save the model
torch.save({"mean": mean, "std": std,
           "model": nn_V_safe.state_dict()}, "nn_V_safe.pt")


# Plot the loss evolution
plt.figure()
plt.grid(True, which="both")
plt.semilogy(train_evol, label="Training", c="b", lw=2)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE Loss (LP filtered)")
plt.title(f"Training evolution")
plt.savefig(f"evolution_training.png")

# test

# compute Value of the network

V_net =  nn_V_safe.forward(torch.from_numpy(((state_grid-mean)/std).reshape((state_grid.shape[1],1))).float().clone().to(device)).detach().cpu().numpy()



plt.figure(figsize=(10,5))
plt.grid(True)
plt.gca().set_aspect('equal')
plt.plot(state_grid, V_prob, 'blue', label="V_prob")
plt.plot(state_grid, V_net, 'red', label="V_net")
plt.xlabel('State Grid')
plt.ylabel('V')
plt.xticks(np.arange(np.min(state_grid),np.max(state_grid)+0.2,0.2),fontsize=10)
plt.legend()
plt.title(f'V(x)  x_th: {x_th} u_max: {u_max} d_max: {d_max} noise: {distr}')
plt.savefig(f'plots/V_comparison_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.png')
# plt.show()
plt.close()