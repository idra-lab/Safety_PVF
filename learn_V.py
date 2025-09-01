import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from net_learner import NeuralNetwork, RegressionNN
from tqdm import tqdm
from system_dyn import u_max,d_max,x_th,distr,DynSystem
import os

# load data
data = np.load("training_data_not_interrupted.npy")
print("Dataset loaded:", data.shape)

data = data[:10000]
V_prob = np.load("V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy")
state_grid = np.load("grid.npy")

dyn_settings = DynSystem()
dyn_settings.V_prob = V_prob
dyn_settings.state_grid = state_grid

def generate_pairs(data):
    """
    Execute on n_episodes x horizon x (states + reached + violated) data
    """
    pairs = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 1):
            if (data[i, j, :] != np.zeros(data.shape[2])).any() and (
                data[i, j + 1, :] != np.zeros(data.shape[2])
            ).any():
                pairs.append(np.hstack((data[i, j, :], data[i, j + 1, :])))
    return np.array(pairs)


state_size = 1

# success and failures array
reached, violated = data[:, :, -2], data[:, :, -1]

# remove a dimension in the data, to obtain only states
# norm_data_state = norm_data_state.reshape(-1, norm_data_state.shape[2])

if not(os.path.exists('data_pairs.npy')):
    data_pairs_chunk = []
    # generate data in pairs
    step = int(data.shape[0]/1000)
    for jj in tqdm(range(0,data.shape[0],step)):
        data_chunk = data[jj:jj+step]
        data_pairs_chunk.append(generate_pairs(data_chunk))
    data_pairs = np.vstack(data_pairs_chunk)

    np.save('data_pairs.npy',data_pairs)
else: data_pairs = np.load('data_pairs.npy')
mean = np.mean(data_pairs[:,np.r_[0:state_size, (state_size + 2):-2]],axis=(0))
dyn_settings.mean = mean
std = np.std(data_pairs[:,np.r_[0:state_size, (state_size + 2):-2]],axis=(0))
dyn_settings.std = std

print(f'mean and std of datasets: {mean} and {std}')
print(f'successes: {np.where(data_pairs[:,-2] == True)[0].shape[0]}, failures {np.where(data_pairs[:,-1] == True)[0].shape[0]}')
data_pairs_norm = np.copy(data_pairs)
# data_pairs_norm[:,np.r_[0:state_size, (state_size + 2):-2]] = (data_pairs_norm[:,np.r_[0:state_size, (state_size + 2):-2]] - mean) / std
print(f'Number of pairs: {data_pairs_norm.shape[0]}')
reached = reached.reshape(-2)
violated = violated.reshape(-2)

print(f'Number of pairs with success {np.where(data_pairs[:,-2]==True)[0].shape[0]} , number of pairs with failure {np.where(data_pairs[:,-1]==True)[0].shape[0]}')

# # concatenate single states in pairs for TD learning
# even_length = get_even(int(norm_data_state.shape[0] / 2))
# data_pairs = norm_data_state.reshape((even_length, 2, norm_data_state.shape[1]))
# data_pairs = norm_data_state.reshape(
#     even_length, 2 * norm_data_state.shape[1]
# )  # n_states*2 x 28

# split dataset in train and test
train_size = int(0.99 * data_pairs_norm.shape[0])

x_train, x_test = data_pairs_norm[:train_size], data_pairs_norm[train_size:]

# transform to torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.Tensor(x_train).to(device)
# x_train = x_train[:int(x_train.shape[0]/10)]

# network
input_size = state_size
nn_V_safe = NeuralNetwork(state_size, 128, 1, 2,log_representation=dyn_settings.log_repr,mean=mean,std=std).to(device)
nn_target = NeuralNetwork(state_size, 128, 1, 2).to(device)
nn_target.load_state_dict(nn_V_safe.state_dict())
for param in nn_target.parameters():
    param.requires_grad = False

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    nn_V_safe.parameters(),
    lr=1e-4,
    weight_decay=2e-5,
    amsgrad=False,
)
batch_size = 1024
regressor = RegressionNN(batch_size, nn_V_safe,nn_target, loss_fn, optimizer, dyn_settings)

# training
epochs = 1000
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

V_net =  nn_V_safe.forward(torch.from_numpy(((state_grid-mean)/std).reshape((state_grid.sahape[1],1))).float().clone().to(device)).detach().cpu().numpy()



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