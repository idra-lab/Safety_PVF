import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from net_learner import NeuralNetwork, RegressionNN, RegressionSupervisedTraditionalSpace, RegressionSupervisedLogTarget, RegressionSupervisedLogSpace, SupervisedDataset
from tqdm import tqdm
from system_dyn import u_max,d_max,x_th,distr,System_conf,state_size
import os
import torch.nn as nn
import sys

torch.cuda.empty_cache()

if len(sys.argv) < 2:
    print('Use state size specified in the script')
    state_size = 12
else:
    state_size = int(sys.argv[1])
    print(f'Use state size {state_size}')
# load data
# data = np.load(f"training_data_interrupted_n{state_size}.npy",mmap_mode='r')

class_type = [RegressionSupervisedTraditionalSpace, RegressionSupervisedLogTarget, RegressionSupervisedLogSpace, RegressionNN]

if len(sys.argv) < 3:
    print('Use target type specified in the script')
    target_type = 2
else:
    target_type = int(sys.argv[2])
    print(f'Use target type {target_type}')

if target_type <= 2:
    data_V = np.load(f"data/training_data_interrupted__tabular_n{state_size}.npy",mmap_mode='r')
elif target_type == 3:
    data = np.load(f"training_data_interrupted_n{state_size}.npy",mmap_mode='r')

activation_functions= [[nn.ELU(),nn.Sigmoid()],[nn.ELU(),nn.Sigmoid()],[nn.ReLU(),nn.LeakyReLU()],[nn.ELU(),nn.Sigmoid()]]

print("Dataset loaded:", data_V.shape)

def clean_data(data):
    """
    Execute on n_episodes x horizon x (states + reached + violated) data
    """
    data_clean = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (data[i, j, :] != np.zeros(data.shape[2])).any():
                data_clean.append(data[i,j])
    return np.array(data_clean)

dyn_settings = System_conf()
dyn_settings.state_size = state_size

if target_type <= 2:
    if not(os.path.exists(f'data/data_supervised_state_size{state_size}.npy')):
        data_clean_chunk = []
        # generate data in pairs
        step = int(data_V.shape[0]/1000)
        for jj in tqdm(range(0,data_V.shape[0],step)):
            data_chunk = data_V[jj:jj+step]
            data_clean_chunk.append(clean_data(data_chunk))
        data_clean = np.vstack(data_clean_chunk)

        np.save(f'data/data_supervised_state_size{state_size}.npy',data_clean)
    else: data_clean = np.load(f'data/data_supervised_state_size{state_size}.npy')

print(f'Data clean size: {data_clean.shape}')

dyn_settings.mean = 0
dyn_settings.std = 1 #std

# split dataset in train and test
train_size = int(0.99 * data_clean.shape[0])

# data_clean = data_clean[:int(data_clean.shape[0]/2)]
x_train, x_test = data_clean[:train_size], data_clean[train_size:]

# x_train=x_train[:20000]
# transform to torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.Tensor(x_train).to(device)
# x_train = SupervisedDataset(x_train,device)
# x_train = x_train[:int(x_train.shape[0]/10)]

class LogLoss(nn.Module):
    def __init__(self):
        super(LogLoss, self).__init__()
        self.alpha = 0.02
        self.log_alpha = np.log10(self.alpha)
        self.log_1_m_alpha = np.log10(1-self.alpha)

    def log_sum(self,l1, l2):
        # compute log(x1+x2) given: l1=log(x1) and l2=log(x2)
        return (torch.max(l1, l2) + torch.log10(1+ torch.pow(10,-torch.abs(l1-l2)))).detach()

    def forward(self, x, x_target):
        l1 = x + self.log_1_m_alpha
        l2 = x_target + self.log_alpha

        loss = torch.mean((x - self.log_sum(l1,l2)) ** 2)  # Mean Squared Error
        return loss

# network
input_size = state_size
nn_V_safe = NeuralNetwork(state_size, 512, 1, 2,log_representation=dyn_settings.log_repr,activation_hidden=activation_functions[target_type][0],activation_output=activation_functions[target_type][1]).to(device)
nn_target = NeuralNetwork(state_size, 512, 1, 2,activation_hidden=activation_functions[target_type][0],activation_output=activation_functions[target_type][1]).to(device)
nn_target.load_state_dict(nn_V_safe.state_dict())
for param in nn_target.parameters():
    param.requires_grad = False

if dyn_settings.log_repr:
    loss_fn = LogLoss()
else:
    loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    nn_V_safe.parameters(),
    lr=5e-4,
    #weight_decay=2e-5,
    amsgrad=True,
)
batch_size = 1024
regressor = class_type[target_type](batch_size, nn_V_safe,nn_target, loss_fn, optimizer, dyn_settings)

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
torch.save({"model": nn_V_safe.state_dict()}, f"models/nn_V_safe_size_{state_size}_target_{target_type}.pt")


# Plot the loss evolution
plt.figure()
plt.grid(True, which="both")
plt.semilogy(train_evol, label="Training", c="b", lw=2)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE Loss (LP filtered)")
plt.title(f"Training evolution")
plt.savefig(f"evolution_training_size_{state_size}_target_{target_type}.png")

# test