# ====================================
#         TD-Learning for Safety-V(x)
# ====================================

import os
import time
import copy
import math
import pickle
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
from system_dyn import System_conf
from datetime import datetime
import sys
now = datetime.now().strftime("%Y-%m-%d_%H-%M")


# ====================================
#      Environment Configuration
# ====================================
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "True"

# ====================================
#            Load Dataset
# ====================================
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

state_size = 20
log_learning = True

# data = np.load(f"data/training_data_interrupted_n{state_size}.npy",mmap_mode='r')
data = np.load(f"data/training_data_interrupted_different_n{state_size}.npy",mmap_mode='r')

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

data_pairs = data_pairs[:5_000_000]
# V_prob = np.load("data/V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy")
V_prob = np.load("data/V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian_different_problem.npy")

state_grid = np.load("data/grid.npy")
env_settings = System_conf()

state_grid_learning = np.linspace(env_settings.x_min,env_settings.x_max,1000+1)


input_dim = state_size

errors, errors_log = [], []

# states = data_pairs[:, :input_dim]
# next_states = data_pairs[:, input_dim+2:-2]
# failed = data_pairs[:, -1:]
# succeded = data_pairs[:, -2]

print("Observation dataset shape:", data_pairs.shape)

# ====================================
#         State Normalization
# ====================================
def normalize_states(states):
    mean_x = np.mean(states[:,:input_dim], axis=0)
    std_x = np.std(states[:,:input_dim], axis=0)

    mean_x_next = np.mean(states[:,-(input_dim+2):-2], axis=0)
    std_x_next = np.std(states[:,-(input_dim+2):-2], axis=0) 

    mean_tot = (mean_x + mean_x_next)/2
    std_tot = (std_x + std_x_next)/2

    states[:,:input_dim] = (states[:,:input_dim] - mean_tot) / (std_tot + 1e-8)
    states[:,-(input_dim+2):-2] = (states[:,-(input_dim+2):-2] - mean_tot) / (std_tot + 1e-8)
    return mean_tot,std_tot

mean, std = normalize_states(data_pairs)
# mean,std=np.zeros(state_size),np.ones(state_size)
# _, _ = normalize_states(data_pairs[:,-(input_dim+2):-2])

# ====================================
#         Convert to JAX tensors
# ====================================
pairs = jnp.array(data_pairs, dtype=jnp.float32)
# next_states = jnp.array(next_states, dtype=jnp.float32)
# failed = jnp.array(failed, dtype=jnp.float32)
# succeded = jnp.array(succeded, dtype=jnp.float32)

# ====================================
#       Define Neural Network
# ====================================
if not(log_learning):
    class ValueNetwork(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)
            x = nn.Dense(256)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)
            x = nn.Dense(128)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)
            # Output initialized to 1 (probability of survival)
            x = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)
            x = nn.sigmoid(x) 
            # x = nn.Dense(1)(x)
            return x.squeeze(-1)
else:
    class ValueNetwork(nn.Module):
        @nn.compact
        def __call__(self, x):
            # x = nn.Dense(512)(x)
            # x = nn.LayerNorm()(x)
            # x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(256)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(128)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(1)(x)
            # x = nn.Dense(1,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.constant(0))(x)

            # x = nn.leaky_relu(x)
            # x = nn.Dense(1)(x)
            return x.squeeze(-1)


# ====================================
#    Initialize / Load Model Params
# ====================================
key = jax.random.PRNGKey(42)
model = ValueNetwork()
params = model.init(key, jnp.ones((1, input_dim)))        # network weights and shapes initialized
# target_params = copy.deepcopy(params)

# ====================================
#       Optimizer & Train State
# ====================================
learning_rate = 5e-5
tau=0.005
optimizer = optax.adam(learning_rate)

class TrainState(train_state.TrainState):   # class equal to parent class
    pass

state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# ====================================
#           Loss Function
# ====================================
""" def loss_fn(params, batch_states, batch_next_states, batch_dones, batch_cp, target_params):
    # compute mask: 1 se stato valido, 0 se padding
    valid_mask = jnp.any(batch_states != 0, axis=1).astype(jnp.float32)

    # usual target
    V_s = model.apply(params, batch_states)
    V_next = jax.lax.stop_gradient(model.apply(target_params, batch_next_states))
    indicators = 1.0 - batch_dones
    target = indicators * ((1 - batch_cp) * V_next + batch_cp)

    loss = jnp.square(V_s - target) * valid_mask
    return jnp.sum(loss) / jnp.sum(valid_mask) """

alpha=0.01
if len(sys.argv) < 2:
    alpha=0.01
else:
    alpha = float(sys.argv[1])


if not(log_learning):
    @jax.jit
    def loss_fn(params, pairs):
        V_s = model.apply(params, pairs[:,:input_dim])
        V_next = jax.lax.stop_gradient(model.apply(params, pairs[:,-(input_dim+2):-2]))
        indicators = 1.0 - pairs[:,-1]
        target = indicators * ((1-pairs[:,-2])*V_next + pairs[:,-2])
        return jnp.mean(jnp.square(V_s - target))
else:
    @jax.jit
    def log_sum(l1, l2):
        # compute log(x1+x2) given: l1=log(x1) and l2=log(x2)
        return jax.lax.stop_gradient(jnp.maximum(l1, l2) + jnp.log10(1+jnp.pow(10,-jnp.abs(l1-l2))))
    
   
    log_1_m_alpha = jnp.log10(1-alpha)
    log_alpha = jnp.log10(alpha)
    # @jax.jit
    # def loss_fn(params, pairs):
    #     V_s = model.apply(params, pairs[:,:input_dim])
    #     V_next = jax.lax.stop_gradient(jnp.clip(model.apply(params, pairs[:,-(input_dim+2):-2]),max=0.001))
    #     indicators = 1.0 - pairs[:,-1]
    #     target = indicators * ((1-pairs[:,-2])*V_next/target_reached + pairs[:,-2]) * target_reached

    #     l1 = V_s + log_1_m_alpha
    #     l2 = target + log_alpha
        
    #     return jnp.mean(jnp.square(V_s - log_sum(l1,l2)))

    
    target_fail = 0
    target_reached = -4
    @jax.jit
    def loss_fn(params, pairs):
        V_s = model.apply(params, pairs[:,:input_dim])
        V_next = jax.lax.stop_gradient(model.apply(params, pairs[:,-(input_dim+2):-2]))
        indicators = 1.0 - pairs[:,-1]
        target = jnp.minimum(-1e-5,indicators * ((1-pairs[:,-2])*V_next/target_reached + pairs[:,-2]) * target_reached + (1-indicators) * target_fail)
        l1 = V_s + log_1_m_alpha
        l2 = target + log_alpha
        target_alpha = log_sum(l1,l2) 

        # return jnp.mean((jnp.square(V_s - target_alpha)))
        # return jnp.mean(jnp.log10(jnp.abs(10**V_s - 10**target_alpha + 1e-8)))
    

        # loss = indicators * ((1-pairs[:,-2])*V_next/target_reached + pairs[:,-2]) * target_reached + (1-indicators) * target_fail
        # return optax.losses.squared_error(jnp.power(10,V_s),jnp.power(10,target)).mean()
        return (jnp.log10(jnp.abs(jnp.power(10,V_s)-jnp.power(10,target) + 1e-8))).mean()
        # return (jnp.log10(jnp.abs(jnp.power(10,V_s)-jnp.power(10,target) + 1e-8))).mean()
    
    
        # return jnp.mean(jnp.log10(jnp.abs(10**V_s - 10**target + 1e-8)))


@jax.jit
def train_step(state,pairs):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, pairs)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def update_target_params(target_params, params, tau):
    """
    Soft update of the target network
    """
    return jax.tree_util.tree_map(lambda t, p: tau * p + (1 - tau) * t, target_params, params)

# @jax.jit
# def get_batches(pairs, batch_size):
#     dataset_len = pairs.shape[0]
#     rand_indx = np.random.permutation(dataset_len)

#     states_batches = pairs[rand_indx]

#     full_batches_num = dataset_len // batch_size
#     split_batch_index = int(full_batches_num*batch_size)
#     partial_batch = dataset_len % batch_size

#     batches = jnp.split(states_batches[:split_batch_index],full_batches_num)
#     batches.append(states_batches[-partial_batch:])
#     return batches

# @jax.jit
def get_batches(pairs, batch_size, full_batches_num, partial_batch_length, key):
    # Get the dataset length
    dataset_len = pairs.shape[0]

    # Generate a permutation of indices
    rand_indx = jax.random.permutation(key, dataset_len)

    # Shuffle the dataset using the permuted indices
    shuffled_pairs = pairs[rand_indx]

    # Split the shuffled data into full batches
    full_batches = jnp.array_split(shuffled_pairs[:full_batches_num * batch_size], full_batches_num)

    
    last_batch = shuffled_pairs[-partial_batch_length:]
    full_batches.append(last_batch)

    return full_batches

def save_errors(precision,ground_truth_log,ground_truth_normal,error_log,error_normal):
    ground_truth_log = np.maximum(ground_truth_log,-4)
    indx_minus1 = np.where(state_grid < -1)[0][-1] 
    x_sx_idx, x_dx_idx = np.abs(ground_truth_log[:indx_minus1] -(np.log10(precision))).argmin() , indx_minus1 + np.abs(ground_truth_log[indx_minus1:] -(np.log10(precision))).argmin()
    ground_truth_log = np.maximum(ground_truth_log,-3)
    V_tot = jax.lax.stop_gradient(model.apply(
                state.params,
                jnp.array(((np.tile(state_grid,(state_size,1)).T - mean)/(std+1e-8)),dtype=jnp.float32).reshape(V_prob.shape[0],input_dim))
        )
    if log_learning:
        V_sx = jax.lax.stop_gradient(model.apply(
                    state.params,
                    jnp.array(((np.array([state_grid[x_sx_idx]]*input_dim) - mean)/(std+1e-8)),dtype=jnp.float32))
            )
        V_dx = jax.lax.stop_gradient(model.apply(
                    state.params,
                    jnp.array(((np.array([state_grid[x_dx_idx]]*input_dim) - mean)/(std+1e-8)),dtype=jnp.float32))
            )
        error = np.linalg.norm(np.array([V_sx,V_dx]) - np.array([ground_truth_log[x_sx_idx],ground_truth_log[x_dx_idx]]))
        error_tot =  np.linalg.norm((1-10**V_tot) - ground_truth_normal)  
    else:
        V_sx = jax.lax.stop_gradient(model.apply(
                    state.params,
                    jnp.array(((np.array([state_grid[x_sx_idx]]*input_dim) - mean)/(std+1e-8)),dtype=jnp.float32))
            )
        V_dx = jax.lax.stop_gradient(model.apply(
                    state.params,
                    jnp.array(((np.array([state_grid[x_dx_idx]]*input_dim) - mean)/(std+1e-8)),dtype=jnp.float32))
            )
        V_sx, V_dx = np.log10(1-V_sx), np.log10(1-V_dx)
        error = np.linalg.norm(np.array([V_sx,V_dx]) - np.array([ground_truth_log[x_sx_idx],ground_truth_log[x_dx_idx]]))
        error_tot =  np.linalg.norm(V_tot - np.maximum(ground_truth_normal,precision))  
    error_log.append(error)
    error_normal.append(error_tot)
    np.save(f'errors/error_extrema_size_{state_size}_log_learning_{log_learning}.npy',error_log)
    np.save(f'errors/error_normal_size_{state_size}_log_learning_{log_learning}.npy',error_normal)
    np.save(f'{saving_path}/error_extrema_size_{state_size}_log_learning_{log_learning}.npy',error_log)
    np.save(f'{saving_path}/error_normal_size_{state_size}_log_learning_{log_learning}.npy',error_normal)

    # print(f'{state_grid[x_sx_idx]} val {V_sx}')
    # print(f'{state_grid[x_dx_idx]} val {V_tot[x_dx_idx]}')

    
    # plt.figure()
    # plt.grid(True)
    # plt.plot(
    #     state_grid, np.log10(1-V_tot), "blue", label="V_prob"
    # )
    # # plt.plot(grid_V_stat, V_stat, "green", label="V_stat")
    # plt.xlabel("State Grid")
    # plt.ylabel("V")

    # plt.legend()
    # plt.show()
    # plt.close()

        


def plot_error(errors,error,errors_log,error_log,step=1):
    errors.append(error)
    errors_log.append(error_log)
    plt.figure()
    plt.grid(True)
    plt.plot(
        np.arange(len(errors))*step, errors, "blue", label="error - traditional space",linewidth=2, linestyle='-', marker='o'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.legend()
    plt.title(
        f"ERROR V(x)"
    )
    plt.savefig(
        f"plots/state_size_{input_dim}_log_learning_{log_learning}_ERROR_JAX.png"
    )
    plt.savefig(
        f"{saving_path}/state_size_{input_dim}_log_learning_{log_learning}_ERROR_JAX.png"
    )
    plt.close()

    plt.figure()
    plt.grid(True)
    plt.plot(
        np.arange(len(errors_log))*step, errors_log, "blue", label="error - log space",linewidth=2, linestyle='-', marker='o'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.legend()
    plt.title(
        f"ERROR_LOG V(x)"
    )
    plt.savefig(
        f"plots/state_size_{input_dim}_log_learning_{log_learning}_ERROR_LOG_JAX.png"
    )
    plt.savefig(
        f"{saving_path}/state_size_{input_dim}_log_learning_{log_learning}_ERROR_LOG_JAX.png"
    )
    plt.close()


def plot_training(ep,loss,step=1):
    if ep % step==0:
        V_net = jax.lax.stop_gradient(model.apply(
                state.params,
                jnp.array(((np.tile(state_grid,(state_size,1)).T - mean)/(std+1e-8)),dtype=jnp.float32).reshape(V_prob.shape[0],input_dim))
        )

        V_prob_ = np.copy(V_prob)
        V_prob_ = np.power(V_prob_,input_dim)

        if not(log_learning):
            err = np.linalg.norm(V_prob_-V_net)
            err_log = np.linalg.norm(np.log10(1-V_prob_) - np.log10(1-V_net))

            plt.figure()
            plt.grid(True)
            plt.plot(
                state_grid, V_prob_, "blue", label="V_prob"
            )
            # plt.xticks(np.arange(np.min(state_grid), np.max(
            #     state_grid)+0.2, 0.2), fontsize=10)
            # plt.gca().set_aspect("equal")
            plt.plot(state_grid, V_net, "red", label="V_net")
            # plt.plot(grid_V_stat, V_stat, "green", label="V_stat")
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"V(x) epoch {ep}  x_th: {env_settings.x_th} u_max: {env_settings.u_max} d_max: {env_settings.d_max} noise: {env_settings.distr}"
            )

            plt.savefig(
                f"plots/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_ep{ep}_JAX.png"
            )
            plt.savefig(
                f"{saving_path}/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_ep{ep}_JAX.png"
            )
            plt.close()

            plt.figure()
            plt.grid(True)
            plt.plot(
                state_grid,
                np.log10(1 - V_prob_),
                "blue",
                label="V_prob",
            )
            plt.plot(
                state_grid, np.log10(1 - V_net), "red", label="V_net"
            )
            # plt.xticks(np.arange(np.min(state_grid), np.max(
            #     state_grid)+0.2, 0.2), fontsize=10)
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"V(x) epoch {ep}  x_th: {env_settings.x_th} u_max: {env_settings.u_max} d_max: {env_settings.d_max} noise: {env_settings.distr} log_10(1-V(x))"
            )
            plt.savefig(
                f"plots/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_log_repr_TRUE_ep{ep}_JAX.png"
            )
            plt.savefig(
                f"{saving_path}/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_log_repr_TRUE_ep{ep}_JAX.png"
            )
            # plt.show()
            plt.close()
        else:
            err_log = np.linalg.norm(np.maximum(np.log10(1-V_prob_),-10)-V_net)
            err = np.linalg.norm(V_prob_ - (1-np.power(10,V_net)))

            plt.figure()
            plt.grid(True)
            plt.plot(
                state_grid, V_prob_, "blue", label="V_prob"
            )
            # plt.xticks(np.arange(np.min(state_grid), np.max(
            #     state_grid)+0.2, 0.2), fontsize=10)
            # plt.gca().set_aspect("equal")
            plt.plot(state_grid, 1-np.power(10,V_net), "red", label="V_net")
            # plt.plot(grid_V_stat, V_stat, "green", label="V_stat")
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"V(x) epoch {ep}  x_th: {env_settings.x_th} u_max: {env_settings.u_max} d_max: {env_settings.d_max} noise: {env_settings.distr}"
            )

            plt.savefig(
                f"plots/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_ep{ep}_JAX.png"
            )
            plt.savefig(
                f"{saving_path}/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_ep{ep}_JAX.png"
            )
            plt.close()

            plt.figure()
            plt.grid(True)
            plt.plot(
                state_grid,
                np.log10(1 - V_prob_),
                "blue",
                label="V_prob",
            )
            plt.plot(
                state_grid, V_net, "red", label="V_net"
            )
            # plt.xticks(np.arange(np.min(state_grid), np.max(
            #     state_grid)+0.2, 0.2), fontsize=10)
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"V(x) epoch {ep}  x_th: {env_settings.x_th} u_max: {env_settings.u_max} d_max: {env_settings.d_max} noise: {env_settings.distr} log_10(1-V(x))"
            )
            plt.savefig(
                f"plots/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_log_repr_TRUE_ep{ep}_JAX.png"
            )
            plt.savefig(
                f"{saving_path}/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_log_repr_TRUE_ep{ep}_JAX.png"
            )
            # plt.show()
            plt.close()

            plt.figure()
            plt.grid(True)
            plt.plot(
                state_grid,
                np.log10(1 - V_prob_),
                "blue",
                label="V_prob",
            )
            plt.plot(
                state_grid, V_net, "red", label="V_net"
            )
            # plt.xticks(np.arange(np.min(state_grid), np.max(
            #     state_grid)+0.2, 0.2), fontsize=10)
            plt.xlabel("State Grid")
            plt.ylabel("V")

            plt.legend()
            plt.title(
                f"V(x) epoch x_th: {env_settings.x_th} u_max: {env_settings.u_max} d_max: {env_settings.d_max} noise: {env_settings.distr} log_10(1-V(x))"
            )
            plt.savefig(
                f"plots/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_log_repr_TRUE_JAX.png"
            )
            plt.savefig(
                f"{saving_path}/state_size_{input_dim}_log_learning_{log_learning}_V_comparison_x_th:{env_settings.x_th}_u_max:_{env_settings.u_max}_d_max:_{env_settings.d_max}_distr_{env_settings.distr}_log_repr_TRUE_JAX.png"
            )
            # plt.show()
            plt.close()

        plot_error(errors,err,errors_log,err_log,step)
        plt.figure()
        plt.grid(True)
        plt.yscale('log')  # log scale on y-axis
        plt.plot(
            np.arange(len(losses)),
            losses,
            "blue",
            label="loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.legend()
        plt.title(
            f"x_th: {env_settings.x_th} u_max: {env_settings.u_max} d_max: {env_settings.d_max} noise: {env_settings.distr} loss"
        )
        plt.savefig(
            f"plots/state_size_{input_dim}_log_learning_{log_learning}x_th: {env_settings.x_th} u_max: {env_settings.u_max} d_max: {env_settings.d_max} noise: {env_settings.distr}_loss_JAX.png"
        )
        plt.savefig(
            f"{saving_path}/state_size_{input_dim}_log_learning_{log_learning}x_th: {env_settings.x_th} u_max: {env_settings.u_max} d_max: {env_settings.d_max} noise: {env_settings.distr}_loss_JAX.png"
        )
        # plt.show()
        plt.close()

# ====================================
#           Training Loop
# ====================================
epochs = 4000
batch_size = 1024 * 8
rng = jax.random.PRNGKey(int(time.time()))
losses, min_losses, max_losses = [], [], []

saving_path = f'plots/{now}_state_size_{state_size}_alpha_{alpha}_lr_{learning_rate}_batch_{batch_size}'
plot_folder = os.mkdir(saving_path)


# Calculate the number of full batches and partial batch size
full_batches_num = pairs.shape[0] // batch_size
partial_batch_length = pairs.shape[0] % batch_size

losses,errors_log_space,errors_normal_space = [],[],[]
with tqdm(range(epochs), desc="Epoch") as pbar:
    for epoch in pbar:
        start_time = time.time()
        rng, subkey = jax.random.split(rng)
        batches = get_batches(pairs,batch_size, full_batches_num, partial_batch_length, rng)

        epoch_loss = 0
        batch_losses = []

        for batch in batches:
            state, loss = train_step(state, batch)
            # target_params = update_target_params(target_params, state.params, tau)
            epoch_loss += loss
            batch_losses.append(loss)


        epoch_avg = epoch_loss / batches[0].shape[0]
        losses.append(epoch_avg)
        min_losses.append(np.min(batch_losses))
        max_losses.append(np.max(batch_losses))
        pbar.set_postfix({"Loss": f"{epoch_avg:.2e}", "Hz": f"{1 / (time.time() - start_time):.2f}"})

        plot_training(epoch,losses,5)
        save_errors(1e-3,np.log10(1-np.power(V_prob,input_dim)),np.power(V_prob,input_dim),errors_log_space,errors_normal_space)

        rng = subkey

# ====================================
#           Save Model
# ====================================
model_path = f'models/NN_JAX_{state_size}_log_learning_{log_learning}'
model_path = f'{saving_path}/NN_JAX_{state_size}_log_learning_{log_learning}'
with open(model_path, 'wb') as f:
    pickle.dump({'model_params': state.params, 'mean': mean, 'std': std}, f)
print("Model saved at", model_path)

# ====================================
#       Value Estimate Example
# ====================================
# failed_episodes = states[dones[:,-1] == 1,:,:]
# cp_reached_episodes = states[capt_p[:,-1] == 1,:,:]

# sample_state = cp_reached_episodes[np.random.randint(0,cp_reached_episodes.shape[0]), 0, :]
# v_est = model.apply(state.params, sample_state)
# print(f"Estimated V(x) (prob. survival) for good state: {v_est.item():.4f}")

# sample_state = failed_episodes[np.random.randint(0,failed_episodes.shape[0]), 0, :]
# v_est = model.apply(state.params, sample_state)
# print(f"Estimated V(x) (prob. survival) for bad state: {v_est.item():.4f}")


# # ====================================
# #       Value Estimate Statistics
# # ====================================

# # Stimiamo V(x) per ogni primo stato di ogni episodio
# all_v_estimates = jnp.array([model.apply(state.params, s[0]) for s in states])

# # Identifica episodi che hanno avuto una terminazione
# terminated_mask = jnp.any(dones == 1, axis=1)
# non_terminated_mask = ~terminated_mask

# # Seleziona stime di V(x) per i due gruppi
# v_est_terminated = all_v_estimates[terminated_mask]
# v_est_non_terminated = all_v_estimates[non_terminated_mask]

# # Calcola statistiche
# mean_terminated = jnp.mean(v_est_terminated)
# std_terminated = jnp.std(v_est_terminated)

# mean_non_terminated = jnp.mean(v_est_non_terminated)
# std_non_terminated = jnp.std(v_est_non_terminated)

# # Stampa risultati
# print("\n===== Value Estimate Statistics =====")
# print(f"Episodes with termination   : n = {v_est_terminated.shape[0]}")
# print(f"Mean V(x): {mean_terminated:.4f}, Std: {std_terminated:.4f}")
# print()
# print(f"Episodes without termination: n = {v_est_non_terminated.shape[0]}")
# print(f"Mean V(x): {mean_non_terminated:.4f}, Std: {std_non_terminated:.4f}")

# print("\n===== All V Estimates for Terminated Episodes =====")
# print(v_est_terminated)
# print("=======================================")
