from tqdm import tqdm
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn

torch.set_printoptions(profile="full")


class NeuralNetwork(nn.Module):
    """A simple feedforward neural network."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        number_hidden,
        activation=nn.Sigmoid(),
        ub=None,
        log_representation=False,
        mean=0,
        std = 1
    ):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)

        # Hidden layers
        for _ in range(number_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.ReLU())

        self.linear_stack = nn.Sequential(*layers)

        self.ub = ub if ub is not None else 1
        self.initialize_weights()

        self.input_size = input_size

        self.log_repr = log_representation
        self.mean = mean
        self.std = std

    def forward(self, x):
        # out = self.linear_stack(x[:,:self.input_size])* self.ub
        out = self.linear_stack(x)

        return out * self.ub  # (out + 1) * self.ub / 2

    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def compute_target(self, x, successes, failures):
        targets = torch.zeros((x.shape[0], 1), dtype=torch.float32).to('cuda')

        # not ended states
        mask = ~(successes.bool() | failures.bool())

        if self.log_repr:
            targets[successes.bool()] = -10
            targets[failures.bool()] = 0.0

        else:
            targets[successes.bool()] = 1.0
            targets[failures.bool()] = 0.0

        targets[mask] = self.forward(x[mask]).detach()

        return targets


class RegressionNN:
    """Class that compute training and test of a neural network."""

    def __init__(self, batch_size, model, model_target, loss_fn, optimizer, settings):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_target = model_target
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.beta = 0.95
        self.batch_size = batch_size
        self.plot_train = True
        self.data_dir = ""
        self.settings = settings
        self.tau = 0.01

    def training(self, x_train: torch.Tensor, epochs):
        """Training of the neural network."""

        progress_bar = tqdm(total=epochs, desc="Training")

        # initially all targets 0
        val_targets = torch.zeros((x_train.shape[0],1), device=self.device)

        loss_evol_train = []
        loss_evol_val = []
        loss_lp = 1

        n = len(x_train)
        for ep in range(epochs):

            self.model.train()
            # Shuffle the data
            idx = torch.randperm(n)
            x_perm = x_train[idx]
            # Split in batches
            x_batches = torch.split(x_perm, self.batch_size)

            for x in x_batches:
                # x = x.to(self.device)

                # print(x[:10, -3])
                # print(x[:10, self.model.input_size + 2 : -2])
                # compute targets with network
                val_targets = self.model.compute_target(
                    x[:, self.model.input_size + 2 : -2], x[:,-2], x[:, -1]
                )
                # Forward pass
                V_pred = self.model( x[:,:self.model.input_size])

                # NAN CHECK
                if torch.isnan(V_pred).any():
                    print(f"NaN detected in outputs at epoch {ep}")
                    break
                # Compute the loss
                loss = self.loss_fn(V_pred, val_targets)

                if torch.isnan(loss).any():
                    print(f"NaN detected in loss at epoch {ep}")
                    break
                # Backward and optimize
                # print(f'loop : {ep}')
                self.optimizer.zero_grad()
                loss.backward()

                for name, param in self.model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print(
                            f"NaN detected in gradients of {name} at epoch {ep}")
                        break

                self.optimizer.step()

                # self.soft_update()

                loss_lp = self.beta * loss_lp + (1 - self.beta) * loss.item()

            if ep % 10 == 0:
                print(f'Loss training: {np.mean(loss_lp)}')
                self.plot_learning(ep)

            # x.detach().cpu()

            loss_evol_train.append(loss_lp)
            # # Validation
            # loss_val = self.validation(x_val, y_val)
            # if ep % 100 == 0:
            #     print(f'Loss training: {loss_lp}')
            #     print(f'Loss validation: {loss_val}')
            # loss_evol_val.append(loss_val)
            progress_bar.update(1)

        progress_bar.close()
        return loss_evol_train

    def validation(self, x_val, y_val):
        """Compute the loss wrt to validation data."""
        x_batches = torch.split(x_val, self.batch_size)
        y_batches = torch.split(y_val, self.batch_size)
        self.model.eval()
        tot_loss = 0
        y_out = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred = self.model(x)
                y_out.append(y_pred)
                loss = self.loss_fn(y_pred, y)
                tot_loss += loss.item()
            y_out = torch.cat(y_out, dim=0)
        return tot_loss / len(x_batches)

    def testing(self, x_test, y_test):
        """Compute the RMSE wrt to training or test data."""
        x_batches = torch.split(x_test, self.batch_size)
        y_batches = torch.split(y_test, self.batch_size)
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred.append(self.model(x))
            y_pred = torch.cat(y_pred, dim=0)
            rmse = torch.sqrt(mse_loss(y_pred, y_test)).item()
            rel_err = (
                y_pred - y_test
            ) / y_test  # torch.maximum(y_test, torch.Tensor([1.]).to(self.device))
        return rmse, rel_err

    def save_model(self):
        # Save the model
        torch.save({"mean": self.model.mean, "std": self.model.std,
           "model": self.model.state_dict()}, f"nn_V_safe_log_{self.model.log_repr}.pt")

    def plot_learning(self,ep=0):
        V_net = self.model.forward(torch.from_numpy((self.settings.state_grid).reshape(
            (self.settings.state_grid.shape[0], 1))).float().clone().to(self.device)).detach().cpu().numpy()

        print(f'output for -0.8: {self.model.forward(torch.from_numpy((np.array([[-0.8]])-self.settings.mean)/self.settings.std).float().to(self.device)).detach().cpu().numpy()}')

        plt.figure()
        plt.grid(True)
        if self.model.log_repr:
            # plt.yscale('log')
            plt.plot(self.settings.state_grid,
                np.log10(1-self.settings.V_prob), 'blue', label="V_prob")
            plt.plot(self.settings.state_grid, V_net, 'red', label="V_net")
            plt.xlabel('State Grid')
            plt.ylabel('V')

            plt.legend()
            plt.title(
                f'V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}')
            plt.savefig(
                f'plots/V_comparison_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_log_repr{self.settings.log_repr}.png')
            # plt.show()
            plt.close()

        else:
            plt.plot(self.settings.state_grid,
                    self.settings.V_prob, 'blue', label="V_prob")
            # plt.xticks(np.arange(np.min(self.settings.state_grid), np.max(
            #     self.settings.state_grid)+0.2, 0.2), fontsize=10)
            plt.gca().set_aspect('equal')
            plt.plot(self.settings.state_grid, V_net, 'red', label="V_net")
            plt.xlabel('State Grid')
            plt.ylabel('V')

            plt.legend()
            plt.title(
                f'V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr}')
            plt.savefig(
                f'plots/V_comparison_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_log_repr{self.settings.log_repr}.png')
            # plt.show()
            plt.close()
            
            plt.figure()
            plt.grid(True)
            plt.plot(self.settings.state_grid,
                np.log10(1-self.settings.V_prob), 'blue', label="V_prob")
            plt.plot(self.settings.state_grid, np.log10(1-V_net), 'red', label="V_net")
            # plt.xticks(np.arange(np.min(self.settings.state_grid), np.max(
            #     self.settings.state_grid)+0.2, 0.2), fontsize=10)
            plt.xlabel('State Grid')
            plt.ylabel('V')

            plt.legend()
            plt.title(
                f'V(x)  x_th: {self.settings.x_th} u_max: {self.settings.u_max} d_max: {self.settings.d_max} noise: {self.settings.distr} log_10(1-V(x))')
            plt.savefig(
                f'plots/V_comparison_x_th:{self.settings.x_th}_u_max:_{self.settings.u_max}_d_max:_{self.settings.d_max}_distr_{self.settings.distr}_log_repr.png')
            # plt.show()
            plt.close()
        

    def soft_update(self):
        for target_param, source_param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
        )