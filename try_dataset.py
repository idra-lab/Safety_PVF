from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

state_size = 14
data = np.load(f"data/training_data_interrupted__tabular_n{state_size}.npy",mmap_mode='r')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SupervisedDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
 
    def __len__(self):
        # this should return the size of the dataset
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features

x = data[:,:,:state_size].reshape(-1, state_size)
y = data[:,:,state_size:].reshape(-1, data.shape[-1] - state_size)

dataset = SupervisedDataset(x,y)
loader = DataLoader(dataset,shuffle=True, batch_size=1)

for x in loader:
    print(x)



