from model import SparseAutoencoder, train_autoencoder
import torch

directions = torch.randn(20, 5)

data = []

# Number of iterations (you have 1 in your loop)
iterations = 1000

# Generate data
for i in range(iterations):
    indices = torch.randperm(20)[:2]
    acts = (torch.rand(2, 1) * directions[indices]).sum(dim=0)
    data.append(acts)

data = torch.stack(data)

print(data.size())
