import torch
from model import SparseAutoencoder, train_autoencoder, SAEConfig


input_dim = 5 
hidden_dim = 30
real_features = 20


data = []

iterations = 2**18
no_active = 3
directions = torch.randn(real_features, 5)
# Generate data
for i in range(iterations):
    indices = torch.randperm(real_features)[:no_active]
    acts = (torch.randint(-2, 2, (no_active,1)) * directions[indices]).sum(dim=0)
    data.append(acts)

data = torch.stack(data)

model = SparseAutoencoder(input_dim, hidden_dim)
train_autoencoder(model, data, SAEConfig(input_dim, hidden_dim, num_epochs=2**12, batch_size=2**6, l1_penalty=2, learning_rate=3e-4))

