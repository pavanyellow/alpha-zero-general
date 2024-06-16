from model import SparseAutoencoder, train_autoencoder
import torch

def generate_data(input_dim, num_samples):
    return torch.randn(num_samples, input_dim) 