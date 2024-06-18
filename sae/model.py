import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass

@dataclass
class SAEConfig:
    input_dim: int 
    hidden_dim: int
    num_epochs: int = 2048
    batch_size: int = 512
    l1_penalty: float = 2
    learning_rate: float = 3e-4
    

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder : nn.Module = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder : nn.Module = nn.Linear(hidden_dim, input_dim, bias=False)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            norm = self.decoder.weight.norm(p=2, dim=0, keepdim=True)
            self.decoder.weight.div_(10*norm) # Set column norm to 0.1
        self.encoder.weight.data.copy_(self.decoder.weight.data.t())

    def forward(self, x):
        encoded = F.relu(self.encoder(x) + self.encoder_bias)
        decoded = self.decoder(encoded) + self.decoder_bias
        return decoded, encoded

def sae_loss(X, reconstructed_X, encoded_X, W_d, lambda_val):
    mse_loss = ((X - reconstructed_X) ** 2).sum(dim = -1).mean(0) # torch.sum((a-b)**2)
    sparsity_loss = lambda_val * ((torch.norm(W_d, p=2, dim=0)*encoded_X).sum(dim=1)).mean(0) # torch.sum(torch.norm(wd, p=2, dim=0)*encoded)
    #print(mse_loss, sparsity_loss)
    return (mse_loss + sparsity_loss)
            

def train_autoencoder(model, dataset, config: SAEConfig) -> None:

    print (f"Training autoencoder with config: {config} and dataset shape: {dataset.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs * 0.8) / (num_epochs * 0.2))
    current_norm = torch.norm(dataset, dim=1, keepdim=True).mean()
    dataset = ((dataset * (config.input_dim**0.5)) / current_norm)
    
    data_iter = iter(DataLoader(dataset, batch_size=config.batch_size, shuffle=True))

    if (config.num_epochs > dataset.shape[0] // config.batch_size):
        raise ValueError("Number of epochs is too large for the dataset size")
    
    for epoch in range(config.num_epochs):
        batch = next(data_iter)
        optimizer.zero_grad()
        reconstructed, encoded = model(batch)
        loss = sae_loss(batch, reconstructed, encoded, model.decoder.weight, config.l1_penalty)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()
        epoch_print_size = config.num_epochs // 16
        if epoch % epoch_print_size == 0:
            print(f'Epoch {epoch}/{config.num_epochs}, Loss: {loss.item()} l0: {encoded.norm(0, dim = 1, keepdim=True).mean()}')