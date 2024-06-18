import torch
import torch.nn as nn
import torch.nn.functional as F


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







