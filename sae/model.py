import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.init_weights()

    def init_weights(self):
        
        # Initialize decoder weights (Wd)
        nn.init.xavier_uniform_(self.decoder.weight)
        with torch.no_grad():
            # Normalize columns of decoder weights
            norm = self.decoder.weight.norm(p=2, dim=0, keepdim=True)
            self.decoder.weight.div_(norm)
            # Rescale to desired norm
            self.decoder.weight.mul_(torch.rand(self.decoder.weight.size(1), 1).mul_(0.95).add_(0.05))

        # Initialize We to Wd^T
        self.encoder.weight.data.copy_(self.decoder.weight.data.t())

    def forward(self, x):
        encoded = F.relu(self.encoder(x) + self.encoder_bias)
        decoded = self.decoder(encoded) + self.decoder_bias
        return decoded, encoded

def loss(X, reconstructed_X, encoded_X, W_d, lambda_val):
    size = X.size(0)
    mse_loss = ((X - reconstructed_X) ** 2).sum(dim = -1).sum(dim = 0) # torch.sum((a-b)**2)
    sparsity_loss = lambda_val * (torch.norm(W_d, p=2, dim=0)*encoded_X).sum(dim = 1).sum(dim = 0) # torch.sum(torch.norm(wd, p=2, dim=0)*encoded)
    return (mse_loss + sparsity_loss) / size
            

def train_autoencoder(model, dataset, input_dim, hidden_dim, lambda_val=5, learning_rate=5e-5, num_epochs=2000, batch_size=16):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs * 0.8) / (num_epochs * 0.2))
    dataset = dataset * (input_dim / torch.norm(dataset, dim=1, keepdim=True)).sqrt()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed, encoded = model(batch)
            loss = loss(batch, reconstructed, encoded, model.decoder.weight, lambda_val * min(epoch / (num_epochs * 0.05), 1.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

input_dim = 10  # Example input dimension
hidden_dim = 100  # Example hidden dimension
dataset = torch.randn(10000, input_dim)  # Example dataset

model = SparseAutoencoder(input_dim, hidden_dim)
train_autoencoder(model, dataset, input_dim, hidden_dim)
