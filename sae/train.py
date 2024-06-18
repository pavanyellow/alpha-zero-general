import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from model import SparseAutoencoder, sae_loss

@dataclass
class SAETrainConfig:
    input_dim: int 
    hidden_dim: int
    num_epochs: int = 2048
    batch_size: int = 512
    l1_penalty: float = 2
    learning_rate: float = 3e-4
    val_split: float = 0.2



def split_data(original_dataset, val_split : float):
    dataset_size = original_dataset.shape[0]
    dataset = original_dataset[torch.randperm(dataset_size)]

    train_dataset, val_dataset = dataset[int(dataset_size*val_split):], dataset[:int(dataset_size*val_split)]
    dataset_size = original_dataset.shape[0]
    train_dataset, val_dataset = original_dataset[int(dataset_size*val_split):], original_dataset[:int(dataset_size*val_split)]

    return train_dataset, val_dataset

@torch.no_grad()
def estimate_losses(model, train_dataset, val_dataset, config: SAETrainConfig):
    model.eval()
    eval_iters = 100
    losses = []
    l_zero = []
    l_one = []
    train_dataset = train_dataset[torch.randperm(train_dataset.shape[0])]
    val_dataset = val_dataset[torch.randperm(val_dataset.shape[0])]

    for dataset in [train_dataset, val_dataset]:
        loss = torch.zeros(eval_iters)
        for i in range(eval_iters):
            batch = dataset[i*config.batch_size:(i+1)*config.batch_size]
            reconstructed, encoded = model(batch)
            loss_ = sae_loss(batch, reconstructed, encoded, model.decoder.weight, config.l1_penalty)
            loss[i] = loss_
            l_zero.append(encoded.norm(0, dim = 1, keepdim=True).mean())
            l_one.append(encoded.norm(1, dim = 1, keepdim=True).mean())
        
        losses.append(loss.mean().item())
    model.train()
    return losses, l_zero # Val loss is losses[1]



def train_autoencoder(model, dataset: torch.Tensor, config: SAETrainConfig) -> None:

    print (f"Training autoencoder with config: {config} and shape: {dataset.shape} \n")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs * 0.8) / (num_epochs * 0.2))

    current_norm = torch.norm(dataset, dim=1, keepdim=True).mean()
    dataset = ((dataset * (config.input_dim**0.5)) / current_norm)

    train_dataset, val_dataset = split_data(dataset, config.val_split)
    #train_iters = min(config.num_epochs, train_dataset.shape[0] // config.batch_size)
    train_iters = config.num_epochs

    data_iter = iter(DataLoader(train_dataset.clone().detach(), batch_size=config.batch_size, shuffle=True)) # No repetition of data

    losses,l0 = estimate_losses(model, train_dataset, val_dataset, config)
    print(f'Starting val loss: {losses[1]:.3f}  train loss: {losses[0]:.3f} l0: {l0[1]:.2f} \n')
    

    for epoch in range(config.num_epochs):
        #batch = next(data_iter)
        batch = train_dataset[torch.randperm(train_dataset.shape[0])][:config.batch_size]
        optimizer.zero_grad()
        reconstructed, encoded = model(batch)
        loss = sae_loss(batch, reconstructed, encoded, model.decoder.weight, config.l1_penalty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()
        epoch_print_size = ((config.num_epochs // 30)//100) * 100 or 50

        if epoch % epoch_print_size == 0:
            losses,l0 = estimate_losses(model, train_dataset, val_dataset, config)
            print(f'Epoch {epoch}/{train_iters} val loss: {losses[1]:.3f}  train loss: {losses[0]:.3f} l0: {l0[1]:.2f}')