import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class FCEncoder(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class FCDecoder(nn.Module):
    def __init__(self, latent_dim=32, output_dim=1024):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=32):
        super().__init__()
        self.encoder = FCEncoder(input_dim, latent_dim)
        self.decoder = FCDecoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    


def train_autoencoder(model, dataloader, device, num_epochs=20, lr=1e-3, verbose='epoch'):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loss_values = []

    epoch_iter = range(num_epochs)
    if verbose == 'epoch':
        epoch_iter = tqdm(epoch_iter, desc="Training epochs", unit='epoch')

    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        total = 0

        batch_iter = dataloader
        if verbose == 'batch':
            batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit='batch', leave=False)

        for x_batch in batch_iter:
            x_batch = x_batch[0].to(device)

            optimizer.zero_grad()
            x_recon = model(x_batch)
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            total += x_batch.size(0)

        epoch_loss = running_loss / total
        train_loss_values.append(epoch_loss)


    return train_loss_values