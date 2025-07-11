import torch
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
    




def train_autoencoder(model, dataloader, device, num_epochs=20, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch in dataloader:
            x_batch = x_batch[0].to(device)  # unpack TensorDataset
            x_recon = model(x_batch)
            loss = criterion(x_recon, x_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):4f}")