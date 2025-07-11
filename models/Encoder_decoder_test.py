import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # (1024 -> 512)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # (512 -> 256)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 32, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.decoder_input = nn.Linear(latent_dim, 256 * 32)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 256)),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # (256 -> 512)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),  # (512 -> 1024)
            nn.Tanh()  # Assuming normalized pulses in [-1, 1]
        )

    def forward(self, z):
        x = self.decoder_input(z)
        return self.decoder(x).squeeze(1)  # (B, 1024)

class Classifier(nn.Module):
    def __init__(self, latent_dim=32, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, z):
        return self.classifier(z)

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=32, num_classes=3):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        logits = self.classifier(z)
        return x_recon, logits
    




def train(model, dataloader, device, num_epochs=10, lambda_recon=1.0, lambda_cls=1.0):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_recon = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x_recon, logits = model(x_batch)

            loss_recon = criterion_recon(x_recon, x_batch)
            loss_cls = criterion_cls(logits, y_batch)

            loss = lambda_recon * loss_recon + lambda_cls * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")