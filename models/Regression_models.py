from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim



## Regression model inspired by large classifier architecture
class CNN1D_Regressor(nn.Module):
    def __init__(self, input_channels=1, output_dim=9):
        super(CNN1D_Regressor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(8, 64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(8, 128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(16, 256)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(16, 256)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, output_dim)  # 9 regression outputs

    def forward(self, x):
        x = self.pool(torch.relu(self.gn1(self.conv1(x))))
        x = self.pool(torch.relu(self.gn2(self.conv2(x))))
        x = self.pool(torch.relu(self.gn3(self.conv3(x))))
        x = self.pool(torch.relu(self.gn4(self.conv4(x))))
        x = self.global_pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        return x
    
# Expected output shape: 9D vector: [n1, k1, d1, n2, k2, d2, ...]


def train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=20,
    lr=1e-3,
    loss_fn=None,
    optimizer_cls=None,
    device=None,
    verbose_level='epoch'  # 'epoch' or 'batch'
):
    model.to(device)
    loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
    optimizer_cls = optimizer_cls if optimizer_cls is not None else optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs), desc='Epochs', disable=(verbose_level == 'batch')):
        model.train()
        running_train_loss = 0.0

        if verbose_level == 'batch':
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        else:
            loop = train_loader

        for batch in loop:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if verbose_level == 'batch':
                loop.set_postfix(train_loss=loss.item())

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        if verbose_level == 'epoch':
            tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.6f}")

        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)
                    val_loss = loss_fn(outputs, targets)
                    running_val_loss += val_loss.item()

            epoch_val_loss = running_val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)

            if verbose_level == 'epoch':
                tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Val Loss: {epoch_val_loss:.6f}")

    return train_losses, val_losses if val_loader is not None else train_losses