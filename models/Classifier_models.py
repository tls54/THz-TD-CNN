import torch
import torch.nn as nn
from tqdm import tqdm


# TODO: Write in a verbose flag for long epoch training

## First basic CNN 
class CNN1D_Small(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(CNN1D_Small, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.mean(x, dim=2)  # Global average pooling
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
    
## Larger model 
class CNN1D_Large(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(CNN1D_Large, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


## Define a training process that works for any of the models

from tqdm import tqdm
import torch

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, verbose='epoch'):
    """
    Train a PyTorch model on THz TDS data.

    Args:
        model (nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: 'cuda' or 'cpu'.
        num_epochs (int): Number of training epochs.
        verbose (str): 'epoch', 'batch', or None. Controls tqdm verbosity.

    Returns:
        loss_values (list): List of average training loss per epoch.
        accuracy_values (list): List of accuracy per epoch.
    """
    model.to(device)
    model.train()

    loss_values = []
    accuracy_values = []

    epoch_iter = range(num_epochs)
    if verbose == 'epoch':
        epoch_iter = tqdm(epoch_iter, desc="Training epochs", unit='epoch')

    for epoch in epoch_iter:
        running_loss = 0.0
        correct = 0
        total = 0

        batch_iter = train_loader
        if verbose == 'batch':
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit='batch', leave=False)

        for inputs, labels in batch_iter:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        loss_values.append(epoch_loss)
        accuracy_values.append(epoch_accuracy)

        if verbose is None:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

    return loss_values, accuracy_values