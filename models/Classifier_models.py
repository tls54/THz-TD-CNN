import torch
import torch.nn as nn
from tqdm import tqdm

# TODO: Convert batchnorm to GroupNorm

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
        self.fc2 = nn.Linear(64, num_classes)

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




## Define a training process that works for any of the models
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10,
                val_loader=None, scheduler=None, verbose='epoch'):
    model.to(device)
    model.train()

    train_loss_values = []
    train_accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []
    learning_rates = []  

    epoch_iter = range(num_epochs)
    if verbose == 'epoch':
        epoch_iter = tqdm(epoch_iter, desc="Training epochs", unit='epoch')

    for epoch in epoch_iter:
        model.train()
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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_loss_values.append(epoch_train_loss)
        train_accuracy_values.append(epoch_train_acc)

        # Optional validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_loss_values.append(epoch_val_loss)
            val_accuracy_values.append(epoch_val_acc)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            elif scheduler is not None:
                scheduler.step()

        else:
            # Still step scheduler if no val set and using epoch-based scheduler
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        if verbose is None:
            print_str = f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f}"
            if val_loader is not None:
                print_str += f" | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}"
            print(print_str)

    return train_loss_values, train_accuracy_values, val_loss_values, val_accuracy_values, learning_rates