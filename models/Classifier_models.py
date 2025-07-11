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

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, verbose=False):
    loss_values = []
    accuracy_values = []
    model.train()

    if verbose:
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            total_steps = len(train_loader)

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Update loss
                    running_loss += loss.item()

                    # Compute accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    pbar.set_postfix(loss=running_loss / (i + 1))

            epoch_loss = running_loss / total_steps
            epoch_accuracy = correct / total
            loss_values.append(epoch_loss)
            accuracy_values.append(epoch_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2%}")
    
    else:
        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            running_loss = 0.0
            total_steps = len(train_loader)

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / total_steps
            loss_values.append(epoch_loss)
            # Optionally compute accuracy here too if you like

    return loss_values, accuracy_values