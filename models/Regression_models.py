import torch
import torch.nn as nn
from tqdm import tqdm


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