import torch
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=6, dropout_rate=0.4):
        """
        Deep Fusion Layer for Ensemble SER.
        Args:
            input_dim (int): Dimension of concatenated features (1024+1024=2048).
            hidden_dim (int): Hidden layer size.
            output_dim (int): Number of emotion classes (8).
            dropout (float): Dropout probability.
        """
        super(FusionNet, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2 (Extra Layer)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3 (Output)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        return x
