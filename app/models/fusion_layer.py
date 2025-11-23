import torch
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=8, dropout=0.3):
        """
        Linear Fusion Layer for Ensemble SER.
        Args:
            input_dim (int): Dimension of concatenated features (1024+1024=2048).
            hidden_dim (int): Hidden layer size.
            output_dim (int): Number of emotion classes (8).
            dropout (float): Dropout probability.
        """
        super(FusionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
