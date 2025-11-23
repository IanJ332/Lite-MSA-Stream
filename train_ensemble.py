import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from app.models.fusion_layer import FusionNet

def train_ensemble():
    # Config
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    DEVICE = "cpu" # Training is light, CPU is fine
    
    print("Loading data from features.h5...")
    try:
        with h5py.File("features.h5", "r") as f:
            features = torch.tensor(f["features"][:]).float()
            labels = torch.tensor(f["labels"][:]).long()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"Data shape: {features.shape}, Labels shape: {labels.shape}")
    
    # Dataset
    dataset = TensorDataset(features, labels)
    
    # Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = FusionNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        test_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "fusion_weights.pth")
            
    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
    print("Model saved to fusion_weights.pth")

if __name__ == "__main__":
    train_ensemble()
