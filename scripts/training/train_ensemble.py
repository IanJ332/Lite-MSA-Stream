import sys
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from app.models.fusion_layer import FusionNet

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_speaker_split(speakers):
    """
    Returns train_indices and val_indices based on Speaker ID.
    Validation:
    - CREMA-D: 1082 - 1091 (Last 10)
    - RAVDESS: 21 - 24 (Last 4)
    - TESS: None (All Train)
    """
    train_indices = []
    val_indices = []
    
    for idx, spk in enumerate(speakers):
        # spk is bytes in H5, decode if needed
        if isinstance(spk, bytes):
            spk = spk.decode('utf-8')
            
        is_val = False
        
        if "CREMA_" in spk:
            # CREMA_1001
            spk_num = int(spk.split("_")[1])
            if spk_num >= 1082:
                is_val = True
        elif "RAVDESS_" in spk:
            # RAVDESS_01
            spk_num = int(spk.split("_")[1])
            if spk_num >= 21:
                is_val = True
        
        if is_val:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
            
    return train_indices, val_indices

def train_ensemble():
    # 1. Load Data
    data_file = "all_features.h5"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Run prepare_data.py first.")
        return

    print(f"Loading data from {data_file}...")
    with h5py.File(data_file, "r") as f:
        features = f["features"][:]
        labels = f["labels"][:]
        speakers = f["speakers"][:]

    print(f"Total Samples: {len(features)}")
    
    # 2. Speaker-Independent Split
    print("Performing Speaker-Independent Split...")
    train_idx, val_idx = get_speaker_split(speakers)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_val = features[val_idx]
    y_val = labels[val_idx]
    
    # 3. Class Balancing (WeightedRandomSampler)
    print("Calculating Class Weights...")
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # 4. DataLoaders
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler) # Use Sampler!
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 5. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Update Output Dim to 6 (Neu, Hap, Sad, Ang, Fea, Dis)
    model = FusionNet(input_dim=2048, hidden_dim=1024, output_dim=6, dropout_rate=0.4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # Lower LR for fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 6. Training Loop
    best_val_acc = 0.0
    EPOCHS = 50 # Enough for convergence with pre-extracted features
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "fusion_weights.pth")
            print("--> Best Model Saved")

    print(f"Training Complete. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_ensemble()
