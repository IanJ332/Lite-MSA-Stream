import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import glob

# --- Configuration ---
SAMPLE_RATE = 16000
N_MFCC = 40
TIME_STEPS = 300 # 3 seconds * 100 frames/sec (approx)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 3 # Positive, Negative, Neutral

# --- 1. Model Architecture (1D-CNN) ---
class AcousticCNN(nn.Module):
    def __init__(self, n_mfcc=N_MFCC, num_classes=NUM_CLASSES):
        super(AcousticCNN, self).__init__()
        
        # Input Shape: (Batch, N_MFCC, Time_Steps)
        
        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=n_mfcc, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense Layer
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.global_pool(x)
        x = x.squeeze(2)
        
        x = self.fc(x)
        return x

# --- 2. Datasets ---

class RAVDESSDataset(Dataset):
    def __init__(self, root_dir):
        self.files = glob.glob(os.path.join(root_dir, "**/*.wav"), recursive=True)
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64, "center": False}
        )
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath = self.files[idx]
        waveform, sr = torchaudio.load(filepath)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Extract MFCC
        mfcc = self.transform(waveform) # (1, N_MFCC, Time)
        mfcc = mfcc.squeeze(0)          # (N_MFCC, Time)
        
        # Pad/Crop to fixed time steps
        if mfcc.shape[1] < TIME_STEPS:
            pad_amount = TIME_STEPS - mfcc.shape[1]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))
        else:
            mfcc = mfcc[:, :TIME_STEPS]
            
        # Parse Label from filename (03-01-03-01-01-01-01.wav)
        # Emotion is 3rd part: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        filename = os.path.basename(filepath)
        parts = filename.split("-")
        emotion_code = int(parts[2])
        
        # Map to 3 classes: 0=Neutral, 1=Positive, 2=Negative
        if emotion_code in [1, 2]: # Neutral, Calm
            label = 0
        elif emotion_code in [3, 8]: # Happy, Surprised
            label = 1
        else: # Sad, Angry, Fearful, Disgust
            label = 2
            
        return mfcc, label

class DummyAudioDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random MFCC-like tensor
        mfcc = torch.randn(N_MFCC, TIME_STEPS)
        label = torch.randint(0, NUM_CLASSES, (1,)).item()
        return mfcc, label

# --- 3. Training Loop ---
def train():
    print("Initializing AcousticCNN Model...")
    model = AcousticCNN()
    
    # Check for data
    data_dir = "data"
    if os.path.exists(data_dir) and len(glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)) > 0:
        print(f"Found data in {data_dir}. Using RAVDESSDataset.")
        dataset = RAVDESSDataset(data_dir)
    else:
        print(f"Data directory '{data_dir}' not found or empty. Using DummyAudioDataset.")
        dataset = DummyAudioDataset(size=100)
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")
        
    print("\nTraining Complete.")
    
    # --- 4. Export to ONNX ---
    print("Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, N_MFCC, TIME_STEPS)
    
    os.makedirs("models/custom", exist_ok=True)
    output_path = "models/custom/acoustic_cnn.onnx"
    
    # Use legacy exporter for stability
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        input_names=["input_mfcc"], 
        output_names=["output_sentiment"],
        dynamic_axes={
            "input_mfcc": {0: "batch_size"},
            "output_sentiment": {0: "batch_size"}
        },
        dynamo=False
    )
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train()
