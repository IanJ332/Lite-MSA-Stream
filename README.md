*   **RAM**: 8GB minimum (16GB recommended for training).
*   **CPU**: Modern Multi-core CPU (AVX2 support recommended).

### 2. Installation

1.  **Clone/Copy the Repository**:
    Ensure all project files are present on the new machine.

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install PyTorch, Transformers, FunASR, and other necessary libraries.*

---

### 3. Data Preparation (Step 1)

Before you can run the app with high accuracy, you must prepare the data. This step extracts features from the **RAVDESS** dataset using the heavy backbone models (HuBERT/Wav2Vec2) and saves them to a database.

1.  **Ensure Data Exists**:
    Make sure the `ravdess_data/` folder is in the project root and contains the `.wav` files.

2.  **Run the Preparation Script**:
    ```bash
    python scripts/prepare_data.py
    ```

3.  **Expected Output**:
    *   You will see a progress bar: `Extracting features (Full Dataset)... 100%|██████| ...`
    *   **Time**: This process takes **1-2 hours** on a CPU (depending on speed).
    *   **Success Indicator**: A file named `features.h5` (~2-3 GB) will be created in the project root.
    *   **Terminal Message**: `Data preparation complete.`

---

### 4. Training the Model (Step 2)

Now that features are extracted, you will train the **Fusion Layer** (the "brain" that combines the models).

1.  **Run the Training Script**:
    ```bash
    python train_ensemble.py
    ```

2.  **Expected Output**:
    *   You will see training logs for each epoch:
        ```text
        Epoch 1/50 - Loss: 1.8421 - Train Acc: 45.20% - Test Acc: 52.10%
        ...
        Epoch 50/50 - Loss: 0.3102 - Train Acc: 92.50% - Test Acc: 88.40%
        ```
    *   **Time**: This is fast! It should take **5-10 minutes**.
    *   **Success Indicator**: A file named `fusion_weights.pth` will be created.
    *   **Terminal Message**: `Training complete. Best Test Accuracy: XX.XX%`

---

### 5. Running the Application (Step 3)

Once `fusion_weights.pth` exists, the application automatically loads it to provide high-accuracy, 8-emotion detection.

1.  **Start the Server**:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

2.  **Access the UI**:
    *   Open your browser and go to: `http://localhost:8000`
    *   Click **Start Recording**.
    *   Adjust the **Model Accuracy** slider to 100% for best results (uses more CPU).

---

### 6. Troubleshooting

*   **`ModuleNotFoundError`**: Run `pip install -r requirements.txt` again.
*   **`fusion_weights.pth not found`**: The app will still run, but it will fall back to the basic SenseVoice model (lower accuracy, fewer emotions). Run Step 4 (Training) to fix this.
*   **High CPU Usage**: This is normal! The Ensemble model runs 3 deep neural networks in parallel to achieve high accuracy. Use the slider in the UI to lower CPU usage if needed.

---

## 🐳 Docker Support (Intel Mac / Linux / Windows)

If you are using an **Intel MacBook** (where PyTorch versions might be tricky) or just want a consistent environment, use Docker.

### 1. Build the Image
```bash
docker build -t vox-pathos .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 vox-pathos
```
*   Access the UI at `http://localhost:8000`.

### 3. Training in Docker (Optional)
If you need to run the training inside Docker (e.g., to generate `fusion_weights.pth`), mount your data volume:
```bash
# Mount current directory to /app so the script can see 'ravdess_data' and save 'fusion_weights.pth' back to your host
docker run -v $(pwd):/app vox-pathos python scripts/prepare_data.py
docker run -v $(pwd):/app vox-pathos python train_ensemble.py
```
*(On Windows PowerShell, replace `$(pwd)` with `${PWD}`)*

---

## ☁️ Cloud Training (Google Colab)

If your local machine is slow or you want to use a free GPU, use the provided Colab notebook.

1.  **Zip your Data**: Zip the `ravdess_data` folder to `ravdess_data.zip`.
2.  **Upload to Drive**: Upload `ravdess_data.zip` to your Google Drive.
3.  **Open Notebook**: Upload `Lite_MSA_Colab_Training.ipynb` to [Google Colab](https://colab.research.google.com/).
4.  **Run All Cells**: The notebook will:
    *   Mount your Drive.
    *   Clone the code.
    *   Install dependencies.
    *   Run Data Prep (Augmentation) & Training.
    *   Save `fusion_weights.pth` back to your Drive.
5.  **Download Weights**: Download `fusion_weights.pth` from Drive to your local project folder and run the app!
