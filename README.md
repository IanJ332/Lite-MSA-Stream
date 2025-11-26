*   **RAM**: 8GB minimum (16GB recommended for training).
*   **CPU**: Modern Multi-core CPU (AVX2 support recommended).

*   **RAM**: 8GB minimum (16GB recommended for training).
*   **CPU**: Modern Multi-core CPU (AVX2 support recommended).

### 2. Installation

#### Option A: Docker (Recommended)
The easiest way to run the application is using Docker. This ensures all dependencies (including system libraries like ffmpeg) are installed correctly.

1.  **Build the Image**:
    ```bash
    docker build -t vox-pathos .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 8000:8000 vox-pathos
    ```
    *   Access the UI at `http://localhost:8000`.

#### Option B: Manual Installation

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

### 3. Project Structure

The project is organized as follows:

*   `app/`: Main application code (API, Services, Models).
*   `frontend/`: Web UI (HTML, JS, CSS).
*   `scripts/`: Utility and training scripts.
    *   `training/`: Scripts for data preparation and model training (`prepare_data.py`, `train_ensemble.py`).
    *   `testing/`: Scripts for benchmarking and validation.
    *   `utils/`: Helper scripts (downloaders, setup).
*   `docs/`: Documentation and reports.
*   `outputs/`: Logs and training outputs.
*   `models/`: Directory for storing model weights.

---

### 4. Data Preparation & Training (Optional)

If you want to retrain the model or run it from scratch (without pre-trained weights):

1.  **Prepare Data**:
    Ensure `ravdess_data/` is in the project root.
    ```bash
    python scripts/training/prepare_data.py
    ```
    *   This creates `all_features.h5` in the root.

2.  **Train Model**:
    ```bash
    python scripts/training/train_ensemble.py
    ```
    *   This creates `fusion_weights.pth` in the root.

---

### 5. Running the Application (Manual)

1.  **Start the Server**:
    ```bash
    # Windows
    start_app.bat
    # Or manually:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

2.  **Access the UI**:
    *   Open `http://localhost:8000`

---

### 6. Troubleshooting

*   **`ModuleNotFoundError`**: Ensure you are in the virtual environment and have installed requirements.
*   **`fusion_weights.pth not found`**: The app will run in "Acoustic Only" mode or fallback mode. Train the model to generate weights.
*   **Docker Audio Issues**: Docker containers might need extra flags to access host audio devices (e.g., `--device /dev/snd` on Linux), but since this is a web app, the browser handles microphone input, so standard Docker run works fine!

---

## ☁️ Cloud Training (Google Colab)

See `docs/Lite_MSA_Colab_Training.ipynb` for instructions on training in the cloud.
