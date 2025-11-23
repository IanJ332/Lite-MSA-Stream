# Lite-MSA-Stream: Real-time Multimodal Sentiment Analysis

**Lite-MSA-Stream** (formerly Vox-Pathos) is a lightweight, real-time multimodal sentiment analysis microservice. It processes streaming audio to infer sentiment using **SenseVoice** (Unified Speech Foundation Model) for simultaneous ASR (Speech-to-Text) and SER (Speech Emotion Recognition).

## 🚀 Features

*   **Real-time Streaming**: WebSocket endpoint (`/ws/analyze`) for continuous audio ingestion.
*   **Unified Architecture**:
    *   Powered by **SenseVoiceSmall**.
    *   Simultaneous ASR and Emotion Recognition.
    *   Robust against background noise and accents.
*   **Legacy Support** (Optional):
    *   Silero VAD + Whisper + DistilBERT + AST (Phase 1/2 architecture).
*   **Dockerized**: Optimized build for easy deployment.

## 📂 Project Structure

```
Lite-MSA-Stream/
├── app/
│   ├── main.py           # FastAPI entry point & WebSocket route
│   ├── services/
│   │   ├── sensevoice_service.py # Unified SenseVoice logic
│   │   └── vad_iterator.py       # VAD logic
│   └── utils/
├── frontend/             # Web Client
├── models/               # Cached models
├── Dockerfile            # Docker build
├── requirements.txt      # Python dependencies
└── README.md
```

## 🛠️ Getting Started

### Prerequisites
*   **Python 3.10+**
*   **FFmpeg** (Required for audio processing backend)
*   **Docker** (Optional)

### Local Setup

1.  **Clone and Setup Environment**:
    ```bash
    git clone <repo-url>
    cd Lite-MSA-Stream
    python -m venv venv
    # Windows
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Install FFmpeg** (Windows):
    ```powershell
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    # Restart your terminal after installation!
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Server**:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *On first run, it will automatically download the necessary models to the `models/` directory.*
    *Access the API at `http://localhost:8000`*

### Usage

**WebSocket Endpoint**: `ws://localhost:8000/ws/analyze`

**Input Format**:
*   **Audio**: Float32 PCM, Mono, 16kHz.
*   **Chunk Size**: 512 samples (32ms) recommended.

**Output Format** (JSON):
```json
{
  "text": "Hello world",
  "sentiment": "positive",
  "confidence": 0.95,
  "processing_time": 0.15
}
```

## 🐳 Docker

Build and run with Docker Compose:
```bash
docker-compose up --build
```

## 📜 License
MIT
