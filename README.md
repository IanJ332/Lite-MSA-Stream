# Vox-Pathos: Real-time Multimodal Sentiment Analysis

**Vox-Pathos** is a lightweight, real-time multimodal sentiment analysis microservice designed for efficient CPU-only inference. It processes streaming audio to infer sentiment using a combination of **Voice Activity Detection (VAD)** and **Acoustic Analysis**.

## 🚀 Features

*   **Real-time Streaming**: WebSocket endpoint (`/ws/analyze`) for continuous audio ingestion.
*   **Voice Activity Detection (VAD)**:
    *   Powered by `onnx-community/silero-vad`.
    *   Automatically segments speech from silence.
    *   Robust against background noise (tuned threshold).
*   **Acoustic Sentiment Analysis**:
    *   Powered by `Xenova/ast-finetuned-speech-commands-v2` (Audio Spectrogram Transformer).
    *   Analyzes speech segments for acoustic features.
    *   Optimized ONNX Runtime execution (CPU).
*   **Dockerized**: Optimized multi-stage build for easy deployment.

## 🏗️ Architecture

```mermaid
graph LR
    Client[Client (WebSocket)] -->|Audio Stream (PCM)| Server[FastAPI Server]
    Server -->|Chunk| Buffer[Audio Buffer]
    Buffer -->|Frame| VAD[Silero VAD]
    VAD -->|Speech Segment| Acoustic[Acoustic Analyzer (AST)]
    Acoustic -->|Sentiment Score| Server
    Server -->|JSON Result| Client
```

## 📂 Project Structure

```
Vox-Pathos/
├── app/
│   ├── main.py           # FastAPI entry point & WebSocket route
│   ├── services/
│   │   ├── vad_iterator.py       # VAD logic (Silero)
│   │   ├── acoustic_analyzer.py  # Acoustic inference (AST)
│   │   └── audio_buffer.py       # Circular buffer
│   └── utils/
│       └── model_utils.py        # Auto-download models from HuggingFace
├── models/               # Cached ONNX models (auto-downloaded)
├── test_client.py        # Verification script
├── Dockerfile            # CPU-optimized Docker build
├── requirements.txt      # Python dependencies
└── README.md
```

## 🛠️ Getting Started

### Prerequisites
*   **Python 3.10+**
*   **Docker** (optional, for containerization)

### Local Setup

1.  **Clone and Setup Environment**:
    ```bash
    git clone <repo-url>
    cd Vox-Pathos
    python -m venv venv
    # Windows
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *On first run, it will automatically download the necessary ONNX models (~40MB) to the `models/` directory.*

### Usage

**WebSocket Endpoint**: `ws://localhost:8000/ws/analyze`

**Input Format**:
*   **Audio**: 16-bit PCM, Mono, 16kHz.
*   **Chunk Size**: 512 samples (32ms) recommended for VAD stability.

**Output Format** (JSON):
```json
{
  "sentiment": "neutral",
  "confidence": 0.95,
  "raw_class_id": 12
}
```

### Verification
docker run -p 8080:8080 vox-pathos
```
Service available at `http://localhost:8080`.

## 📜 License
MIT
