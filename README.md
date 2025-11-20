# Vox-Pathos: Real-time Multimodal Sentiment Analysis

Vox-Pathos is a lightweight, real-time multimodal sentiment analysis microservice designed for CPU-only environments (specifically Google Cloud Run). It processes real-time audio streams to infer sentiment using a combination of linguistic content (Text) and paralinguistic features (Audio/Tone).

## Phase 1: Infrastructure & Ingestion

This phase establishes the core infrastructure, including the FastAPI application skeleton, WebSocket ingestion endpoint, and a highly optimized Docker environment.

### Project Structure

```
Vox-Pathos/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI entry point & WebSocket route
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py     # Configuration management
│   └── services/
│       ├── __init__.py
│       └── audio_buffer.py # (Placeholder) Circular buffer for audio
├── Dockerfile            # Multi-stage build, optimized for CPU
├── requirements.txt      # Python dependencies
└── README.md
```

### Features (Phase 1)

*   **FastAPI Microservice**: High-performance async web framework.
*   **WebSocket Endpoint**: `/ws/analyze` for real-time binary audio streaming (Int16 PCM).
*   **Health Check**: `/health` endpoint for Cloud Run readiness probes.
*   **Docker Optimization**:
    *   Based on `python:3.10-slim`.
    *   Pre-installed CPU-only PyTorch (`torch`, `torchaudio`) to minimize image size.
    *   Multi-stage build to keep the runtime image clean.
    *   `libsndfile1` installed for audio processing.

### Getting Started

#### Prerequisites

*   Docker installed
*   Python 3.10+ (for local development)

#### Local Development

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    # Note: For local dev on Windows/Mac, you might get standard PyTorch.
    # The Dockerfile specifically installs the CPU-only Linux wheels.
    ```

2.  **Run the Server**:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

3.  **Test Health Check**:
    Open [http://localhost:8000/health](http://localhost:8000/health) in your browser.

4.  **Test WebSocket**:
    You can use a tool like Postman or a simple Python script to send binary audio data to `ws://localhost:8000/ws/analyze`.

#### Docker Build & Run

1.  **Build the Image**:
    ```bash
    docker build -t vox-pathos .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 8080:8080 vox-pathos
    ```
    The service will be available at `http://localhost:8080`.

### API Endpoints

*   `GET /health`: Returns `{"status": "healthy", "service": "Vox-Pathos"}`.
*   `WS /ws/analyze`: Accepts binary audio streams (16kHz, Mono, Int16 PCM). Returns JSON status updates.

### Next Steps (Phase 2)

*   Integrate Silero VAD for voice activity detection.
*   Integrate Faster-Whisper for ASR.
*   Integrate DistilBERT and Audio CNN for sentiment analysis.
