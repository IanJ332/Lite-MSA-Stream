import asyncio
import logging
import json
import os
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Services
from app.services.vad_iterator import VADIterator
from app.services.sensevoice_service import SenseVoiceService

# Configure Logging
if not os.path.exists("logs"):
    os.makedirs("logs")

# Session Log File
log_filename = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vox-pathos")

# File Handler for Data Logging
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(file_formatter)
data_logger = logging.getLogger("vox-pathos-data")
data_logger.addHandler(file_handler)
data_logger.setLevel(logging.INFO)

app = FastAPI(title="Vox-Pathos: Real-time Multimodal Sentiment Analysis")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Service Instances
sensevoice_service = None

@app.on_event("startup")
async def startup_event():
    global sensevoice_service
    
    logger.info("--- Starting Vox-Pathos Services ---")
    
    # Initialize Unified SenseVoice Service
    logger.info("Initializing SenseVoice Service (Unified ASR + Emotion)...")
    sensevoice_service = SenseVoiceService(device="cpu")
    
    logger.info("--- All Services Initialized ---")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Vox-Pathos"}

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    vad_iterator = VADIterator()
    loop = asyncio.get_running_loop()
    
    try:
        while True:
            # Receive Audio
            audio_chunk = await websocket.receive_bytes()
            
            # 1. VAD
            speech_segment, speech_prob = vad_iterator.process(audio_chunk)
            
            # Send VAD Probability for Real-time UI Feedback
            await websocket.send_json({
                "type": "vad_update",
                "prob": speech_prob,
                "triggered": vad_iterator.triggered
            })
            
            if speech_segment:
                logger.info(f"Processing speech segment: {len(speech_segment)} bytes")
                
                # 2. Unified Inference (SenseVoice)
                # Run in executor to avoid blocking the event loop
                result = await loop.run_in_executor(
                    None, sensevoice_service.predict, speech_segment
                )
                
                # 3. Response & Logging
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "result",
                    "transcription": result["text"],
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"],
                    "details": {
                        "processing_time": result.get("processing_time", 0),
                        "raw_output": result.get("raw_output", "")
                    }
                }
                
                # Log to file
                data_logger.info(json.dumps(response))
                
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket loop: {e}")
        await websocket.close()

# Mount Frontend (Must be last to avoid overriding API routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0:8000 to match README and standard uvicorn port
    uvicorn.run(app, host="0.0.0.0", port=8000)
