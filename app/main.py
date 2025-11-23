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
ensemble_service = None

@app.on_event("startup")
async def startup_event():
    global sensevoice_service, ensemble_service
    
    logger.info("--- Starting Vox-Pathos Services ---")
    
    # Initialize Unified SenseVoice Service
    logger.info("Initializing SenseVoice Service (ASR)...")
    sensevoice_service = SenseVoiceService(device="cpu")
    
    # Initialize Ensemble SER Service
    logger.info("Initializing Ensemble SER Service (Emotion)...")
    from app.services.ensemble_service import EnsembleService
    ensemble_service = EnsembleService(device="cpu")
    
    # Load Fusion Weights if available
    if os.path.exists("fusion_weights.pth"):
        ensemble_service.load_fusion_model("fusion_weights.pth")
    else:
        logger.warning("fusion_weights.pth not found. Ensemble SER will be disabled.")
    
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
    
    # Config State
    beam_size = 1
    
    try:
        while True:
            # Receive Message (Text or Bytes)
            message = await websocket.receive()
            
            if "text" in message:
                # Handle Config Message
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        if "vad_threshold" in data:
                            # Slider 0-100% -> Threshold 1.0-0.0
                            # Also map to Beam Size: 0-100% -> 1-10
                            # Higher Sensitivity (lower threshold) = Higher Beam Size (more CPU)
                            
                            threshold = float(data["vad_threshold"])
                            vad_iterator.set_threshold(threshold)
                            
                            # Map threshold (1.0 -> 0.0) to beam_size (1 -> 10)
                            # Sensitivity = 1.0 - threshold
                            sensitivity = 1.0 - threshold
                            beam_size = max(1, int(sensitivity * 10))
                            logger.info(f"Updated Config: Threshold={threshold:.2f}, Beam Size={beam_size}")
                            
                except Exception as e:
                    logger.error(f"Failed to parse config message: {e}")
                continue

            if "bytes" in message:
                audio_chunk = message["bytes"]
            else:
                continue
            
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
                
                # 2. Parallel Inference (SenseVoice + Ensemble)
                # Run in executor to avoid blocking the event loop
                
                # ASR (SenseVoice)
                asr_future = loop.run_in_executor(
                    None, lambda: sensevoice_service.predict(speech_segment, beam_size=beam_size)
                )
                
                # Emotion (Ensemble)
                emotion_future = loop.run_in_executor(
                    None, lambda: ensemble_service.predict_emotions(speech_segment)
                )
                
                # Wait for both
                asr_result, emotion_probs = await asyncio.gather(asr_future, emotion_future)
                
                # Merge Results
                final_sentiment = asr_result["sentiment"]
                final_confidence = asr_result["confidence"]
                
                # If Ensemble returned valid probabilities, use them
                if emotion_probs:
                    # Find max probability emotion
                    top_emotion = max(emotion_probs, key=emotion_probs.get)
                    top_prob = emotion_probs[top_emotion]
                    
                    final_sentiment = top_emotion
                    final_confidence = top_prob
                    
                    logger.info(f"Ensemble Emotion: {top_emotion} ({top_prob:.2f})")
                
                # 3. Response & Logging
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "result",
                    "transcription": asr_result["text"],
                    "sentiment": final_sentiment,
                    "confidence": final_confidence,
                    "emotions": emotion_probs, # Full 8-emotion distribution
                    "details": {
                        "processing_time": asr_result.get("processing_time", 0),
                        "raw_output": asr_result.get("raw_output", ""),
                        "beam_size": beam_size
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
