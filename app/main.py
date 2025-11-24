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
from app.services.text_service import TextService

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
text_service = None

@app.on_event("startup")
async def startup_event():
    global sensevoice_service, ensemble_service, text_service
    
    logger.info("--- Starting Vox-Pathos Services ---")
    
    # Initialize Unified SenseVoice Service
    logger.info("Initializing SenseVoice Service (ASR)...")
    sensevoice_service = SenseVoiceService(device="cpu")
    
    # Initialize Ensemble SER Service
    logger.info("Initializing Ensemble SER Service (Emotion)...")
    from app.services.ensemble_service import EnsembleService
    ensemble_service = EnsembleService(device="cpu")

    # Initialize Text Emotion Service
    logger.info("Initializing Text Emotion Service (DistilRoberta)...")
    text_service = TextService()
    
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
    
    # Streaming Buffer
    # 16000Hz * 2 bytes (16-bit) = 32000 bytes per second
    # Window Size = 2 seconds (64000 bytes)
    # Step Size = 1 second (32000 bytes)
    WINDOW_SIZE = 64000
    STEP_SIZE = 32000
    audio_buffer = bytearray()
    
    # from textblob import TextBlob (Removed)
    
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
                            threshold = float(data["vad_threshold"])
                            vad_iterator.set_threshold(threshold)
                            sensitivity = 1.0 - threshold
                            beam_size = max(1, int(sensitivity * 10))
                            logger.info(f"Updated Config: Threshold={threshold:.2f}, Beam Size={beam_size}")
                except Exception as e:
                    logger.error(f"Failed to parse config message: {e}")
                continue

            if "bytes" in message:
                chunk = message["bytes"]
                audio_buffer.extend(chunk)
                
                # 1. VAD (Process Incoming Chunk Only)
                # We process the raw stream to update VAD state correctly
                # Chunk is likely small (e.g. 4096 bytes = 1024 samples)
                speech_segment, speech_prob = vad_iterator.process(chunk)
                
                # Send VAD Update
                await websocket.send_json({
                    "type": "vad_update",
                    "prob": speech_prob,
                    "triggered": vad_iterator.triggered
                })
            else:
                continue
            
            # Streaming Logic: Process only if buffer is full enough
            if len(audio_buffer) >= WINDOW_SIZE:
                # Extract Window
                window_bytes = audio_buffer[:WINDOW_SIZE]
                
                # Slide Buffer (Remove Step Size)
                audio_buffer = audio_buffer[STEP_SIZE:]
                
                # Only process if VAD triggered OR speech probability is high (on the latest chunk)
                if vad_iterator.triggered or speech_prob > 0.5:
                    logger.info(f"Processing Streaming Window: {len(window_bytes)} bytes")
                    
                    # 2. Parallel Inference
                    # Convert bytearray to bytes for SenseVoice
                    window_bytes_frozen = bytes(window_bytes)
                    
                    asr_future = loop.run_in_executor(
                        None, lambda: sensevoice_service.predict(window_bytes_frozen, beam_size=beam_size)
                    )
                    emotion_future = loop.run_in_executor(
                        None, lambda: ensemble_service.predict_emotions(window_bytes_frozen)
                    )
                    
                    asr_result, emotion_probs = await asyncio.gather(asr_future, emotion_future)
                    
                    final_sentiment = asr_result["sentiment"]
                    final_confidence = asr_result["confidence"]
                    text_content = asr_result["text"]
                    
                    # Filter Garbage / Short Hallucinations
                    # Ignore if empty, too short (< 2 chars), or just punctuation
                    # Also ignore common short hallucinations like "The.", "You.", "I."
                    hallucinations = ["The.", "You.", "I.", "It.", "A.", "And.", "But.", "So.", "He.", "She.", "They.", "We."]
                    if not text_content or len(text_content.strip()) < 2 or text_content.strip() in [".", ",", "?", "!", "。", "，", "？", "！"] or text_content.strip() in hallucinations:
                        logger.info(f"Ignored Garbage Output: '{text_content}'")
                        continue
                    text_emotions = {}
                    if text_content:
                        # Analyze Text Sentiment (DistilRoberta)
                        text_emotions = text_service.analyze_sentiment(text_content)
                        logger.info(f"Text Emotions: {text_emotions}")
                        
                        if emotion_probs and text_emotions:
                            # Fusion Strategy: Semantic Bias
                            for emotion, text_prob in text_emotions.items():
                                if emotion in emotion_probs:
                                    emotion_probs[emotion] = emotion_probs.get(emotion, 0) * (1.0 + text_prob * 2.0)
                            
                            # Acoustic Calibration
                            if text_emotions.get('neutral', 0) > 0.5:
                                emotion_probs['neutral'] = emotion_probs.get('neutral', 0) * 2.0
                                emotion_probs['calm'] = emotion_probs.get('calm', 0) * 1.5

                            # Normalize
                            total = sum(emotion_probs.values())
                            if total > 0:
                                for k in emotion_probs:
                                    emotion_probs[k] /= total
                                
                            # Find Top Emotion
                            top_emotion = max(emotion_probs, key=emotion_probs.get)
                            final_sentiment = top_emotion
                            final_confidence = emotion_probs[top_emotion]
                            
                            logger.info(f"Fused Emotion: {top_emotion} (Text Bias Applied)")

                    # 4. Response
                    response = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "result",
                        "transcription": text_content,
                        "sentiment": final_sentiment,
                        "confidence": final_confidence,
                        "emotions": emotion_probs,
                        "details": {
                            "processing_time": asr_result.get("processing_time", 0),
                            "raw_output": asr_result.get("raw_output", ""),
                            "beam_size": beam_size,
                            "text_emotions": text_emotions
                        }
                    }
                    data_logger.info(json.dumps(response))
                    await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket loop: {e}")
        try:
            await websocket.close()
        except RuntimeError:
            pass

# Mount Frontend (Must be last to avoid overriding API routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0:8000 to match README and standard uvicorn port
    uvicorn.run(app, host="0.0.0.0", port=8000)
