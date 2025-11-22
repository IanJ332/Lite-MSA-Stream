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
from app.services.acoustic_analyzer import AcousticAnalyzer
from app.services.transcription_service import TranscriptionService
from app.services.text_sentiment_analyzer import TextSentimentAnalyzer

# Configure Logging
if not os.path.exists("logs"):
    os.makedirs("logs")

# Session Log File (One per server start for simplicity, or per connection?)
# Let's do per-server-start for now, or we can log to a rotating file.
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
    allow_origins=["*"], # Allow all for local dev convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Service Instances
acoustic_analyzer = None
transcription_service = None
text_analyzer = None

@app.on_event("startup")
async def startup_event():
    global acoustic_analyzer, transcription_service, text_analyzer
    
    logger.info("--- Starting Vox-Pathos Services ---")
    
    # 1. Acoustic Analyzer
    logger.info("Initializing Acoustic Analyzer...")
    acoustic_analyzer = AcousticAnalyzer()
    
    # 2. Transcription Service (ASR)
    logger.info("Initializing Transcription Service (Faster-Whisper)...")
    transcription_service = TranscriptionService(model_size="base.en", device="cpu", compute_type="int8")
    
    # 3. Text Sentiment Analyzer (NLP)
    logger.info("Initializing Text Sentiment Analyzer (DistilBERT)...")
    text_analyzer = TextSentimentAnalyzer()
    
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
    
    # Fusion Hyperparameter
    ALPHA = 0.7 
    
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
                
                # 2. Parallel Inference
                acoustic_future = loop.run_in_executor(
                    None, acoustic_analyzer.predict, speech_segment
                )
                asr_future = loop.run_in_executor(
                    None, transcription_service.transcribe, speech_segment
                )
                
                acoustic_result, transcript = await asyncio.gather(acoustic_future, asr_future)
                
                # 3. Text Sentiment
                text_result = None
                if transcript:
                    text_result = await loop.run_in_executor(
                        None, text_analyzer.predict, transcript
                    )
                
                # 4. Fusion Logic
                final_sentiment = acoustic_result["sentiment"]
                final_confidence = acoustic_result["confidence"]
                fusion_details = {}
                
                if text_result and "scores" in text_result and "scores" in acoustic_result:
                    fused_scores = {}
                    for label in ["positive", "negative", "neutral"]:
                        text_p = text_result["scores"].get(label, 0.0)
                        audio_p = acoustic_result["scores"].get(label, 0.0)
                        fused_scores[label] = (ALPHA * text_p) + ((1 - ALPHA) * audio_p)
                    
                    winner = max(fused_scores, key=fused_scores.get)
                    final_sentiment = winner
                    final_confidence = fused_scores[winner]
                    fusion_details = fused_scores
                
                # 5. Response & Logging
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "result",
                    "transcription": transcript,
                    "sentiment": final_sentiment,
                    "confidence": round(final_confidence, 4),
                    "details": {
                        "acoustic": acoustic_result,
                        "text": text_result,
                        "fusion": fusion_details
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
    # Bind to 0.0.0.0:8080 as requested for the "web page port"
    uvicorn.run(app, host="0.0.0.0", port=8080)
