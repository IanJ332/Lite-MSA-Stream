import asyncio
import logging
import json
import os
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Services
from app.services.vad_iterator import VADIterator
from app.services.sensevoice_service import SenseVoiceService
from app.services.text_service import TextService
from app.services.fusion_engine import FusionEngine

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

# --- Inference Worker (Consumer) ---
async def inference_worker(queue: asyncio.Queue, websocket: WebSocket, fusion_engine: FusionEngine, loop):
    """
    Background task to process speech segments from the queue.
    This ensures the WebSocket loop is never blocked by inference.
    """
    logger.info("Inference Worker Started")
    beam_size = 1 # Default
    
    try:
        while True:
            # Get segment from queue
            item = await queue.get()
            
            # Check for Config Update (Special Item)
            if isinstance(item, dict) and item.get("type") == "config":
                beam_size = item.get("beam_size", 1)
                queue.task_done()
                continue
                
            # Process Speech Segment
            speech_segment = item
            
            try:
                logger.info(f"Worker Processing Segment: {len(speech_segment)} bytes")
                
                # Parallel Inference
                segment_frozen = bytes(speech_segment)
                
                asr_future = loop.run_in_executor(
                    None, lambda: sensevoice_service.predict(segment_frozen, beam_size=beam_size)
                )
                emotion_future = loop.run_in_executor(
                    None, lambda: ensemble_service.predict_emotions(segment_frozen)
                )
                
                asr_result, emotion_probs = await asyncio.gather(asr_future, emotion_future)
                
                # Text Emotion Analysis
                text_content = asr_result.get("text", "")
                text_emotions = {}
                if text_content:
                        text_emotions = text_service.analyze_sentiment(text_content)

                # Fusion Engine (Entropy Fusion + Smoothing)
                final_emotions = fusion_engine.fuse(asr_result, emotion_probs, text_emotions)
                
                # STRICT Filtering (Hallucination Blocklist)
                if final_emotions is None:
                    logger.info(f"Discarded Noise/Hallucination: '{text_content}'")
                    queue.task_done()
                    continue

                # Find Top Emotion
                top_emotion = max(final_emotions, key=final_emotions.get)
                final_confidence = final_emotions[top_emotion]
                
                # Extra Safety: Discard low confidence short text
                if len(text_content) < 5 and final_confidence < 0.4:
                        logger.info(f"Discarded Low Confidence Short Text: '{text_content}' ({final_confidence:.2f})")
                        queue.task_done()
                        continue
                
                logger.info(f"Result: [{top_emotion.upper()}] {text_content}")

                # Response
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "result",
                    "transcription": text_content,
                    "sentiment": top_emotion,
                    "confidence": final_confidence,
                    "emotions": final_emotions,
                    "details": {
                        "processing_time": asr_result.get("processing_time", 0),
                        "raw_output": asr_result.get("raw_output", ""),
                        "beam_size": beam_size,
                        "text_emotions": text_emotions
                    }
                }
                data_logger.info(json.dumps(response))
                
                # Send back to client
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(response)
                    
            except Exception as e:
                logger.error(f"Inference Worker Error: {e}")
            
            finally:
                queue.task_done()
                
    except asyncio.CancelledError:
        logger.info("Inference Worker Cancelled")

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    # Services
    vad_iterator = VADIterator()
    fusion_engine = FusionEngine()
    loop = asyncio.get_running_loop()
    
    # Async Queue for Producer-Consumer
    queue = asyncio.Queue()
    
    # Start Worker Task
    worker_task = asyncio.create_task(
        inference_worker(queue, websocket, fusion_engine, loop)
    )
    
    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break

            # Receive Message (Text or Bytes)
            # This loop must remain FAST to handle VAD updates
            try:
                message = await websocket.receive()
            except RuntimeError:
                # Handle "Cannot call 'receive' once a disconnect message has been received"
                break
            
            if message["type"] == "websocket.disconnect":
                break

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
                            
                            # Update Worker Config via Queue
                            queue.put_nowait({"type": "config", "beam_size": beam_size})
                            
                except Exception as e:
                    logger.error(f"Failed to parse config message: {e}")
                continue

            if "bytes" in message:
                chunk = message["bytes"]
                
                # 1. VAD Processing (Fast, Synchronous-ish)
                speech_segment, speech_prob = vad_iterator.process(chunk)
                
                # Send VAD Update (Visual Feedback) - Immediate
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "type": "vad_update",
                        "prob": speech_prob,
                        "triggered": vad_iterator.triggered
                    })
                
                # 2. Producer: Push to Queue if Segment Ready
                if speech_segment:
                    logger.info("Endpoint Detected -> Queuing for Inference")
                    queue.put_nowait(speech_segment)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket loop: {e}")
    finally:
        # Cleanup
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except RuntimeError:
            pass

# Mount Frontend (Must be last to avoid overriding API routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0:8000 to match README and standard uvicorn port
    uvicorn.run(app, host="0.0.0.0", port=8000)
