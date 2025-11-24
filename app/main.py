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
    
    # Streaming Buffer
    # 16000Hz * 2 bytes (16-bit) = 32000 bytes per second
    # Window Size = 2 seconds (64000 bytes)
    # Step Size = 1 second (32000 bytes)
    WINDOW_SIZE = 64000
    STEP_SIZE = 32000
    audio_buffer = bytearray()
    
    from textblob import TextBlob
    
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
            else:
                continue
            
            # Streaming Logic: Process only if buffer is full enough
            if len(audio_buffer) >= WINDOW_SIZE:
                # Extract Window
                window_bytes = audio_buffer[:WINDOW_SIZE]
                
                # Slide Buffer (Remove Step Size)
                # Keep the overlap for the next window
                audio_buffer = audio_buffer[STEP_SIZE:]
                
                # 1. VAD (Check if this window has speech)
                # We still use VAD to avoid processing silence
                speech_segment, speech_prob = vad_iterator.process(window_bytes)
                
                # Send VAD Update
                await websocket.send_json({
                    "type": "vad_update",
                    "prob": speech_prob,
                    "triggered": vad_iterator.triggered
                })
                
                # Only process if VAD triggered OR speech probability is high
                if vad_iterator.triggered or speech_prob > 0.5:
                    logger.info(f"Processing Streaming Window: {len(window_bytes)} bytes")
                    
                    # 2. Parallel Inference
                    asr_future = loop.run_in_executor(
                        None, lambda: sensevoice_service.predict(window_bytes, beam_size=beam_size)
                    )
                    emotion_future = loop.run_in_executor(
                        None, lambda: ensemble_service.predict_emotions(window_bytes)
                    )
                    
                    asr_result, emotion_probs = await asyncio.gather(asr_future, emotion_future)
                    
                    final_sentiment = asr_result["sentiment"]
                    final_confidence = asr_result["confidence"]
                    text_content = asr_result["text"]
                    
                        # 3. Text Fusion (Correct Acoustic Bias)
                    if text_content:
                        # A. Semantic Keyword Layer (Bag of Words)
                        # Explicitly catch strong emotional words
                        text_lower = text_content.lower()
                        
                        # Define Keywords
                        keywords = {
                            "angry": ["fuck", "shit", "damn", "hell", "stupid", "hate", "kill", "angry", "mad"],
                            "happy": ["happy", "great", "awesome", "love", "amazing", "good", "excited", "wow"],
                            "sad": ["sad", "cry", "sorry", "depressed", "unhappy", "bad", "grief"],
                            "fearful": ["scared", "fear", "afraid", "help", "danger"],
                            "disgusted": ["gross", "yuck", "disgusting", "nasty", "eww"]
                        }
                        
                        # Check for matches
                        keyword_boost = {}
                        for emotion, words in keywords.items():
                            for word in words:
                                if word in text_lower:
                                    keyword_boost[emotion] = keyword_boost.get(emotion, 0) + 1.0
                        
                        # B. TextBlob Polarity (General Sentiment)
                        blob = TextBlob(text_content)
                        polarity = blob.sentiment.polarity # -1.0 to 1.0
                        
                        if emotion_probs:
                            # Apply Keyword Boost (Strongest Signal)
                            for emotion, boost in keyword_boost.items():
                                # Massive boost for explicit keywords (e.g. "fuck" -> Angry)
                                emotion_probs[emotion] = emotion_probs.get(emotion, 0) * (2.0 + boost)
                                logger.info(f"Keyword Boost: {emotion} x{2.0+boost}")

                            # Apply General Polarity Bias
                            if polarity > 0.2: # Positive
                                emotion_probs['happy'] = emotion_probs.get('happy', 0) * (1.0 + polarity)
                                emotion_probs['surprised'] = emotion_probs.get('surprised', 0) * (1.0 + polarity)
                            elif polarity < -0.2: # Negative
                                emotion_probs['sad'] = emotion_probs.get('sad', 0) * (1.0 + abs(polarity))
                                emotion_probs['angry'] = emotion_probs.get('angry', 0) * (1.0 + abs(polarity))
                            
                            # Apply Acoustic Calibration (Neutral/Calm Boost)
                            # Only apply if NO strong keywords found
                            if not keyword_boost:
                                emotion_probs['neutral'] = emotion_probs.get('neutral', 0) * 3.0
                                emotion_probs['calm'] = emotion_probs.get('calm', 0) * 2.0
                            
                            # Normalize
                            total = sum(emotion_probs.values())
                            for k in emotion_probs:
                                emotion_probs[k] /= total
                                
                            # Find Top Emotion
                            top_emotion = max(emotion_probs, key=emotion_probs.get)
                            final_sentiment = top_emotion
                            final_confidence = emotion_probs[top_emotion]
                            
                            logger.info(f"Fused Emotion: {top_emotion} (Text Polarity: {polarity:.2f})")

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
                            "text_polarity": polarity if text_content else 0.0
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
