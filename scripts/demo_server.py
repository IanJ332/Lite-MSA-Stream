import asyncio
import logging
import json
import random
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yayo-demo")

app = FastAPI(title="Yayo-MSA-Stream Demo")

# --- Simulation Data ---
SCENARIOS = [
    {
        "text": "I am absolutely furious about this service! It's unacceptable!",
        "emotion": "angry",
        "confidence": 0.95,
        "emotions": {"angry": 0.95, "sad": 0.02, "neutral": 0.01, "happy": 0.0, "fearful": 0.01, "disgusted": 0.01}
    },
    {
        "text": "Wow, this is actually amazing. I love how fast it works.",
        "emotion": "happy",
        "confidence": 0.92,
        "emotions": {"happy": 0.92, "neutral": 0.05, "sad": 0.0, "angry": 0.0, "fearful": 0.0, "disgusted": 0.03}
    },
    {
        "text": "I'm not sure what to do next. I feel a bit lost.",
        "emotion": "fearful",
        "confidence": 0.88,
        "emotions": {"fearful": 0.88, "sad": 0.08, "neutral": 0.04, "happy": 0.0, "angry": 0.0, "disgusted": 0.0}
    },
    {
        "text": "This is just okay. Nothing special.",
        "emotion": "neutral",
        "confidence": 0.85,
        "emotions": {"neutral": 0.85, "sad": 0.05, "happy": 0.05, "angry": 0.0, "fearful": 0.0, "disgusted": 0.05}
    },
    {
        "text": "Ugh, that smell is terrible. Get it away from me.",
        "emotion": "disgusted",
        "confidence": 0.90,
        "emotions": {"disgusted": 0.90, "angry": 0.05, "sad": 0.02, "neutral": 0.03, "happy": 0.0, "fearful": 0.0}
    },
    {
        "text": "I can't believe he's gone. It hurts so much.",
        "emotion": "sad",
        "confidence": 0.94,
        "emotions": {"sad": 0.94, "fearful": 0.03, "neutral": 0.02, "angry": 0.01, "happy": 0.0, "disgusted": 0.0}
    }
]

async def simulation_worker(websocket: WebSocket):
    """
    Simulates a conversation loop:
    1. Silence (Listening)
    2. Speech (VAD Active)
    3. Result (Analysis)
    """
    logger.info("Starting Simulation Worker")
    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break

            # Pick a random scenario
            scenario = random.choice(SCENARIOS)
            
            # Phase 1: Silence (1-2 seconds)
            logger.info("Simulating: Silence...")
            for _ in range(10):
                if websocket.client_state != WebSocketState.CONNECTED: return
                await websocket.send_json({
                    "type": "vad_update",
                    "prob": random.uniform(0.0, 0.1),
                    "triggered": False
                })
                await asyncio.sleep(0.1)

            # Phase 2: Speech (2-3 seconds)
            logger.info("Simulating: Speaking...")
            speech_duration = random.randint(20, 30)
            for _ in range(speech_duration):
                if websocket.client_state != WebSocketState.CONNECTED: return
                await websocket.send_json({
                    "type": "vad_update",
                    "prob": random.uniform(0.6, 0.99), # High probability
                    "triggered": True
                })
                await asyncio.sleep(0.1)

            # Phase 3: Result
            logger.info(f"Simulating: Result -> {scenario['emotion']}")
            response = {
                "timestamp": datetime.now().isoformat(),
                "type": "result",
                "transcription": scenario["text"],
                "sentiment": scenario["emotion"],
                "confidence": scenario["confidence"],
                "emotions": scenario["emotions"],
                "details": {
                    "processing_time": 0.05,
                    "raw_output": "SIMULATED",
                    "beam_size": 1,
                    "text_emotions": scenario["emotions"]
                }
            }
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(response)
            
            # Pause before next turn
            await asyncio.sleep(1.0)

    except Exception as e:
        logger.error(f"Simulation Error: {e}")

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to Demo Simulation")
    
    # Start the simulation loop
    task = asyncio.create_task(simulation_worker(websocket))
    
    try:
        while True:
            # Keep the connection open and drain any incoming audio (ignore it)
            await websocket.receive()
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        task.cancel()

# Mount Frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Run on port 8000 to match frontend config
    uvicorn.run(app, host="0.0.0.0", port=8000)
