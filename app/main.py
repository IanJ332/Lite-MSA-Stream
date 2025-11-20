import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vox-pathos")

app = FastAPI(title="Vox-Pathos: Real-time Multimodal Sentiment Analysis")

@app.get("/health")
async def health_check():
    """
    Cloud Run 健康检查端点 [cite: 199]
    只有当模型加载完毕（未来阶段实现）才返回 200
    """
    return {"status": "healthy", "service": "Vox-Pathos"}

from app.services.vad_iterator import VADIterator
from app.services.acoustic_analyzer import AcousticAnalyzer

# Global Analyzer Instance
acoustic_analyzer = None

@app.on_event("startup")
async def startup_event():
    global acoustic_analyzer
    logger.info("Initializing Acoustic Analyzer...")
    acoustic_analyzer = AcousticAnalyzer()
    logger.info("Acoustic Analyzer initialized.")

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    """
    处理实时音频流的 WebSocket 端点 
    """
    await websocket.accept()
    logger.info("Client connected")
    
    # Per-connection VAD state
    vad_iterator = VADIterator()
    loop = asyncio.get_running_loop()
    
    try:
        while True:
            # 接收二进制音频数据 (Int16 PCM) [cite: 15]
            audio_chunk = await websocket.receive_bytes()
            
            # 1. VAD Processing
            speech_segment = vad_iterator.process(audio_chunk)
            
            if speech_segment:
                logger.info(f"Detected speech segment: {len(speech_segment)} bytes")
                
                # 2. Acoustic Analysis (Blocking -> Async)
                # Run inference in thread pool to avoid blocking the event loop
                emotion_result = await loop.run_in_executor(
                    None, 
                    acoustic_analyzer.predict, 
                    speech_segment
                )
                
                # 3. Send Feedback
                await websocket.send_json({
                    "type": "acoustic_result",
                    "emotions": emotion_result
                })
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket loop: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
