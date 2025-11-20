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

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    """
    处理实时音频流的 WebSocket 端点 
    """
    await websocket.accept()
    logger.info("Client connected")
    
    try:
        while True:
            # 接收二进制音频数据 (Int16 PCM) [cite: 15]
            # 客户端应发送 bytes, 避免 base64 开销 [cite: 16]
            audio_chunk = await websocket.receive_bytes()
            
            # TODO: 
            # 1. 放入 RingBuffer
            # 2. VAD 检测 (Stage 1)
            # 3. ASR & NLP 推理 (Stage 2 & 3)
            
            # 暂时简单回显数据长度，证明链路通畅
            await websocket.send_json({
                "status": "processing",
                "chunk_size": len(audio_chunk)
            })
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket loop: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
