import asyncio
import websockets
import time
import struct
import numpy as np
import json

# --- Configuration ---
URI = "ws://localhost:8000/ws/analyze"
SAMPLE_RATE = 16000
# Silero VAD requires specific chunk sizes: 512, 1024, 1536 samples for 16kHz
# 512 samples = 32ms
CHUNK_SIZE = 512 
CHUNK_SIZE_MS = (CHUNK_SIZE / SAMPLE_RATE) * 1000

# --- Audio Generation ---
def generate_audio_chunk(duration_ms: float, is_speech: bool) -> bytes:
    """
    Generate PCM 16-bit audio bytes.
    """
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    
    if is_speech:
        # Simulate "Speech" using high-energy White Noise
        # Silero VAD often ignores pure sine waves. White noise has energy across all frequencies.
        # Amplitude 0.8 (near max)
        waveform = np.random.uniform(-0.8, 0.8, num_samples).astype(np.float32)
        waveform = (waveform * 32767).astype(np.int16)
    else:
        # Simulate silence
        waveform = np.zeros(num_samples, dtype=np.int16)
        
    return waveform.tobytes()

async def audio_stream_client():
    """
    Simulate client sending audio stream to verify VAD segmentation and inference.
    """
    print("--- Phase 3 WebSocket Client Test ---")
    print(f"Chunk Size: {CHUNK_SIZE} samples ({CHUNK_SIZE_MS} ms)")
    
    try:
        async with websockets.connect(URI) as websocket:
            print(f"Connected to: {URI}")
            
            # 1. Send Speech (2.5 seconds)
            # 2.5s / 0.032s = ~78 chunks
            num_chunks_speech = int(2500 / CHUNK_SIZE_MS)
            print(f"Sending 2.5s of 'speech' ({num_chunks_speech} chunks)...")
            
            for _ in range(num_chunks_speech):
                chunk = generate_audio_chunk(CHUNK_SIZE_MS, is_speech=True)
                await websocket.send(chunk)
                # Sleep to simulate real-time streaming (slightly faster to avoid lag accumulation)
                await asyncio.sleep(CHUNK_SIZE_MS / 1000 * 0.9)

            # 2. Send Silence (1.5 seconds) - Trigger Endpointing
            # 1.5s / 0.032s = ~47 chunks
            num_chunks_silence = int(1500 / CHUNK_SIZE_MS)
            print(f"Sending 1.5s of 'silence' ({num_chunks_silence} chunks)...")
            
            for _ in range(num_chunks_silence):
                chunk = generate_audio_chunk(CHUNK_SIZE_MS, is_speech=False)
                await websocket.send(chunk)
                await asyncio.sleep(CHUNK_SIZE_MS / 1000 * 0.9)

            # 3. Listen for Result
            print("Listening for Analysis result...")
            
            try:
                # Wait for result
                result = await asyncio.wait_for(websocket.recv(), timeout=10) 
                print("\n\n---------------------------------")
                print("Received Result:")
                try:
                    data = json.loads(result)
                    print(json.dumps(data, indent=2))
                except:
                    print(result)
                print("---------------------------------")
                
            except asyncio.TimeoutError:
                print("Error: No result received within 10s.")
                
            await websocket.close()

    except ConnectionRefusedError:
        print("Connection Failed. Ensure Uvicorn is running on http://localhost:8000.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(audio_stream_client())