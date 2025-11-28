# Real-Time Multimodal Sentiment Analysis System Blueprint: CPU-Based DistilBERT & MFCC Streaming Microservices Architecture Deep Dive

## 1. Executive Summary & Architecture Vision

This research report aims to construct a comprehensive engineering blueprint for developing and deploying a real-time multimodal sentiment analysis system. The core task of the system is to process real-time audio streams, extracting linguistic content (text) and paralinguistic features (acoustic features), and utilizing DistilBERT and Mel-frequency cepstral coefficients (MFCCs) respectively for emotion inference. Finally, these two modalities are fused at the decision level to generate comprehensive emotional insights. The key constraint of this project is that the deployment environment is limited to CPU architectures, which requires the system design to rigorously pursue computational efficiency, resource management, and latency optimization.

In the current deep learning ecosystem, GPUs are typically the preferred hardware accelerators for real-time inference. However, in edge computing, cost-sensitive cloud services (such as Google Cloud Run), or traditional enterprise servers, CPU inference still dominates. Achieving "real-time" processing in a CPU environment (i.e., processing speed faster than the audio stream playback speed, typically requiring a Real-Time Factor (RTF) < 1.0) faces enormous challenges. The compute-intensive nature of Automatic Speech Recognition (ASR) and Transformer models can easily lead to CPU bottlenecks, causing audio frame drops, WebSocket connection timeouts, or system crashes. Therefore, this report is not just a simple guide to model stacking, but a deep technical document on how to break through performance limits under constrained computing resources through algorithm optimization, model compression (quantization, distillation), and asynchronous architecture design.

The solution proposed in this blueprint adopts a "Late Fusion" strategy. The system architecture is based on Python's asyncio asynchronous framework and FastAPI, utilizing the WebSocket protocol for full-duplex communication. After the audio stream enters the system, it is split: one path converts it to text via a lightweight ASR engine (such as Faster-Whisper Int8), followed by text sentiment analysis using an ONNX Runtime-optimized DistilBERT model; the other path directly extracts MFCC features and inputs them into a lightweight Convolutional Neural Network (CNN) for tonal emotion classification. This decoupled design ensures that even if ASR processing experiences minor delays, the analysis of acoustic features can maintain a near real-time response, thereby enhancing system robustness. This report will cover everything from the mathematical principles of acoustic signal processing to the microscopic details of Docker containerized deployment, providing an irreplaceable implementation reference for the engineering team.

## 2. System Architecture Design & Microservice Patterns

To achieve low latency in a CPU environment, the system must abandon the traditional blocking I/O model and switch to an event-driven asynchronous stream processing architecture. This chapter will delve into the choice of microservice architecture, data flow design, and the decision rationale for the core technology stack.

### 2.1 Overall Architecture Pattern: Monolithic Microservice & Async I/O

In discussions of microservice architecture, there is usually a tendency to split different functional modules (such as ASR service, NLP service, audio analysis service) into independently deployed network services. However, under the specific constraints of this project (real-time, CPU limits), a traditional distributed microservice architecture could introduce unacceptable network latency and serialization/deserialization overhead [1]. ASR and NLP models typically need to be loaded into memory; splitting them into multiple containers not only increases cold start times but also leads to duplicate memory resource usage.

Therefore, this blueprint recommends a **Monolithic Microservice** architecture. That is, within a single independent Docker container, Python's asyncio event loop manages WebSocket connections, audio buffering, feature extraction, and model inference simultaneously. All models (ASR, DistilBERT, Audio-CNN) share the same process space (or multi-processing via shared memory), thereby eliminating network transmission overhead.

### 2.2 Data Flow & Pipeline Design

The system's data flow design follows the "Pipeline-Filter" pattern, aiming to maximize parallel processing capability and reduce blocking.

1.  **Ingestion Layer**:
    *   Clients (Web or Mobile) establish a long connection via WebSocket and send PCM audio data as binary streams.
    *   Protocol Optimization: Avoid using Base64 encoding to transmit audio, as Base64 decoding consumes extra CPU cycles and increases data transmission volume by about 33%. Direct binary Blob transmission is the best practice.
2.  **Filter Layer**:
    *   **Voice Activity Detection (VAD)**: This is the first line of defense for CPU optimization. The audio stream first passes through the VAD module; silent segments are directly discarded and do not enter the subsequent expensive ASR and feature extraction stages. Only segments containing valid speech are sent to the buffer [2].
3.  **Bifurcation & Buffering**:
    *   Audio filtered by VAD is copied into two independent queues.
    *   **Acoustic Queue**: Data processing in this queue is near real-time. The system extracts MFCC features from the audio in short time windows (e.g., 100ms-500ms) and immediately sends them to the acoustic emotion model for inference. This means that even while the user is still speaking, the system can provide real-time feedback on "emotional fluctuations" based on tonal changes.
    *   **Language Queue**: This queue requires "Endpointing". ASR models generally have higher accuracy when processing complete sentences or longer phrases. Therefore, the system buffers audio until VAD detects a significant pause (e.g., 500ms of silence), marking the end of a sentence, and then triggers ASR inference [4].
4.  **Inference Layer**:
    *   **ASR Engine**: Responsible for transcribing data from the audio buffer into text. To adapt to CPU, a quantized model (such as Int8 precision Faster-Whisper) must be used.
    *   **Text Analysis Engine**: Receives the text output from ASR and uses the ONNX Runtime-accelerated DistilBERT model for emotion classification.
    *   **Acoustic Analysis Engine**: Receives MFCC feature vectors and uses a lightweight CNN/LSTM model to output acoustic emotion probabilities.
5.  **Fusion & Response Layer**:
    *   The system performs a weighted fusion (Late Fusion) of the emotion scores from the text engine and the acoustic emotion scores within the corresponding time period to generate the final comprehensive emotion result, which is pushed back to the client via WebSocket.

### 2.3 Tech Stack Selection & Decision Rationale

The choice of technology stack directly determines the system's performance ceiling. In CPU-constrained scenarios, every layer of the tech stack must be screened through strict performance benchmarking.

*   **Web Framework**: FastAPI + Uvicorn
    *   **Reason**: FastAPI is based on Starlette, providing native asyncio support and extremely high WebSocket performance. Compared to Flask or Django, FastAPI has higher throughput and lower latency when handling concurrent connections [6]. Uvicorn, as an ASGI server, uses uvloop (based on libuv) to replace Python's default event loop, with performance approaching that of Go.
    *   **Not Selected**: Synchronous frameworks (like Flask) need to allocate a thread for each connection when handling long-lived WebSockets. In CPU-constrained and high-concurrency situations, this leads to severe context switching overhead, triggering "Thrashing".
*   **Inference Engine**: ONNX Runtime (CPU)
    *   **Reason**: ONNX Runtime is a high-performance inference engine open-sourced by Microsoft, specifically performing graph optimizations (such as operator fusion, constant folding) for Transformer models. On CPU, ONNX Runtime combined with quantization technology is typically 2-4 times faster than native PyTorch [8].
    *   **Comparison**: Although PyTorch JIT can also accelerate, ONNX Runtime is more mature in cross-platform and quantization support, and generates smaller model files, which is beneficial for containerized deployment [11].
*   **ASR Backend**: Faster-Whisper (CTranslate2)
    *   **Reason**: OpenAI's original Whisper model is based on PyTorch and has slow inference speed on CPU. The faster-whisper project rewrote Whisper based on the CTranslate2 inference engine, supporting 8-bit quantization (Int8). While maintaining the same accuracy, the speed is increased by more than 4 times, and memory usage is reduced by half [12]. This is crucial for CPU real-time systems.
*   **Audio Processing Library**: Torchaudio vs. Numpy
    *   **Reason**: Although librosa is the standard library for audio analysis, its dependencies are heavy, and processing speed in some scenarios is not as fast as highly optimized torchaudio or direct use of numpy/scipy. Given that DistilBERT requires a PyTorch environment (or for export), using torchaudio allows reusing underlying dependencies, and its C++ core provides efficient MFCC calculation capabilities [13].

## 3. Audio Ingestion & Preprocessing Pipeline Deep Dive

The entry point of the system is the WebSocket handler, which is the source of the data flow. Any latency generated here will be amplified in subsequent stages. Therefore, audio ingestion and preprocessing must be extremely lightweight and efficient.

### 3.1 WebSocket Streaming Protocol Design

To reduce CPU decoding overhead, the communication protocol between client and server should be designed as binary-first.

*   **Audio Format**: Recommended to use 16kHz sample rate, Mono, 16-bit signed integer (Int16 PCM).
    *   **Reason**: Most VAD and ASR models (including Whisper and Silero) internally use a 16kHz sample rate. If resampling is performed on the server side, it consumes valuable CPU resources. Therefore, resampling work should be forced down to the client (browser AudioContext or mobile audio API).
    *   **Packet Size**: It is suggested that the client sends 30ms - 50ms of audio data per packet. For 16kHz 16-bit mono audio, the data volume for 30ms is $16000 \times 0.03 \times 2 = 960$ bytes. This granularity is very suitable for the input requirements of VAD models (Silero VAD typically processes windows of 30ms+) [3].

### 3.2 Voice Activity Detection (VAD): The Gatekeeper of CPU Compute

In real-time stream processing, users often have significant pauses, thinking time, or silence. If the ASR engine continuously processes background noise, it not only wastes compute power but may also lead to hallucinations. VAD is the system's "gatekeeper".

*   **Model Selection**: Silero VAD
    *   **Performance Benchmark**: Silero VAD is currently recognized as the king of CPU efficiency in the open-source community. Benchmarks show that processing a 30ms audio chunk on a single-core CPU takes less than 1ms (RTF < 0.03) [3]. In contrast, while WebRTC VAD is also fast, its noise resistance is inferior to neural network-based Silero VAD.
    *   **Deployment Details**: For further extreme optimization, the **ONNX version** of Silero VAD should be used. This avoids introducing a full PyTorch dependency just to run VAD (if other components are already ONNX-ified), and ONNX Runtime has lower startup and execution overhead [15].
*   **Streaming Logic**:
    The system maintains a VADIterator object, which holds the model's Hidden State.
    1.  **Input**: Receive an audio Chunk.
    2.  **Decision**: VAD outputs the probability that the chunk contains speech.
    3.  **State Machine**:
        *   *Trigger*: When the speech probability of N consecutive chunks is greater than a threshold (e.g., 0.5), the state transitions to SPEECH, and audio buffering begins.
        *   *Release*: When the speech probability of M consecutive chunks is less than the threshold, the state transitions to SILENCE.
    4.  **Endpointing**: When the state transitions from SPEECH to SILENCE and the duration exceeds a set threshold (e.g., 500ms-1000ms), the system determines that a sentence has ended. At this point, the audio in the buffer is packaged and sent to the ASR engine, and the buffer is cleared to prepare for the next sentence.

### 3.3 Buffering Strategy & Memory Management

Frequent memory allocation and copying in Python are CPU killers.

*   **Avoid List Concatenation**: Do not use `buffer += chunk` or `list.append()` followed by `b''.join()` to frequently manipulate data. This creates a large number of temporary objects.
*   **Pre-allocation Strategy**: Use `io.BytesIO` or pre-allocate a fixed-size `bytearray` (Ring Buffer) to store audio data. For ASR input, it usually needs to be converted to a float tensor. This conversion should be deferred until the moment before inference, utilizing `numpy.frombuffer(..., dtype=np.int16).astype(np.float32) / 32768.0` for efficient conversion.

## 4. Acoustic Feature Extraction & Analysis Module (MFCC & CNN)

One of the core features of this system is utilizing acoustic features (MFCC) as an independent basis for emotion judgment. This modality does not rely on *what* the user said, but focuses on *how* the user said it (tone, stress, speaking rate).

### 4.1 Mathematical Principles & Calculation of MFCC

MFCCs are acoustic features that simulate human auditory perception characteristics. The human ear has higher resolution for low-frequency sounds than high-frequency ones; MFCCs simulate this characteristic via the Mel Scale.

**Calculation Flow & CPU Consumption Analysis**:

1.  **Pre-emphasis**: $y(t) = x(t) - \alpha x(t-1)$, typically $\alpha=0.97$. Used to boost energy in the high-frequency part.
2.  **Framing**: Slice the speech signal into short frames (e.g., 25ms), with a hop length of 10ms.
3.  **Windowing**: Multiply each frame by a Hamming Window to reduce spectral leakage.
4.  **Fast Fourier Transform (FFT)**: Convert time-domain signals to frequency-domain power spectra. This is the most computationally intensive step, but modern CPU AVX instruction sets optimize this well.
5.  **Mel Filterbank**: Map the power spectrum to the Mel scale through a set of triangular filters (typically 40).
6.  **Log Energy**: Take the logarithm of the filterbank outputs to simulate the non-linear perception of loudness by the human ear.
7.  **Discrete Cosine Transform (DCT)**: Perform DCT on the log energies to remove correlation between features, obtaining MFCC coefficients. Typically, the first 13-40 coefficients are taken.

**Library Selection & Discrepancy Warning**:
Research indicates that different libraries have discrepancies in MFCC implementation details, directly affecting model consistency [13].
*   `python_speech_features`: Some versions use energy instead of the 0th coefficient and lack standard Delta calculation.
*   `librosa`: Default parameters (like DCT type, normalization method) differ from traditional Kaldi or HTK standards. `librosa.feature.mfcc` defaults to using Slaney-style Mel filterbanks.
*   `torchaudio`: `torchaudio.transforms.MFCC` offers extremely high configuration flexibility, and its C++ underlying implementation (based on ATen) is highly efficient in PyTorch environments.
*   **Conclusion**: To ensure compatibility with deep learning models and computational efficiency, it is recommended to use **torchaudio**. If PyTorch must be stripped out, ensure that the alternative library used (e.g., numpy implementation) aligns strictly mathematically with the library used during training (especially DCT type and Mel filterbank normalization).

### 4.2 Acoustic Emotion Model Architecture

Since text sentiment analysis (DistilBERT) already consumes significant computing resources, the acoustic model must be designed to be extremely lightweight.

*   **Model Architecture**: 1D-CNN
    *   Compared to LSTM or Transformer, **1D Convolutional Neural Networks (1D-CNN)** have better parallelism on CPU. Convolution operations can be compiled into highly optimized matrix multiplications without recurrent dependencies.
    *   **Input**: MFCC matrix, shape (Batch, N_MFCC, Time_Steps). For example, (1, 40, 300) corresponds to 3 seconds of audio.
    *   **Structure**:
        *   Conv1D (Kernel=3, Filters=64, ReLU)
        *   MaxPool1D (Size=2)
        *   Conv1D (Kernel=3, Filters=128, ReLU)
        *   GlobalAveragePooling1D (Compresses variable-length sequence to fixed-length vector)
        *   Dense (Units=3, Softmax: Pos, Neg, Neu)
    *   The parameter size of such a model is typically in the range of a few hundred KB, and inference latency can be controlled within 5ms.
*   **Training Data**: The model should be trained on audio datasets containing emotion annotations, such as RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) or CMU-MOSEI. Note that audio augmentation (noise addition, speed perturbation) should be applied during training to improve robustness [18].

## 5. Automatic Speech Recognition (ASR) Module Optimization

ASR is the biggest computational bottleneck in the entire system. Running real-time ASR on CPU requires a difficult trade-off between accuracy and latency.

### 5.1 ASR Model Landscape Comparison (2024/2025)

Based on the latest benchmarks [20], here is a comparison of current mainstream open-source ASR solutions:

| Model Solution | Accuracy (WER) | CPU Real-time (RTF) | Resource Consumption | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **OpenAI Whisper (Tiny/Base)** | Excellent (Strong Multilingual) | Poor (Native PyTorch) | High | Offline batch processing, GPU environment |
| **Faster-Whisper (Int8)** | Excellent | Good (0.2 - 0.5) | Medium (Memory < 1GB) | CPU real-time stream processing |
| **Vosk (Kaldi)** | Medium (Vocabulary dependent) | Excellent (< 0.1) | Low | Extremely low-resource devices, specific command words |
| **Silero Models** | Good | Excellent | Extremely Low | Compact applications |
| **Wav2Vec2** | Good | Medium | High | Requires domain-specific fine-tuning |

**Decision**: Although Vosk and Silero have advantages in speed, Whisper has overwhelming advantages in handling colloquial speech, noisy environments, and zero-shot multilingual capabilities. For sentiment analysis, the semantic accuracy of the transcription is crucial. Therefore, choosing **Faster-Whisper (Int8 Quantized)** optimized via CTranslate2 is the best balance point.

### 5.2 CPU Optimization Strategies for Faster-Whisper

The faster-whisper library achieves real-time performance on CPU through the following technologies:

1.  **CTranslate2 Engine**: This is a C++ library specifically optimized for Transformer model inference, supporting direct quantized computation of weights.
2.  **8-bit Quantization (Int8)**: Converts model weights from 32-bit floating point (FP32) to 8-bit integers. This not only reduces model size by 4 times (Tiny model requires only ~75MB) but also utilizes CPU vectorization instructions (like AVX2/AVX-512 VNNI) to accelerate integer arithmetic [12].
3.  **Beam Search Restriction**: In real-time streams, setting `beam_size` to 1 (greedy decoding) or a very small value (e.g., 2) can significantly reduce computation. Although this sacrifices a tiny bit of accuracy, it yields a significant reduction in latency.

**Implementation Details**:
The ASR engine should run in "non-streaming" mode on "micro-streaming" data. That is, whenever VAD detects the end of a sentence, the ASR engine processes this audio segment. This method is more suitable for sentiment analysis scenarios than trying to output word-by-word streaming (which requires constant correction of previous context), because emotion is usually judged based on complete thought groups.

## 6. Text Sentiment Analysis Module (DistilBERT & ONNX)

The user specified using DistilBERT for text sentiment analysis. This is a classic application case of Knowledge Distillation.

### 6.1 DistilBERT Architecture Advantages

DistilBERT is a lightweight version of BERT. It learns knowledge from a large BERT model (Teacher) during the pre-training phase via knowledge distillation techniques [23].

*   **Halved Layers**: DistilBERT has only 6 Transformer Encoder layers (BERT-Base has 12).
*   **Reduced Parameters**: Parameter count is about 66M (BERT-Base is about 110M).
*   **Performance Retention**: Retains 97% of BERT's performance, but inference speed on CPU is increased by 60%.

### 6.2 ONNX Runtime Deep Optimization

Using DistilBERT alone is not enough; to achieve ultra-low latency (<50ms) on CPU, **ONNX Runtime (ORT)** must be used.

*   **Graph Optimization**:
    ORT performs a series of graph-level optimizations when loading the model:
    *   **Operator Fusion**: Fuses multiple fine-grained operators (like MatMul + Add + ReLU) into one large operator, reducing memory access and kernel launch overhead. For example, Multi-Head Attention in Transformers is fused into a single Attention operator.
    *   **Constant Folding**: Pre-computes constant nodes in the static graph.
    *   **Redundancy Elimination**: Removes unused nodes and unnecessary Reshape operations.
*   **Dynamic Quantization**:
    For Transformer-based models, dynamic quantization is the best practice for CPU inference [25].
    *   **Principle**: Weights are pre-quantized to 8-bit integers (Int8), but during inference, activations are read dynamically. Before each layer's calculation, quantization parameters (Scale and Zero-point) are dynamically calculated based on the range of the current input data. Activations are quantized to Int8 for integer matrix multiplication, then de-quantized back to FP32.
    *   **Benefit**: Compared to static quantization (which requires a calibration dataset), dynamic quantization has minimal impact on accuracy, is simple to implement, and can bring 2-3x acceleration.
    *   **Toolchain**: Using Hugging Face's `optimum` library can implement conversion in one line of code:
        ```python
        from optimum.onnxruntime import ORTQuantizer
        # ... Load model and configure quantization parameters ...
        quantizer.quantize(save_dir="quantized_model", quantization_config=qconfig)
        ```

## 7. Multimodal Fusion Strategy

How to combine the emotion of sound (Tone) and the emotion of text (Content)?

### 7.1 Fusion Level Selection: Late Fusion

In multimodal deep learning, there are three main fusion strategies:

*   **Early Fusion**: Concatenate MFCC features and text Embeddings then input into a large model.
*   **Intermediate Fusion**: Interaction in the intermediate layers of respective models (e.g., Cross-Attention).
*   **Late Fusion (Decision Level Fusion)**: Respective models independently output prediction results (probabilities or Logits), then perform weighting or voting at the decision layer [26].

This blueprint recommends: **Late Fusion**.

*   **Reason 1**: Sample rate mismatch. MFCC is a continuous sequence of time frames, while DistilBERT processes discrete Token sequences. In real-time streams, it is difficult to achieve precise time alignment at the feature level.
*   **Reason 2**: Robustness. If ASR transcription errors occur in a noisy environment (causing text sentiment analysis to fail), late fusion allows the system to degrade gracefully to rely on the acoustic model, still capturing the user's angry or excited tone.
*   **Reason 3**: Modularity. Late fusion allows independent development, training, and upgrading of text and acoustic models without interference.

### 7.2 Fusion Algorithm Implementation

Assume the emotion categories are three: {Negative, Neutral, Positive}.

1.  **Input Alignment**:
    The ASR engine outputs a segment of text $T$, corresponding to the time period $t_{start}$ to $t_{end}$.
    The acoustic model may have produced $N$ prediction results within that time period (e.g., one every 500ms).
2.  **Acoustic Result Aggregation**:
    Pool the $N$ acoustic probability vectors. Typically use **Average Pooling** or **Max Pooling** (for capturing strong emotions).
    $$P_{audio} = \frac{1}{N} \sum_{i=1}^{N} p_{audio}^{(i)}$$
3.  **Weighted Fusion**:
    $$P_{final} = \alpha \cdot P_{text} + (1 - \alpha) \cdot P_{audio}$$
    Where $\alpha$ is a hyperparameter. Usually, text contains more explicit information, so $\alpha$ can be set to 0.6 to 0.7. Alternatively, a simple Logistic Regression model can be used as a fuser to dynamically adjust weights based on confidence. For example, if DistilBERT's output entropy is high (indicating uncertainty), the weight of text can be automatically reduced, relying more on acoustic judgment.

## 8. System Implementation & Engineering Optimization

At the code implementation level, the efficiency of every line of Python code is crucial.

### 8.1 Python Asynchronous Concurrency Control

FastAPI applications run in a single process (usually). Although asyncio can handle high concurrency I/O, model inference is a **CPU-intensive** task that blocks the event loop.

*   **Fatal Trap**: If `model.predict()` is called directly in an `async def` route, the entire service will be unable to respond to any heartbeats or new WebSocket connections during inference, leading to disconnection.
*   **Solution**: Inference tasks must be offloaded to a `ThreadPoolExecutor` or `ProcessPoolExecutor` via `run_in_executor` [1].
    ```python
    loop = asyncio.get_running_loop()
    # Run blocking ASR inference in thread pool, freeing main thread for WebSocket heartbeats
    text = await loop.run_in_executor(pool, asr_model.transcribe, audio_data)
    ```
*   **Thread Count Limit**: In Docker containers, set environment variables `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` to limit the number of threads for underlying libraries (PyTorch/ONNX/Numpy). Otherwise, if 4 requests are concurrent and each attempts to launch 4 OpenMP threads, it will cause severe CPU context switching and drastic performance degradation [31].

### 8.2 Docker Image Extreme Optimization

Container image size directly affects cloud service cold start time (Image Pull time).

*   **Base Image**: Although Alpine Linux is the smallest, it uses musl libc, while many Python scientific computing libraries (PyTorch, NumPy, Pandas) are compiled against glibc. Installing these on Alpine often requires compiling from source, which is extremely time-consuming and error-prone. Therefore, **Debian Slim** versions (e.g., `python:3.10-slim`) are recommended [33].
*   **PyTorch CPU Version**: Standard `pip install torch` downloads a massive whl file (>700MB) containing CUDA support. You must explicitly specify downloading the CPU-only whl file, reducing size to ~100MB.
    *   Command Example: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` [35].
*   **Multi-stage Build**: Use a build stage to install build dependencies (like build-essential, libsndfile1-dev), then only copy the installed `/usr/local/lib/python3.x/site-packages` to the final runtime image, thereby eliminating compilers and cache files, significantly reducing image size [37].

## 9. Deployment & Infrastructure Strategy (Google Cloud Run)

This blueprint targets Google Cloud Run as the deployment environment, a serverless container platform ideally suited for handling stateless WebSocket services.

### 9.1 CPU Allocation & Concurrency Settings

Cloud Run's CPU allocation mechanism is key [31].

*   **CPU Always Allocated**: For WebSocket services, the "CPU always allocated" option must be enabled. If CPU is only allocated "during request processing", WebSocket long connections may be frozen when no data is being transmitted, causing connection drops.
*   **vCPU Count**: Recommended configuration is **2 vCPU + 4GB RAM**.
    *   1 vCPU is usually stretched for running ASR and WebSocket loops concurrently, easily leading to audio packet loss.
    *   Faster-Whisper and DistilBERT occupy about 1.5GB memory after loading; 4GB provides enough buffer to prevent OOM (Out of Memory).
*   **Concurrency**: The number of requests a single container instance handles simultaneously.
    *   Since ASR is a heavy compute task, it is recommended to set Concurrency to **1** or a very small value (e.g., **2-4**) [38]. This means each container instance serves only 1 user at a time. This ensures that the user gets full CPU power, achieving true "real-time". If set to 80, 80 users competing for 2 vCPUs will result in latency reaching tens of seconds, making the system completely unusable. Cloud Run will automatically scale out container instances to handle more users.

### 9.2 Startup Boost & Cold Start

Deep learning model loading (Load Weights) typically takes a few seconds. This causes high cold start latency for the first user's request.

*   **CPU Boost**: Cloud Run offers "Startup CPU Boost", temporarily providing extra CPU power (e.g., doubling) during container startup, significantly shortening model loading and application initialization time [39]. Be sure to enable this.
*   **Liveness Probe**: Configure a health check endpoint. The application should only return 200 OK after the model is fully loaded into memory. Only then will Cloud Run direct traffic to that instance, avoiding user requests hitting an unready service.

## 10. Performance Benchmarks & Expected Metrics

Based on the above architecture, expected performance metrics on a standard 2 vCPU cloud instance are as follows:

| Metric | Expected Value | Remarks |
| :--- | :--- | :--- |
| **VAD Latency** | < 5 ms | Almost imperceptible |
| **ASR Latency** | 300 - 800 ms | Depends on sentence length (Faster-Whisper Int8) |
| **NLP Latency** | 30 - 60 ms | DistilBERT ONNX Int8 |
| **Acoustic Model Latency** | < 10 ms | CNN Model |
| **End-to-End Latency** | 500 - 1000 ms | Time from user finishing speaking to seeing result |
| **Throughput** | 1 Stream/Instance | Limit concurrency to ensure real-time performance |
| **Cold Start Time** | < 5 Seconds | With CPU Boost enabled |

## 11. Conclusion

This report details a feasible path for building a real-time multimodal sentiment analysis system in a CPU-constrained environment. The core strategy lies in **"Algorithm Slimming" and "Asynchronous Architecture"**. By adopting Faster-Whisper Int8 quantized models, ONNX Runtime-accelerated DistilBERT, and efficient Silero VAD, combined with a late fusion strategy, we can achieve commercially viable near real-time sentiment analysis services without relying on GPUs.

The key to engineering implementation lies in the details: strict Docker image slimming, correct torch CPU version installation, fine-grained thread pool management, and Cloud Run CPU Boost configuration. This blueprint not only meets user functional requirements but also finds the optimal balance between cost, performance, and maintainability.

*Note: All technology selections in this report are based on open-source community best practices and benchmark data from the 2024-2025 period.*

##### Works cited
*(References 1-39 preserved from original document)*

---

#### 🗓 Phase 1: Infrastructure & Data Ingestion (SWE + VAD)

**Goal**: Build a high-performance asynchronous backend capable of handling WebSocket connections, integrate Voice Activity Detection (VAD) to filter invalid audio, and establish an efficient audio buffering mechanism.

| Field | Description |
| :--- | :--- |
| **Acceptance Criteria** | 1. WebSocket service can stably receive binary audio streams.<br>2. VAD module can accurately distinguish speech and silence (Accuracy > 90%).<br>3. System can correctly segment "speech segments" into queues with CPU usage < 10%. |
| **Key Tech Stack** | Python 3.10+, FastAPI, WebSockets, Silero VAD (ONNX), NumPy |
| **Core Challenge** | Buffer memory management. Python list expansion and bytes concatenation are very time-consuming; use pre-allocated buffer or `io.BytesIO`. |
| **Resources Needed** | Silero VAD ONNX Model File (Github Link). |

**🚀 Implementation Steps**:

1.  **Initialize FastAPI Project Structure**:
    *   Create `app/main.py`, define WebSocket route `/ws/analyze`.
    *   Configure `uvicorn` startup parameters, set `log_level="info"` for debugging.
2.  **Implement Ring Buffer**:
    *   Create an `AudioBuffer` class, using `bytearray` to store raw PCM data (Recommended format: 16kHz, Mono, Int16).
    *   Implement `append(chunk)` and `get_window(size)` methods to avoid frequent memory allocation.
3.  **Integrate Silero VAD (ONNX)**:
    *   Download `silero_vad.onnx`.
    *   Use `onnxruntime` to load the model.
    *   Write `VADIterator` class, processing 30ms-50ms audio windows, maintaining VAD internal state, outputting speech probability.
4.  **Develop Endpointing Logic**:
    *   In the WebSocket loop, when VAD detects N consecutive silent frames (e.g., lasting 500ms), trigger "Sentence End" event.
    *   Package this complete speech data and put it into `asyncio.Queue` for subsequent processing.
5.  **Benchmark**:
    *   Use `wscat` or write a simple Python script to simulate a client sending audio streams, measuring VAD processing latency (Target: < 1ms per frame).

#### 🗓 Phase 2: Acoustic Sentiment Analysis Pipeline (ML - Acoustic)

**Goal**: Develop and deploy a lightweight acoustic model that judges emotion based solely on physical characteristics of sound (tone, speed, energy), independent of text content.

| Field | Description |
| :--- | :--- |
| **Acceptance Criteria** | 1. Can extract MFCC features from raw audio in real-time.<br>2. Acoustic model inference latency < 20ms (CPU).<br>3. Model accuracy on validation set reaches baseline (approx. 60%-70% is acceptable for acoustic unimodal). |
| **Key Tech Stack** | Torchaudio (C++ accelerated MFCC), ONNX Runtime, PyTorch (Training only), Scikit-learn |
| **Core Challenge** | Real-time feature calculation. MFCC calculation must be extremely fast on CPU. Avoid using `librosa` for real-time inference (heavy dependencies); recommend `torchaudio` or `python_speech_features`. |
| **Resources Needed** | Datasets: RAVDESS (Speech Audio only) or TESS. |

**🚀 Implementation Steps**:

1.  **Data Preparation & Preprocessing**:
    *   Download RAVDESS dataset.
    *   Write script to resample all audio to 16kHz Mono.
    *   Extract MFCC features (Suggested params: `n_mfcc=40`, `n_fft=400`, `hop_length=160`).
2.  **Train Lightweight 1D-CNN Model**:
    *   Define a simple model using PyTorch: `Conv1d(40, 64) -> ReLU -> MaxPool -> Conv1d(64, 128) -> GlobalAvgPool -> FC(3)` (Pos, Neg, Neu).
    *   Train model, save as `.pt` file.
3.  **Model Export & Quantization**:
    *   Use `torch.onnx.export` to export model to ONNX format.
    *   (Optional) Use ONNX Runtime quantization tools to convert to INT8 format for further acceleration.
4.  **Integrate Inference Service**:
    *   Create an independent `AcousticWorker` in the FastAPI app.
    *   When receiving a speech segment confirmed by VAD, slice and extract MFCCs and send to ONNX Session for inference.
    *   Note: MFCC extraction needs exception handling (handling extremely short audio segments).

#### 🗓 Phase 3: Text Sentiment Analysis Pipeline (ML - Text & ASR)

**Goal**: Integrate ASR engine to convert speech to text, and run NLP model to analyze text semantic emotion. This is the most computationally intensive part.

| Field | Description |
| :--- | :--- |
| **Acceptance Criteria** | 1. ASR can complete short sentence transcription in < 500ms.<br>2. Text sentiment model inference latency < 50ms.<br>3. Pipeline can handle concurrent requests without blocking WebSocket heartbeats. |
| **Key Tech Stack** | Faster-Whisper (Int8), DistilBERT (ONNX Quantized), HuggingFace Optimum |
| **Core Challenge** | CPU resource contention. Both ASR and NLP models consume significant CPU. Must limit thread count (`OMP_NUM_THREADS=1`) and use `run_in_executor` to avoid blocking main thread. |
| **Resources Needed** | Pre-trained models: `systran/faster-whisper-tiny` or `small`, `distilbert-base-uncased-finetuned-sst-2-english`. |

**🚀 Implementation Steps**:

1.  **ASR Engine Integration**:
    *   Install `faster-whisper`.
    *   Load `tiny.en` or `small.en` model, set `compute_type="int8"`.
    *   Write `transcribe` function, receiving NumPy array audio, returning text string.
2.  **NLP Model Optimization (Optimum)**:
    *   Use HuggingFace `optimum` library to load DistilBERT model.
    *   Execute Dynamic Quantization and export to ONNX.
    ```python
    from optimum.onnxruntime import ORTModelForSequenceClassification
    model = ORTModelForSequenceClassification.from_pretrained("distilbert...", export=True)
    ```
3.  **Async Pipeline Orchestration**:
    *   In FastAPI, when VAD determines sentence end:
        1.  `await loop.run_in_executor(pool, asr_model.transcribe, audio)`
        2.  After getting text -> `await loop.run_in_executor(pool, nlp_model.predict, text)`
4.  **Error Handling**:
    *   Handle "Hallucination" text from ASR (e.g., silence transcribed as "Thank you" or repeated characters). Set up simple text filters.

#### 🗓 Phase 4: Fusion, Containerization & Deployment (Fusion + Deploy)

**Goal**: Fuse results from both modalities, package the application, and deploy to cloud serverless environment.

| Field | Description |
| :--- | :--- |
| **Acceptance Criteria** | 1. Docker image size < 1.5GB (Ideal < 800MB).<br>2. Cloud Run cold start time < 5 seconds.<br>3. System returns JSON containing `text_sentiment`, `audio_sentiment`, and `fused_score`. |
| **Key Tech Stack** | Docker (Multi-stage), Google Cloud Run, Late Fusion Logic |
| **Core Challenge** | Docker image slimming. PyTorch installs CUDA version by default, which is huge. Must force install CPU version. |
| **Resources Needed** | Google Cloud Platform Account (Cloud Run Service). |

**🚀 Implementation Steps**:

1.  **Implement Late Fusion Strategy**:
    *   Write fusion function: `final_score = w1 * text_prob + w2 * audio_prob`. Suggested `w1=0.7` (text weight), `w2=0.3` (acoustic weight).
    *   Design final JSON format returned by WebSocket.
2.  **Write Dockerfile**:
    *   Use `python:3.10-slim` as base image.
    *   Key step: Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` to install only CPU version of PyTorch.
    *   Clean apt cache and pip cache (`rm -rf /var/lib/apt/lists/*`).
3.  **Local Integration Test**:
    *   Use `docker build` to build image.
    *   Run container and use test script (send a 5-second sad voice) to verify end-to-end flow.
4.  **Cloud Run Deployment Configuration**:
    *   Deploy to Google Cloud Run.
    *   **Required Config**:
        *   CPU allocation: "CPU is always allocated" (Prevent WebSocket disconnect).
        *   Minimum instances: 1 (If budget allows) or enable Startup CPU Boost (Accelerate cold start).
        *   Concurrency: Set to 1 or 2 (Ensure each request gets full CPU time slice).

#### 💡 Special Tips for Developers

*   **Debugging Tool**: During development, save audio received by WebSocket as `.wav` files simultaneously to facilitate listening back and troubleshooting VAD segmentation correctness.
*   **Performance Monitoring**: Add `time.perf_counter()` in code to record VAD time, ASR time, NLP time, and print in logs; this is crucial for subsequent optimization.
*   **Degradation Strategy**: If ASR takes too long (>1 second), design to return acoustic model emotion results first (since it's fast), and push text emotion result updates after ASR completes.
