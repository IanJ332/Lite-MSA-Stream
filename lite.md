# [cite\_start]实时多模态情感分析系统蓝图: 基于CPU环境的DistilBERT与MFCC流式微服务架构深度研究报告 [cite: 1]

-----

## 1\. 执行摘要与架构愿景

[cite\_start]本研究报告旨在构建一个详尽的工程蓝图，用于开发和部署一个实时多模态情感分析系统 [cite: 3][cite\_start]。该系统的核心任务是处理实时音频流，从中提取语言内容（文本）和副语言特征（声学特征），并分别利用 **DistilBERT** 模型和 **梅尔频率倒谱系数 (MFCCs)** 进行情感推断，最终在决策层融合这两种模态以生成全面的情感洞察 [cite: 3]。

[cite\_start]该项目的关键约束在于部署环境限制为 **CPU架构**，这要求系统在设计上必须极致追求计算效率、资源管理和延迟优化 [cite: 3]。

[cite\_start]在CPU环境下实现“实时”处理（即处理速度快于音频流的播放速度，通常要求实时因子 RTF \< 1.0）面临巨大挑战 [cite: 4][cite\_start]。本报告不仅仅是模型堆叠指南，更是一份关于如何在受限计算资源下，通过算法优化、模型压缩（量化、蒸馏）和异步架构设计来突破性能极限的深度技术文档 [cite: 4]。

**核心解决方案：**

  * [cite\_start]**策略**：采用“晚期融合”（Late Fusion）策略 [cite: 5]。
  * [cite\_start]**架构**：基于 Python `asyncio` 和 FastAPI，利用 WebSocket 实现全双工通信 [cite: 5]。
  * **双流设计**：
      * [cite\_start]**文本路径**：轻量级 ASR 引擎（Faster-Whisper Int8） -\> ONNX Runtime 优化的 DistilBERT [cite: 5, 6]。
      * [cite\_start]**声学路径**：直接提取 MFCC 特征 -\> 轻量级 CNN 进行声调情感分类 [cite: 6]。
  * [cite\_start]**优势**：解耦设计确保即使 ASR 处理出现微小延迟，声学特征分析也能保持准实时响应 [cite: 6]。

-----

## 2\. 系统架构设计与微服务模式

### 2.1 整体架构模式: 单体微服务与异步I/O

[cite\_start]为了在CPU环境中实现低延迟，系统必须摒弃传统的阻塞式 I/O 模型，转而采用事件驱动的异步流处理架构 [cite: 8]。

  * [cite\_start]**架构选择**：推荐采用 **单体微服务 (Monolithic Microservice)** 架构 [cite: 11]。
  * [cite\_start]**理由**：在实时性和CPU限制下，传统的分布式微服务会引入网络延迟和序列化开销 [cite: 10][cite\_start]。所有模型（ASR、DistilBERT、Audio-CNN）共享同一进程空间，消除了网络传输开销 [cite: 11]。

### 2.2 数据流与流水线设计

[cite\_start]系统遵循“流水线-过滤器”（Pipeline-Filter）模式 [cite: 13]：

1.  **接入层 (Ingestion Layer)**：
      * [cite\_start]客户端通过 WebSocket 发送二进制 PCM 音频流 [cite: 15]。
      * [cite\_start]**优化**：避免使用 Base64 编码，直接使用二进制 Blob 传输 [cite: 16]。
2.  **过滤层 (Filter Layer)**：
      * [cite\_start]**语音活动检测 (VAD)**：音频流首先经过 VAD 模块，静音片段被丢弃，只有有效语音进入缓冲区 [cite: 18]。
3.  **分流与缓冲 (Bifurcation & Buffering)**：
      * [cite\_start]**声学队列**：准实时处理（100ms-500ms 窗口），提取 MFCC 并立即推理，实时反馈“情绪波动” [cite: 21]。
      * [cite\_start]**语言队列**：进行“端点检测”（Endpointing），缓冲直到检测到句子结束（如 500ms 静音），触发 ASR [cite: 22]。
4.  **推理层 (Inference Layer)**：
      * [cite\_start]**ASR 引擎**：Faster-Whisper (Int8) 将音频转写为文本 [cite: 24]。
      * [cite\_start]**文本分析**：ONNX Runtime 加速的 DistilBERT 进行分类 [cite: 25]。
      * [cite\_start]**声学分析**：轻量级 CNN/LSTM 模型处理 MFCC 向量 [cite: 26]。
5.  **融合与响应层 (Fusion & Response Layer)**：
      * [cite\_start]执行晚期融合（Late Fusion），生成最终综合情感结果并推送回客户端 [cite: 27]。

### [cite\_start]2.3 技术栈选型与决策依据 [cite: 28]

| 组件 | 选型 | 决策理由 |
| :--- | :--- | :--- |
| **Web 框架** | **FastAPI + Uvicorn** | [cite\_start]原生 asyncio 支持，WebSocket 性能极高；Uvicorn 基于 uvloop，性能接近 Go [cite: 30, 31]。 |
| **推理引擎** | **ONNX Runtime (CPU)** | [cite\_start]微软开源，针对 Transformer 进行图优化（算子融合、常量折叠），配合量化比原生 PyTorch 快 2-4 倍 [cite: 33, 35]。 |
| **ASR 后端** | **Faster-Whisper (CTranslate2)** | [cite\_start]基于 CTranslate2 重写，支持 Int8 量化，速度提升 4 倍以上，内存占用减半 [cite: 37, 39]。 |
| **音频处理** | **Torchaudio** | [cite\_start]C++ 核心提供高效 MFCC 计算，且能复用 PyTorch 底层依赖 [cite: 42]。 |

-----

## 3\. 音频摄取与预处理流水线深度解析

### 3.1 WebSocket 流式传输协议设计

  * [cite\_start]**音频格式**：推荐 16kHz 采样率、单声道 (Mono)、16位有符号整数 (Int16 PCM) [cite: 47]。
      * [cite\_start]*原因*：大多数模型（Whisper, Silero）内部使用 16kHz，重采样应下放至客户端完成以节省服务端 CPU [cite: 48]。
  * [cite\_start]**数据包大小**：建议每包 30ms-50ms（约 960 字节），适合 VAD 输入要求 [cite: 49]。

### 3.2 语音活动检测 (VAD): CPU算力的守门员

  * [cite\_start]**模型选择**：**Silero VAD** [cite: 52]。
      * [cite\_start]*性能*：单核 CPU 处理 30ms 音频块仅需 \<1ms (RTF \< 0.03) [cite: 53]。
      * [cite\_start]*部署*：使用 ONNX 版本以降低启动和执行开销 [cite: 54, 55]。
  * [cite\_start]**流式处理逻辑** [cite: 57-63]：
    1.  **输入**：接收音频块 (Chunk)。
    2.  **判断**：输出语音概率。
    3.  **状态机**：
          * *Trigger*：连续 N 个块概率 \> 阈值 -\> 状态转为 SPEECH。
          * *Release*：连续 M 个块概率 \< 阈值 -\> 状态转为 SILENCE。
    4.  **端点检测**：从 SPEECH 转为 SILENCE 且持续超过阈值（如 500ms），判定一句话结束，触发 ASR。

### 3.3 缓冲策略与内存管理

  * [cite\_start]**避免列表拼接**：不要使用 `buffer += chunk`，会产生大量临时对象 [cite: 66]。
  * [cite\_start]**预分配策略**：使用 `io.BytesIO` 或预分配固定大小的 `bytearray` (Ring Buffer) [cite: 67]。
  * [cite\_start]**类型转换**：推迟到推理前一刻使用 `numpy` 进行 Int16 到 Float32 的转换 [cite: 67]。

-----

## 4\. 声学特征提取与分析模块 (MFCC与CNN)

### 4.1 MFCC 的数学原理与计算

[cite\_start]MFCC 模拟人类听觉感知特性 [cite: 71][cite\_start]。计算流程包括：预加重、分帧、加窗、FFT、梅尔滤波器组、对数能量、DCT [cite: 73-81]。

  * [cite\_start]**库的选择**：建议使用 **torchaudio** [cite: 87]。
      * [cite\_start]*原因*：`torchaudio.transforms.MFCC` 提供极高效率（基于 C++ ATen），且能保证与深度学习模型的兼容性 [cite: 86][cite\_start]。相比之下，librosa 依赖较重且某些默认参数（如 DCT 类型）可能不一致 [cite: 85]。

### 4.2 声学情感模型架构

  * [cite\_start]**架构**：**1D-CNN** [cite: 89]。
      * [cite\_start]*优势*：在 CPU 上比 LSTM/Transformer 具有更好的并行性，卷积操作可被编译为优化的矩阵乘法 [cite: 90]。
  * [cite\_start]**模型结构示例** [cite: 93-97]：
      * `Conv1D (Kernel=3, Filters=64, ReLU)`
      * `MaxPool1D (Size=2)`
      * `Conv1D (Kernel=3, Filters=128, ReLU)`
      * `GlobalAverage Pooling1D`
      * `Dense (Units=3, Softmax)`
  * [cite\_start]**性能**：参数量几百 KB，推理延迟 \< 5ms [cite: 98]。

-----

## 5\. 自动语音识别 (ASR) 模块优化

[cite\_start]ASR 是最大的计算瓶颈 [cite: 103]。

### 5.1 模型对比与决策

[cite\_start]虽然 Vosk 和 Silero 更快，但 **Faster-Whisper (Int8)** 是最佳平衡点，因为它在处理口语、噪音环境及零样本多语言能力上具有压倒性优势 [cite: 107]。

### 5.2 Faster-Whisper 的 CPU 优化策略

1.  [cite\_start]**CTranslate2 引擎**：专为 Transformer 推理优化的 C++ 库 [cite: 110]。
2.  [cite\_start]**8-bit 量化 (Int8)**：模型体积减少 4 倍，利用 CPU 向量化指令 (AVX2) 加速 [cite: 111]。
3.  [cite\_start]**Beam Search 限制**：将 `beam_size` 设为 1 (贪婪解码)，显著降低延迟 [cite: 112]。

[cite\_start]**实施细节**：ASR 引擎应以“非流式”模式运行在“微流式”数据上（即按句处理），这比逐字流式输出更适合情感分析 [cite: 114]。

-----

## 6\. 文本情感分析模块 (DistilBERT与ONNX)

### 6.1 DistilBERT 架构优势

  * [cite\_start]**轻量化**：6 层 Transformer (BERT-Base 是 12 层)，参数量 \~66M [cite: 120, 121]。
  * [cite\_start]**性能**：保留 BERT 97% 的性能，CPU 推理速度提升 60% [cite: 122]。

### 6.2 ONNX Runtime 深度优化

[cite\_start]要在 CPU 上达到 \<50ms 延迟，必须使用 ONNX Runtime [cite: 124]。

  * [cite\_start]**图优化**：算子融合 (Operator Fusion)、常量折叠、消除冗余 [cite: 126-132]。
  * **动态量化 (Dynamic Quantization)**：
      * [cite\_start]*原理*：权重预先量化为 Int8，推理时根据输入动态计算激活值的量化参数 [cite: 135, 136]。
      * [cite\_start]*收益*：比静态量化实施简单，能带来 2-3 倍加速 [cite: 136]。
      * [cite\_start]*代码示例* [cite: 138-141]:
        ```python
        from optimum.onnxruntime import ORTQuantizer
        # ... 加载模型并配置量化参数 ...
        quantizer.quantize(save_dir="quantized_model", quantization_config=qconfig)
        ```

-----

## 7\. 多模态融合策略 (Multimodal Fusion)

### 7.1 融合层级选择: 晚期融合 (Late Fusion)

[cite\_start]本蓝图推荐 **晚期融合** [cite: 150]。

  * [cite\_start]**理由 1**：采样率不匹配（连续时间帧 vs 离散 Token）难对齐 [cite: 151]。
  * [cite\_start]**理由 2**：鲁棒性高，若 ASR 失败，系统可降级依赖声学模型 [cite: 152]。
  * [cite\_start]**理由 3**：模块化强，模型可独立开发 [cite: 153]。

### 7.2 融合算法实现

[cite\_start]假设情感类别为 {Negative, Neutral, Positive} [cite: 155]。

1.  [cite\_start]**声学结果聚合**：对时间段内的 $N$ 个声学概率向量进行池化（如平均池化）[cite: 160]。
    [cite\_start]$$P_{audio} = \frac{1}{N} \sum_{i=1}^{N} p_{audio}^{(i)}$$ [cite: 161]
2.  **加权融合**：
    [cite\_start]$$P_{final} = \alpha \cdot P_{text} + (1-\alpha) \cdot P_{audio}$$ [cite: 163]
    [cite\_start]其中 $\alpha$ 是超参数（如 0.6-0.7），也可根据文本模型输出的熵动态调整权重 [cite: 164]。

-----

## 8\. 系统实现与工程优化

### 8.1 Python 异步并发控制

  * [cite\_start]**陷阱**：在 `async def` 中直接运行 CPU 密集型推理会阻塞 WebSocket 心跳 [cite: 169]。
  * [cite\_start]**解决方案**：使用 `loop.run_in_executor` 将推理放入线程池 [cite: 170, 175]。
    ```python
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(pool, asr_model.transcribe, audio_data)
    ```
  * [cite\_start]**线程限制**：设置 `OMP_NUM_THREADS=1`，防止底层库争抢 CPU 导致上下文切换 [cite: 176, 177]。

### 8.2 Docker 镜像极致优化

  * [cite\_start]**基础镜像**：推荐 **Debian Slim** (`python:3.10-slim`)，避免 Alpine 编译问题 [cite: 181]。
  * [cite\_start]**PyTorch CPU 版本**：必须明确指定下载 CPU 专用 whl，体积可从 \>700MB 缩减至 \~100MB [cite: 182]。
      * [cite\_start]命令：`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` [cite: 183]。
  * [cite\_start]**多阶段构建**：利用 Multi-stage Build 剔除编译器和缓存 [cite: 184]。

-----

## 9\. 部署与基础设施策略 (Google Cloud Run)

### 9.1 CPU 分配与并发设置

  * [cite\_start]**CPU Always Allocated**：必须开启，防止 WebSocket 长连接在无数据时被冻结 [cite: 190]。
  * [cite\_start]**配置**：建议 2 vCPU + 4GB RAM [cite: 191]。
  * [cite\_start]**并发度 (Concurrency)**：建议设置为 **1** [cite: 195]。
      * [cite\_start]*原因*：保证每个用户获得完整 CPU 算力，实现真正“实时”。Cloud Run 会自动横向扩展 [cite: 195]。

### 9.2 启动加速与冷启动

  * [cite\_start]**Startup CPU Boost**：开启此功能以在启动阶段提供额外算力，缩短模型加载时间 [cite: 198]。
  * [cite\_start]**健康检查**：配置 Liveness Probe，确保模型加载完毕后再接收流量 [cite: 199]。

-----

## 10\. 性能基准与预期指标

[cite\_start]基于标准 2 vCPU 云实例的预期性能 [cite: 201, 202]：

| 指标 | 预期值 | 备注 |
| :--- | :--- | :--- |
| **VAD 延迟** | **\< 5 ms** | 几乎无感 |
| **ASR 延迟** | **300-800 ms** | 取决于句子长度 (Faster-Whisper Int8) |
| **NLP 延迟** | **30-60 ms** | DistilBERT ONNX Int8 |
| **声学模型延迟** | **\< 10 ms** | CNN 模型 |
| **端到端延迟** | **500-1000 ms** | 用户说完话到看到结果的时间 |
| **吞吐量** | **1流/实例** | 为保证实时性，建议限制并发 |
| **冷启动时间** | **\< 5秒** | 开启 CPU Boost 后 |

-----

## 11\. 结论

[cite\_start]本报告证实了在 CPU 受限环境下构建实时多模态情感分析系统的可行性 [cite: 204][cite\_start]。核心策略在于 **“算法瘦身”与“架构异步”** [cite: 204]。

[cite\_start]通过结合 **Faster-Whisper Int8**、**ONNX Runtime** 加速的 **DistilBERT** 以及 **Silero VAD**，配合 **晚期融合** 策略，我们可以在不依赖 GPU 的情况下实现具备商业可用性的准实时服务 [cite: 204][cite\_start]。工程实施的关键在于 Docker 镜像瘦身、正确的 PyTorch CPU 版本安装以及精细的线程池管理 [cite: 205]。

-----
### 阶段 1: 基础架构与数据摄取 (SWE+VAD)

[cite\_start]**阶段目标**: 构建能够处理 WebSocket 连接的高性能异步后端，并集成静音检测 (VAD) 以过滤无效音频，建立高效的音频缓冲机制 [cite: 276]。

#### 1.1 核心指标与资源

| 字段 | 描述 |
| :--- | :--- |
| **验收标准** | 1. [cite\_start]WebSocket 服务能稳定接收二进制音频流 [cite: 278]。<br>2. [cite\_start]VAD 模块能准确区分语音和静音 (准确率\>90%) [cite: 279]。<br>3. [cite\_start]系统能正确切分“语音段”并放入队列，且 CPU 占用率 \<10% [cite: 280]。 |
| **关键技术栈** | [cite\_start]Python 3.10+, FastAPI, WebSockets, Silero VAD (ONNX), NumPy [cite: 281]。 |
| **核心挑战** | [cite\_start]缓冲区的内存管理。Python 的 list 扩容和字节拼接 (bytes concatenation) 非常耗时，需使用预分配 buffer 或 `io.BytesIO` [cite: 281]。 |
| **所需资源** | [cite\_start]Silero VAD ONNX 模型文件 (Github Link) [cite: 281]。 |

#### 1.2 实施步骤

1.  **初始化 FastAPI 项目结构**:
      * [cite\_start]创建 `app/main.py`, 定义 WebSocket 路由 `/ws/analyze` [cite: 284]。
      * [cite\_start]配置 `uvicorn` 启动参数，设置 `log_level="info"` 以便于调试 [cite: 285]。
2.  **实现环形缓冲区 (Ring Buffer)**:
      * [cite\_start]创建一个 `AudioBuffer` 类，使用 `bytearray` 存储原始 PCM 数据（推荐格式: 16kHz, Mono, Int16）[cite: 287]。
      * [cite\_start]实现 `append(chunk)` 和 `get_window(size)` 方法，避免频繁内存分配 [cite: 288]。
3.  **集成 Silero VAD (ONNX)**:
      * [cite\_start]下载 `silero_vad.onnx` [cite: 290]。
      * [cite\_start]使用 `onnxruntime` 加载模型 [cite: 291]。
      * [cite\_start]编写 `VADIterator` 类，处理 30ms-50ms 的音频窗，维护 VAD 的内部状态 (Stateful)，输出语音概率 [cite: 292, 293]。
4.  **开发分段逻辑 (Endpointing)**:
      * [cite\_start]在 WebSocket 循环中，当 VAD 检测到连续 N 个静音帧（例如持续 500ms）时，触发“句子结束”事件 [cite: 295]。
      * [cite\_start]将这段完整的语音数据打包，放入 `asyncio.Queue` 供后续处理 [cite: 296]。
5.  **基准测试**:
      * [cite\_start]使用 `wscat` 或编写简单的 Python 脚本模拟客户端发送音频流，测量 VAD 的处理延迟（目标: 每帧 \<1ms）[cite: 298]。

-----

### 阶段 2: 声学情感分析管道 (ML - Acoustic)

[cite\_start]**阶段目标**: 开发并部署一个轻量级的声学模型，仅根据声音的物理特征（语调、语速、能量）判断情感，不依赖文本内容 [cite: 300]。

#### 2.1 核心指标与资源

| 字段 | 描述 |
| :--- | :--- |
| **验收标准** | 1. [cite\_start]能够从原始音频中实时提取 MFCC 特征 [cite: 301]。<br>2. [cite\_start]声学模型推理延迟 \<20ms (CPU) [cite: 301]。<br>3. [cite\_start]模型在验证集上准确率达到基准线（约 60%-70% 即可，声学单模态通常较低）[cite: 301]。 |
| **关键技术栈** | [cite\_start]Torchaudio (C++加速 MFCC), ONNX Runtime, PyTorch (仅训练用), Scikit-learn [cite: 301]。 |
| **核心挑战** | [cite\_start]特征计算的实时性。MFCC 计算必须在 CPU 上极快。避免使用 `librosa` 进行实时推理（其依赖较重），推荐 `torchaudio` 或 `python_speech_features` [cite: 301]。 |
| **所需资源** | [cite\_start]数据集: RAVDESS (Speech Audio only) 或 TESS [cite: 301]。 |

#### 2.2 实施步骤

1.  **数据准备与预处理**:
      * [cite\_start]下载 RAVDESS 数据集 [cite: 305]。
      * [cite\_start]编写脚本将所有音频重采样为 16kHz 单声道 [cite: 306]。
      * [cite\_start]提取 MFCC 特征（建议参数: `n_mfcc=40`, `n_fft=400`, `hop_length=160`）[cite: 307]。
2.  **训练轻量级 1D-CNN 模型**:
      * [cite\_start]使用 PyTorch 定义一个简单的模型: `Conv1d (40, 64) -> ReLU -> MaxPool -> Conv1d (64, 128) -> GlobalAvgPool -> FC(3)` (Pos, Neg, Neu) [cite: 309]。
      * [cite\_start]训练模型，保存为 `.pt` 文件 [cite: 310]。
3.  **模型导出与量化**:
      * [cite\_start]使用 `torch.onnx.export` 将模型导出为 ONNX 格式 [cite: 312]。
      * (可选) [cite\_start]使用 ONNX Runtime 的量化工具将其转换为 INT8 格式以进一步加速 [cite: 313]。
4.  **集成推理服务**:
      * [cite\_start]在 FastAPI 应用中创建一个独立的 `AcousticWorker` [cite: 315]。
      * [cite\_start]当接收到 VAD 确认的语音片段时，切片提取 MFCC 并送入 ONNX Session 推理 [cite: 316]。
      * [cite\_start]注意: MFCC 提取需包含异常处理（处理极短音频片段）[cite: 317]。

-----

### 阶段 3: 文本情感分析管道 (ML - Text & ASR)

[cite\_start]**阶段目标**: 集成 ASR 引擎将语音转为文本，并运行 NLP 模型分析文本语义情感。这是计算最密集的环节 [cite: 319]。

#### 3.1 核心指标与资源

| 字段 | 描述 |
| :--- | :--- |
| **验收标准** | 1. [cite\_start]ASR 能够在 \<500ms 内完成短句转录 [cite: 320]。<br>2. [cite\_start]文本情感模型推理延迟 \<50ms [cite: 320]。<br>3. [cite\_start]管道能处理并发请求而不阻塞 WebSocket 心跳 [cite: 320]。 |
| **关键技术栈** | [cite\_start]Faster-Whisper (Int8), DistilBERT (ONNX Quantized), Hugging Face Optimum [cite: 323]。 |
| **核心挑战** | [cite\_start]CPU 资源争夺。ASR 和 NLP 模型都会大量占用 CPU。必须限制线程数 (`OMP_NUM_THREADS=1`) 并使用 `run_in_executor` 避免阻塞主线程 [cite: 323]。 |
| **所需资源** | [cite\_start]预训练模型: `systran/faster-whisper-tiny` 或 `small`, `distilbert-base-uncased-finetuned-sst-2-english` [cite: 323]。 |

#### 3.2 实施步骤

1.  **ASR 引擎集成**:
      * [cite\_start]安装 `faster-whisper` [cite: 326]。
      * [cite\_start]加载 `tiny.en` 或 `small.en` 模型，设置 `compute_type="int8"` [cite: 327]。
      * [cite\_start]编写 `transcribe` 函数，接收 NumPy 数组音频，返回文本字符串 [cite: 328]。
2.  **NLP 模型优化 (Optimum)**:
      * [cite\_start]使用 HuggingFace `optimum` 库加载 DistilBERT 模型 [cite: 330]。
      * [cite\_start]执行动态量化 (Dynamic Quantization) 导出为 ONNX [cite: 331]。
      * [cite\_start]代码示例 [cite: 333-336]:
        ```python
        from optimum.onnxruntime import ORTModelForSequenceClassification
        model = ORTModelForSequenceClassification.from_pretrained("distilbert...", export=True)
        ```
3.  **异步管道编排**:
      * [cite\_start]在 FastAPI 中，当 VAD 判定句子结束[cite: 338]:
        1.  [cite\_start]`await loop.run_in_executor(pool, asr_model.transcribe, audio)` [cite: 340-341]。
        2.  [cite\_start]获取文本后 -\> `await loop.run_in_executor(pool, nlp_model.predict, text)` [cite: 342-343]。
4.  **错误处理**:
      * [cite\_start]处理 ASR 产生的“幻觉”文本（如静音段被转录为 "Thank you" 或重复字符）。设置简单的文本过滤器 [cite: 339, 344]。

-----

### 阶段 4: 融合、容器化与部署 (Fusion + Deploy)

[cite\_start]**阶段目标**: 将两个模态的结果融合，打包应用，并部署到云端无服务器环境 [cite: 346]。

#### 4.1 核心指标与资源

| 字段 | 描述 |
| :--- | :--- |
| **验收标准** | 1. [cite\_start]Docker 镜像体积 \<1.5GB (理想 \<800MB) [cite: 347]。<br>2. [cite\_start]Cloud Run 冷启动时间 \<5秒 [cite: 347]。<br>3. [cite\_start]系统能够返回包含 `text_sentiment`, `audio_sentiment` 和 `fused_score` 的 JSON [cite: 347]。 |
| **关键技术栈** | [cite\_start]Docker (Multi-stage), Google Cloud Run, Late Fusion Logic [cite: 347]。 |
| **核心挑战** | [cite\_start]Docker 镜像瘦身。PyTorch 默认安装 CUDA 版本，体积巨大。必须强制安装 CPU 版本 [cite: 347]。 |
| **所需资源** | [cite\_start]Google Cloud Platform 账号 (Cloud Run 服务) [cite: 347]。 |

#### 4.2 实施步骤

1.  **实现晚期融合策略**:
      * [cite\_start]编写融合函数: `final_score = w1 * text_prob + w2 * audio_prob`。建议 `w1=0.7` (文本权重)，`w2=0.3` (声学权重) [cite: 351]。
      * [cite\_start]设计最终 WebSocket 返回的 JSON 格式 [cite: 352]。
2.  **编写 Dockerfile**:
      * [cite\_start]使用 `python:3.10-slim` 作为基础镜像 [cite: 354]。
      * [cite\_start]关键步骤: 使用 `pip install torch --index-url https://download.pytorch.org/whl/cpu` 仅安装 CPU 版本 PyTorch [cite: 355, 356]。
      * [cite\_start]清理 apt 缓存和 pip 缓存 (`rm -rf /var/lib/apt/lists/*`) [cite: 357]。
3.  **本地集成测试**:
      * [cite\_start]使用 `docker build` 构建镜像 [cite: 359]。
      * [cite\_start]运行容器并使用测试脚本（发送一段 5 秒的悲伤语音）验证端到端流程 [cite: 360]。
4.  **Cloud Run 部署配置**:
      * [cite\_start]部署到 Google Cloud Run [cite: 362]。
      * **必选配置**:
          * [cite\_start]**CPU allocation**: "CPU is always allocated" (防止 WebSocket 断连) [cite: 364]。
          * [cite\_start]**Minimum instances**: 1 (若预算允许) 或开启 **Startup CPU Boost** (加速冷启动) [cite: 366]。
          * [cite\_start]**Concurrency**: 设置为 1 或 2 (保证每个请求获得足额 CPU 时间片) [cite: 367]。

#### 给开发者的特别提示

  * [cite\_start]**调试利器**: 在开发阶段，将 WebSocket 接收到的音频同时保存为 .wav 文件，方便回听排查 VAD 是否切分正确 [cite: 369]。
  * [cite\_start]**性能监控**: 在代码中加入 `time.perf_counter()` 记录 VAD 耗时、ASR 耗时、NLP 耗时，并在日志中打印，这对于后续优化至关重要 [cite: 370, 371]。
  * [cite\_start]**降级策略**: 如果 ASR 耗时过长 (\>1秒)，可以设计为先返回声学模型的情感结果（因为它很快），待 ASR 完成后再推送文本情感结果更新 [cite: 372]。