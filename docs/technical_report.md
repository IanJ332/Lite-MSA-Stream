# Lite-MSA-Stream Technical Report

## 1. Project Overview
**Goal**: Develop a real-time, multimodal sentiment analysis system capable of running on consumer hardware (CPU-optimized) with high accuracy and low latency.
**Core Challenge**: Balancing the trade-off between model complexity (accuracy) and inference speed (latency), while addressing domain generalization issues (theatrical vs. real-world data).

## 2. System Architecture

### 2.1 Acoustic Pipeline (The "Ears")
*   **Voice Activity Detection (VAD)**:
    *   **Model**: Silero VAD (v4, ONNX).
    *   **Role**: Filters out silence to reduce processing load.
    *   **Optimization**: Quantized ONNX model for <10ms inference on CPU.
*   **Feature Extraction (Ensemble)**:
    *   **Backbone 1**: `facebook/hubert-large-ls960-ft` (Fine-tuned on LibriSpeech). Captures phonetic and linguistic content.
    *   **Backbone 2**: `facebook/wav2vec2-large-robust`. Captures acoustic robustness against noise.
    *   **Fusion**: Concatenation of mean-pooled hidden states (1024 + 1024 = **2048 dimensions**).
*   **Emotion Classification**:
    *   **Model**: `FusionNet` (Custom MLP).
    *   **Structure**:
        *   Input: 2048 dim
        *   Hidden 1: 1024 dim + BatchNorm + ReLU + Dropout (0.4)
        *   Hidden 2: 512 dim + BatchNorm + ReLU + Dropout (0.4)
        *   Output: 6 classes (Softmax).
*   **Streaming Strategy**:
    *   **Method**: Sliding Window Buffer.
    *   **Window Size**: 2.0 seconds.
    *   **Step Size**: 1.0 second (50% overlap).
    *   **Benefit**: Provides "YOLO-style" continuous feedback without waiting for sentence completion.

### 2.2 Text Pipeline (The "Brain")
*   **Automatic Speech Recognition (ASR)**:
    *   **Model**: `FunASR / SenseVoiceSmall`.
    *   **Decoding**: Non-autoregressive (CTC-based).
    *   **Beam Search**: Configurable `beam_size` (Default: 1 for speed, can be increased for accuracy).
    *   **Optimization**: Supports quantization and ONNX export.
*   **Sentiment Analysis (Evolution)**:
    *   *Phase 1 (Initial)*: `TextBlob` (Polarity) + Keyword Matching (Bag of Words). Simple but lacked nuance.
    *   *Phase 2 (Current)*: **DistilRoberta (ONNX)**.
        *   **Model**: `j-hartmann/emotion-english-distilroberta-base`.
        *   **Mechanism**: Transformer-based classification of implicit emotions (e.g., "My blood is boiling" -> Anger).
        *   **Optimization**: Converted to ONNX for CPU inference (~30ms).

### 2.3 Multimodal Fusion
*   **Strategy**: Late Fusion (Decision Level).
*   **Logic**:
    *   Acoustic model provides the *base* emotion probability.
    *   Text model provides a *semantic bias*.
    *   **Fusion Rule**: If text confidence is high (>0.6), it boosts the corresponding acoustic emotion probability.
    *   **Correction**: Explicitly handles "Smiling Voice" (Happy tone, Sad text) and "Sarcasm" (Neutral tone, Angry text).

## 3. Datasets & Data Strategy

### 3.1 Datasets Used
1.  **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song):
    *   *Characteristics*: Studio quality, North American accent, highly theatrical/acted.
    *   *Role*: Initial training baseline.
2.  **TESS** (Toronto Emotional Speech Set):
    *   *Characteristics*: Older female speakers, high quality, clear emotion.
    *   *Role*: Supplementing female voice data.
3.  **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset):
    *   *Characteristics*: 91 actors, diverse accents, varying recording quality, "Wild/Real-world" style.
    *   *Role*: **Critical for Domain Generalization**. Used for validation and training robustness.

### 3.2 Data Preparation Strategy
*   **Unified Label Map**:
    *   Mapped all datasets to 6 universal emotions: **Neutral, Happy, Sad, Angry, Fearful, Disgusted**.
    *   Dropped "Surprised" (inconsistent across datasets) and "Calm" (merged into Neutral).
*   **Augmentation (The "Real-World"ifier)**:
    *   **Noise Injection**: Random Gaussian noise with SNR factor `U(0.001, 0.015)`.
    *   **Purpose**: Prevents the model from overfitting to "studio silence" and forces it to focus on voice characteristics.

## 4. Training & Optimization Findings

### 4.1 The "Overfitting" Discovery
*   **Observation**: Model trained *only* on RAVDESS achieved **87%** test accuracy on RAVDESS but **<40%** on CREMA-D.
*   **Diagnosis**: The model learned "RAVDESS Acting" (specific prosody/silence patterns) rather than generalized human emotion.
*   **Solution**: **Domain Generalization**.
    *   **Speaker-Independent Split**: Validation set consists of *unseen speakers* (Last 10 CREMA, Last 4 RAVDESS).
    *   **WeightedRandomSampler**: Strictly enforces balanced batches (e.g., 8 Angry, 8 Happy) to prevent class bias.

### 4.2 Key Hyperparameters
*   **Learning Rate**: `1e-4` (Fine-tuning requires lower LR).
*   **Scheduler**: `ReduceLROnPlateau` (Patience: 5 epochs, Factor: 0.5).
*   **Dropout**: Increased from `0.1` to `0.4` to combat the massive domain shift between datasets.
*   **Batch Size**: 32.

## 5. Summary of Achievements
1.  **Real-Time Performance**: Achieved <500ms end-to-end latency on CPU.
2.  **Robustness**: Transitioned from a fragile "Studio Model" to a robust "General Model" via multi-dataset training.
3.  **Semantic Intelligence**: Integrated LLM-grade text understanding (DistilRoberta) without the heavy compute cost (via ONNX).
