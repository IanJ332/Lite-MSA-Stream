const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const indicator = document.getElementById('sentiment-indicator');
const sentimentText = document.getElementById('sentiment-text');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceText = document.getElementById('confidence-text');
const logOutput = document.getElementById('log-output');

let audioContext;
let processor;
let input;
let globalStream;
let websocket;

const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${protocol}//${window.location.host}/ws/analyze`;

const sensitivitySlider = document.getElementById('sensitivity-slider');
const sensitivityVal = document.getElementById('sensitivity-val');

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

sensitivitySlider.addEventListener('input', () => {
    const val = sensitivitySlider.value;
    sensitivityVal.innerText = `${val}%`;

    // Calculate Threshold: Higher Sensitivity = Lower Threshold
    // 100% -> 0.0 (Most Sensitive), 0% -> 1.0 (Least Sensitive)
    const threshold = 1.0 - (val / 100);

    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            type: "config",
            vad_threshold: threshold
        }));
    }
});

async function startRecording() {
    try {
        // 1. Connect WebSocket
        websocket = new WebSocket(WS_URL);

        websocket.onopen = () => {
            console.log("WebSocket Connected");
            startAudioCapture();
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleResult(data);
        };

        websocket.onerror = (error) => {
            console.error("WebSocket Error:", error);
            log("Error: WebSocket connection failed.");
        };

        websocket.onclose = () => {
            console.log("WebSocket Closed");
            stopRecordingUI();
        };

        updateUIState(true);

    } catch (err) {
        console.error("Error starting:", err);
        log("Error: " + err.message);
    }
}

async function startAudioCapture() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
    });

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000
            }
        });

        globalStream = stream;
        input = audioContext.createMediaStreamSource(stream);

        // Buffer Size 512 -> 32ms latency (Standard for Silero VAD)
        processor = audioContext.createScriptProcessor(512, 1, 1);

        processor.onaudioprocess = (e) => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                const inputData = e.inputBuffer.getChannelData(0);

                // Send Float32 directly (required for accurate backend emotion recognition on CPU)
                websocket.send(inputData.buffer);
            }
        };

        input.connect(processor);
        processor.connect(audioContext.destination);

        log("Recording started...");

    } catch (err) {
        console.error("Microphone Error:", err);
        log("Error: Microphone access denied.");
        stopRecording();
    }
}

function stopRecording() {
    if (websocket) websocket.close();
    if (globalStream) globalStream.getTracks().forEach(track => track.stop());
    if (processor) processor.disconnect();
    if (input) input.disconnect();
    if (audioContext) audioContext.close();

    updateUIState(false);
    log("Recording stopped.");
}

function stopRecordingUI() {
    updateUIState(false);
}

function updateUIState(isRecording) {
    startBtn.disabled = isRecording;
    stopBtn.disabled = !isRecording;
    if (!isRecording) {
        indicator.className = "indicator neutral";
        indicator.style.transform = "scale(1)";
        indicator.style.boxShadow = "none";
        sentimentText.innerText = "Ready";
        confidenceFill.style.width = "0%";
        confidenceText.innerText = "Confidence: 0%";
    }
}

const emotionEmojis = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😡',
    'fearful': '😨',
    'disgusted': '🤢',
    'neutral': '😐',
    'surprised': '😲',
    'calm': '😌'
};

function handleResult(data) {
    if (data.type === 'vad_update') {
        const prob = data.prob;
        const triggered = data.triggered;

        // Visual Feedback for VAD
        if (prob > 0.5 || triggered) {
            indicator.style.transform = `scale(${1 + prob * 0.2})`;
            indicator.style.boxShadow = `0 0 ${prob * 20}px var(--accent-color)`;
            sentimentText.innerText = triggered ? "Listening..." : "Ready";
        } else {
            indicator.style.transform = "scale(1)";
            indicator.style.boxShadow = "none";
        }
        return;
    }

    if (data.type === 'result') {
        const sentiment = data.sentiment;
        const confidence = data.confidence;
        const transcript = data.transcription;

        // Update Indicator
        indicator.className = `indicator ${sentiment}`;
        const emoji = emotionEmojis[sentiment] || '😐';
        sentimentText.innerText = `${emoji} ${sentiment.toUpperCase()}`;

        // Update Bar
        const pct = Math.round(confidence * 100);
        confidenceFill.style.width = `${pct}%`;
        confidenceFill.style.backgroundColor = getColorForSentiment(sentiment);
        confidenceText.innerText = `Confidence: ${pct}%`;

        // Log Transcript (Streaming Append)
        if (transcript) {
            appendLog(transcript, sentiment, emoji);
        }
    }
}

function getColorForSentiment(sentiment) {
    switch (sentiment) {
        case 'happy':
        case 'surprised':
        case 'positive':
            return 'var(--positive-color)';

        case 'sad':
        case 'angry':
        case 'fearful':
        case 'disgusted':
        case 'negative':
            return 'var(--negative-color)';

        case 'neutral':
        default:
            return 'var(--neutral-color)';
    }
}

let lastLogDiv = null;
let lastSentiment = null;

function appendLog(msg, sentiment, emoji) {
    // If sentiment is same as last one, append to same bubble
    if (lastLogDiv && lastSentiment === sentiment) {
        const textSpan = lastLogDiv.querySelector('.text');
        textSpan.innerText += " " + msg;
        // Scroll to bottom of this div if needed, or just let it grow
    } else {
        // Create new bubble
        const div = document.createElement('div');
        div.className = `log-entry ${sentiment}`;
        div.innerHTML = `
            <div class="meta">${emoji} ${sentiment.toUpperCase()}</div>
            <div class="text">${msg}</div>
        `;
        logOutput.prepend(div); // Or append if we want chat style, but prepend matches current "latest top"
        lastLogDiv = div;
        lastSentiment = sentiment;
    }
}

function log(msg) {
    // System messages
    const div = document.createElement('div');
    div.className = 'log-entry system';
    div.innerHTML = `<div class="meta">SYSTEM</div><div class="text">${msg}</div>`;
    logOutput.prepend(div);
    lastLogDiv = null; // Break the chain
}
