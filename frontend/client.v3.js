const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const indicator = document.getElementById('sentiment-indicator');
const sentimentText = document.getElementById('sentiment-text');
const logOutput = document.getElementById('log-output');
const topEmotionsList = document.getElementById('top-emotions');

// State
let lastLogDiv = null;
let lastSentiment = null;
let audioContext;
let processor;
let input;
let globalStream;
let websocket;
let emotionChart = null;

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

const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${protocol}//${window.location.host}/ws/analyze`;

const sensitivitySlider = document.getElementById('sensitivity-slider');
const sensitivityVal = document.getElementById('sensitivity-val');

// --- Radar Chart ---
function initChart() {
    const canvas = document.getElementById('emotionRadar');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    emotionChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Neutral'],
            datasets: [{
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(187, 134, 252, 0.2)',
                borderColor: '#bb86fc',
                pointBackgroundColor: '#bb86fc',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#bb86fc',
                borderWidth: 2
            }]
        },
        options: {
            animation: {
                duration: 500, // Smooth transition
                easing: 'easeOutQuart'
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { display: false }, // Hide labels on chart
                    ticks: { display: false }, // Hide numbers
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            },
            plugins: {
                legend: { display: false }
            }

    // Sort emotions by value
    const sorted = Object.entries(emotions)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 3); // Top 3

            topEmotionsList.innerHTML = sorted.map(([emo, val]) => `
        <div class="emotion-row">
            <span class="label">${emotionEmojis[emo] || ''} ${emo.toUpperCase()}</span>
            <span class="value">${(val * 100).toFixed(0)}%</span>
        </div>
    `).join('');
        }

// --- Logging (Debounce & Append) ---
function log(msg) {
        const div = document.createElement('div');
        div.className = 'log-entry system';
        div.innerHTML = `<div class="text">${msg}</div>`;
        logOutput.prepend(div); // Newest at bottom (flex-col-reverse)
        lastLogDiv = null;
    }

function appendLog(msg, sentiment, emoji) {
            // If sentiment is same as last one, append to same bubble
            if (lastLogDiv && lastSentiment === sentiment) {
                const textSpan = lastLogDiv.querySelector('.text');
                // Simple debounce: don't repeat exact same text if sent twice
                if (!textSpan.innerText.endsWith(msg)) {
                    textSpan.innerText += " " + msg;
                }
            } else {
                // Create new bubble
                const div = document.createElement('div');
                div.className = `log-entry ${sentiment}`;
                div.innerHTML = `
            <div class="meta">${emoji} ${sentiment.toUpperCase()}</div>
            <div class="text">${msg}</div>
        `;
                logOutput.prepend(div); // Flex-col-reverse handles order
                lastLogDiv = div;
                lastSentiment = sentiment;
            }
        }

// --- UI Updates ---
function updateUIState(isRecording) {
            startBtn.disabled = isRecording;
            stopBtn.disabled = !isRecording;
            if (!isRecording) {
                if (indicator) {
                    indicator.className = "indicator neutral";
                    indicator.style.transform = "scale(1)";
                    indicator.style.boxShadow = "none";
                }
                if (sentimentText) sentimentText.innerText = "READY";
            }
        }

function handleResult(data) {
            if (data.type === 'vad_update') {
                const prob = data.prob;
                const triggered = data.triggered;

                if (indicator) {
                    if (prob > 0.5 || triggered) {
                        indicator.style.transform = `scale(${1 + prob * 0.1})`;
                        indicator.style.boxShadow = `0 0 ${prob * 20}px var(--accent-color)`;
                        sentimentText.innerText = triggered ? "LISTENING" : "READY";
                    } else {
                        indicator.style.transform = "scale(1)";
                        indicator.style.boxShadow = "none";
                    }
                }
                return;
            }

            if (data.type === 'result') {
                const sentiment = data.sentiment;
                const transcript = data.transcription;
                const emotions = data.emotions;

                // Update Indicator
                if (indicator) {
                    indicator.className = `indicator ${sentiment}`;
                    const emoji = emotionEmojis[sentiment] || '😐';
                    sentimentText.innerText = `${emoji} ${sentiment.toUpperCase()}`;
                }

                // Update Radar & Top 3
                if (emotions) {
                    updateChart(emotions);
                }

                // Log Transcript
                if (transcript) {
                    appendLog(transcript, sentiment, emotionEmojis[sentiment] || '😐');
                }
            }
        }

// --- Audio & WebSocket ---
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
                processor = audioContext.createScriptProcessor(512, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        websocket.send(inputData.buffer);
                    }
                };

                input.connect(processor);
                processor.connect(audioContext.destination);

                log("Recording started...");

            } catch (err) {
                console.error("Microphone Error:", err);
                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    log("Error: Microphone access denied.");
                } else {
                    log(`Error: Could not access microphone (${err.message}).`);
                }
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

async function startRecording() {
            try {
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
                    updateUIState(false);
                };

                updateUIState(true);

            } catch (err) {
                console.error("Error starting:", err);
                log("Error: " + err.message);
            }
        }

// --- Initialization ---
startBtn.addEventListener('click', startRecording);
