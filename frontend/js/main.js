// --- Configuration ---
const CONFIG = {
    wsUrl: 'ws://localhost:8000/ws/analyze',
    emotions: ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Neutral'], // Capitalized for consistency
    colors: {
        Happy: '#10B981',
        Sad: '#3B82F6',
        Angry: '#EF4444',
        Fearful: '#F59E0B',
        Disgusted: '#8B5CF6',
        Neutral: '#9CA3AF'
    }
};

// --- State ---
let state = {
    isConnected: false,
    isRecording: false,
    currentEmotions: { Happy: 0, Sad: 0, Angry: 0, Fearful: 0, Disgusted: 0, Neutral: 1 },
    targetColor: new THREE.Color(CONFIG.colors.Neutral),
    audioLevel: 0,
    isDark: true
};

// Audio Context State
let audioContext;
let processor;
let input;
let globalStream;
let ws;

// --- 1. Visualizer (Three.js) ---
const initVisualizer = () => {
    const container = document.getElementById('visualizer-container');
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Geometry - High detail
    const geometry = new THREE.IcosahedronGeometry(2.2, 30);

    // Custom Shader Material
    const material = new THREE.ShaderMaterial({
        uniforms: {
            uTime: { value: 0 },
            uColor: { value: new THREE.Color(CONFIG.colors.Neutral) },
            uAudioLevel: { value: 0.0 }
        },
        vertexShader: `
            uniform float uTime;
            uniform float uAudioLevel;
            varying vec3 vNormal;
            varying vec3 vPosition;

            // Simplex Noise
            vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
            vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
            float snoise(vec3 v) {
                const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
                const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
                vec3 i  = floor(v + dot(v, C.yyy) );
                vec3 x0 = v - i + dot(i, C.xxx) ;
                vec3 g = step(x0.yzx, x0.xyz);
                vec3 l = 1.0 - g;
                vec3 i1 = min( g.xyz, l.zxy );
                vec3 i2 = max( g.xyz, l.zxy );
                vec3 x1 = x0 - i1 + C.xxx;
                vec3 x2 = x0 - i2 + C.yyy;
                vec3 x3 = x0 - D.yyy;
                i = mod289(i);
                vec4 p = permute( permute( permute(
                        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                        + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
                        + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
                float n_ = 0.142857142857;
                vec3  ns = n_ * D.wyz - D.xzx;
                vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
                vec4 x_ = floor(j * ns.z);
                vec4 y_ = floor(j - 7.0 * x_ );
                vec4 x = x_ *ns.x + ns.yyyy;
                vec4 y = y_ *ns.x + ns.yyyy;
                vec4 h = 1.0 - abs(x) - abs(y);
                vec4 b0 = vec4( x.xy, y.xy );
                vec4 b1 = vec4( x.zw, y.zw );
                vec4 s0 = floor(b0)*2.0 + 1.0;
                vec4 s1 = floor(b1)*2.0 + 1.0;
                vec4 sh = -step(h, vec4(0.0));
                vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
                vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
                vec3 p0 = vec3(a0.xy,h.x);
                vec3 p1 = vec3(a0.zw,h.y);
                vec3 p2 = vec3(a1.xy,h.z);
                vec3 p3 = vec3(a1.zw,h.w);
                vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
                p0 *= norm.x;
                p1 *= norm.y;
                p2 *= norm.z;
                p3 *= norm.w;
                vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
                m = m * m;
                return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                            dot(p2,x2), dot(p3,x3) ) );
            }

            void main() {
                vNormal = normal;
                vPosition = position;
                
                // Displacement Logic
                float noise = snoise(position * 2.0 + uTime * 0.5);
                // Smoother, less spikey displacement
                float displacement = noise * (0.1 + uAudioLevel * 0.3);
                
                vec3 newPosition = position + normal * displacement;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
            }
        `,
        fragmentShader: `
            uniform vec3 uColor;
            varying vec3 vNormal;
            varying vec3 vPosition;

            void main() {
                // Fresnel Glow
                vec3 viewDirection = normalize(cameraPosition - vPosition);
                float fresnel = pow(1.0 - dot(viewDirection, vNormal), 3.0);
                
                // Liquid Core Look
                vec3 baseColor = uColor * 0.6;
                vec3 glowColor = uColor * 1.5;
                
                gl_FragColor = vec4(mix(baseColor, glowColor, fresnel), 0.85); 
            }
        `,
        transparent: true
    });

    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.y = 0.3; // Move up to avoid text overlap
    scene.add(sphere);
    camera.position.z = 4.5; // Slightly closer

    // Animation Loop
    const animate = () => {
        requestAnimationFrame(animate);

        material.uniforms.uTime.value += 0.01;

        // Smooth LERP for Color
        material.uniforms.uColor.value.lerp(state.targetColor, 0.05);

        // Smooth LERP for Audio Level
        material.uniforms.uAudioLevel.value += (state.audioLevel - material.uniforms.uAudioLevel.value) * 0.1;

        // Gentle Rotation
        sphere.rotation.y += 0.002;
        sphere.rotation.z += 0.001;

        renderer.render(scene, camera);
    };
    animate();

    // Handle Resize
    window.addEventListener('resize', () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    });
};

// --- 2. Radar Chart (Chart.js) ---
let radarChart;

const initRadar = () => {
    const ctx = document.getElementById('emotionRadar').getContext('2d');

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: CONFIG.emotions,
            datasets: [{
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderColor: '#ffffff',
                borderWidth: 2,
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { display: false }, // Hide default labels
                    ticks: { display: false, max: 1, min: 0 }
                }
            },
            plugins: { legend: { display: false } },
            animation: { duration: 500 }
        }
    });

    // Create HTML Labels
    const labelContainer = document.getElementById('radar-labels');
    CONFIG.emotions.forEach(emotion => {
        const div = document.createElement('div');
        div.className = 'radar-label';
        div.id = `label-${emotion}`;
        div.innerText = emotion;
        labelContainer.appendChild(div);
    });

    // Position Labels Loop
    const positionLabels = () => {
        if (!radarChart) return;

        const scale = radarChart.scales.r;
        const center = { x: radarChart.width / 2, y: radarChart.height / 2 };
        const radius = scale.drawingArea + 22; // Increased to prevent overlap with title

        CONFIG.emotions.forEach((emotion, i) => {
            const angle = scale.getIndexAngle(i) - Math.PI / 2;
            const x = center.x + Math.cos(angle) * radius;
            const y = center.y + Math.sin(angle) * radius;

            const label = document.getElementById(`label-${emotion}`);
            if (label) {
                label.style.left = `${x}px`;
                label.style.top = `${y}px`;
            }
        });
        requestAnimationFrame(positionLabels);
    };
    positionLabels();
};

// --- 3. UI Updates ---
const updateUI = (data) => {
    // Normalize data keys to Capitalized
    const normalizedEmotions = {};
    Object.keys(data.emotions).forEach(k => {
        const key = k.charAt(0).toUpperCase() + k.slice(1).toLowerCase();
        // Filter: Only allow emotions defined in CONFIG
        if (CONFIG.emotions.includes(key)) {
            normalizedEmotions[key] = data.emotions[k];
        }
    });
    data.emotions = normalizedEmotions;
    data.sentiment = data.sentiment.charAt(0).toUpperCase() + data.sentiment.slice(1).toLowerCase();

    // 1. Update Chart
    const emotionValues = CONFIG.emotions.map(e => data.emotions[e] || 0);
    radarChart.data.datasets[0].data = emotionValues;

    // Dynamic Border Color based on Top Emotion
    const topEmotion = data.sentiment;
    const topColor = CONFIG.colors[topEmotion] || CONFIG.colors.Neutral;
    radarChart.data.datasets[0].borderColor = topColor;
    radarChart.data.datasets[0].backgroundColor = topColor + '33'; // 20% opacity
    radarChart.update();

    // 2. Update Labels (Highlight Top 3)
    const sortedEmotions = Object.entries(data.emotions)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 3);

    document.querySelectorAll('.radar-label').forEach(el => el.classList.remove('highlight', 'text-shadow-glow'));

    sortedEmotions.forEach(([emotion, score]) => {
        const label = document.getElementById(`label-${emotion}`);
        if (label) {
            label.classList.add('highlight');
            label.style.color = CONFIG.colors[emotion];
            label.style.borderColor = CONFIG.colors[emotion];
        }
    });

    // 3. Update Visualizer State
    state.targetColor.set(topColor);
    state.audioLevel = data.confidence || 0.5;

    // 4. Update Scoreboard (SINGLE COLUMN, LARGE TEXT)
    const list = document.getElementById('top-emotions-list');
    list.className = 'space-y-4'; // Vertical stack, no grid
    list.innerHTML = '';

    sortedEmotions.forEach(([emotion, score], index) => {
        const color = CONFIG.colors[emotion];
        const width = (score * 100).toFixed(0) + '%';

        list.innerHTML += `
            <div class="flex flex-col gap-1">
                <div class="flex justify-between text-sm font-bold opacity-90">
                    <span style="color: ${color} !important">${emotion}</span>
                    <span>${width}</span>
                </div>
                <div class="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                    <div class="h-full rounded-full transition-all duration-500" style="width: ${width}; background-color: ${color}"></div>
                </div>
            </div>
        `;
    });

    // 5. Append Chat Log
    addLogEntry(data);
};

const addLogEntry = (data) => {
    const log = document.getElementById('chat-log');
    const color = CONFIG.colors[data.sentiment];

    const div = document.createElement('div');
    div.className = 'msg-card p-4 cursor-pointer group';
    div.style.borderLeftColor = color;

    // Distribution Bars for Expanded View (SINGLE COLUMN)
    let distBars = '';
    const sortedEmotions = Object.entries(data.emotions).sort(([, a], [, b]) => b - a);

    sortedEmotions.forEach(([e, score]) => {
        const val = (score * 100).toFixed(0);
        // Added color style to the emotion name
        distBars += `
            <div class="flex items-center gap-2 text-xs mb-2">
                <span class="w-20 text-right font-bold capitalize" style="color: ${CONFIG.colors[e]}">${e}</span>
                <div class="flex-1 h-1.5 bg-white/10 rounded-full">
                    <div class="h-full rounded-full" style="width: ${val}%; background-color: ${CONFIG.colors[e]}"></div>
                </div>
                <span class="w-10 opacity-70">${val}%</span>
            </div>
        `;
    });

    div.innerHTML = `
        <div class="flex justify-between items-start mb-2">
            <span class="text-xs font-mono opacity-50">${new Date().toLocaleTimeString()}</span>
            <span class="text-xs font-bold px-2 py-1 rounded bg-white/5 uppercase tracking-wider" style="color: ${color}">
                ${data.sentiment} ${(data.confidence * 100).toFixed(0)}%
            </span>
        </div>
        <p class="text-base leading-relaxed opacity-90">${data.transcription || "..."}</p>
        
        <!-- Expanded Details -->
        <div class="msg-details">
            <div class="text-xs uppercase tracking-widest opacity-50 mb-3 mt-2">Probability Distribution</div>
            <div class="grid grid-cols-2 gap-x-4 gap-y-1">
                ${distBars}
            </div>
        </div>
    `;

    // Click to Expand
    div.addEventListener('click', () => {
        div.classList.toggle('expanded');
    });

    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
};

// --- 4. Audio Capture & WebSocket ---

const startAudioCapture = async () => {
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
            if (state.isConnected && ws && ws.readyState === WebSocket.OPEN) {
                const inputData = e.inputBuffer.getChannelData(0);
                // Calculate rough volume for visualizer
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) sum += Math.abs(inputData[i]);
                // Boost gain for visual (Increased sensitivity as requested)
                state.audioLevel = sum / inputData.length * 15.0;

                ws.send(inputData.buffer);
            }
        };

        input.connect(processor);
        processor.connect(audioContext.destination);
        console.log("Audio capture started");

    } catch (err) {
        console.error("Microphone Error:", err);
        alert("Microphone access denied or failed.");
        stopSession();
    }
};

const stopAudioCapture = () => {
    if (globalStream) globalStream.getTracks().forEach(track => track.stop());
    if (processor) processor.disconnect();
    if (input) input.disconnect();
    if (audioContext) audioContext.close();
    console.log("Audio capture stopped");
};

const connectWebSocket = () => {
    ws = new WebSocket(CONFIG.wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log('Connected to WebSocket');
        state.isConnected = true;
        startAudioCapture(); // Start mic once connected
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'vad_update') {
            // Optional: Use VAD prob to pulse visualizer if needed
            // state.audioLevel = data.prob; 
        } else if (data.type === 'result') {
            updateUI(data);
        }
    };

    ws.onclose = (event) => {
        console.log(`WebSocket closed. Code: ${event.code}, Reason: ${event.reason}`);
        state.isConnected = false;
        stopAudioCapture();

        // Ensure UI resets
        stopSession();
    };

    ws.onerror = (err) => {
        console.error('WebSocket error', err);
        ws.close();
    };
};

const updateLiveStatus = (isLive) => {
    const dot = document.getElementById('live-dot');
    const text = document.getElementById('live-text');

    if (isLive) {
        dot.className = "w-1.5 h-1.5 rounded-full bg-happy animate-ping";
        text.className = "text-happy";
        text.innerText = "LIVE";
    } else {
        dot.className = "w-1.5 h-1.5 rounded-full bg-gray-500";
        text.className = "text-gray-500";
        text.innerText = "OFFLINE";
    }
};

const startSession = () => {
    if (state.isRecording) return;
    state.isRecording = true;

    const btn = document.getElementById('session-btn');
    btn.innerHTML = '<div class="w-2 h-2 bg-white rounded-full animate-pulse"></div> End Session';
    btn.classList.add('from-red-500', 'to-red-600');
    btn.classList.remove('from-green-500', 'to-green-600');

    updateLiveStatus(true);
    connectWebSocket();
};

const stopSession = () => {
    state.isRecording = false;

    const btn = document.getElementById('session-btn');
    btn.innerHTML = '<i class="fa-solid fa-play"></i> Start Analysis';
    btn.classList.remove('from-red-500', 'to-red-600');
    btn.classList.add('from-green-500', 'to-green-600');

    updateLiveStatus(false);
    if (ws) ws.close();
    stopAudioCapture();
};

// --- 5. Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initVisualizer();
    initRadar();

    // Theme Toggle
    const toggleBtn = document.getElementById('theme-toggle');
    toggleBtn.addEventListener('click', () => {
        state.isDark = !state.isDark;
        document.documentElement.setAttribute('data-theme', state.isDark ? 'dark' : 'light');
        const icon = toggleBtn.querySelector('i');
        icon.className = state.isDark ? 'fa-solid fa-moon' : 'fa-solid fa-sun';

        // Update radar chart colors
        if (radarChart) {
            const gridColor = state.isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
            const borderColor = state.isDark ? '#ffffff' : '#000000';
            radarChart.options.scales.r.angleLines.color = gridColor;
            radarChart.options.scales.r.grid.color = gridColor;
            radarChart.data.datasets[0].borderColor = borderColor;
            radarChart.data.datasets[0].pointBackgroundColor = borderColor;
            radarChart.data.datasets[0].pointBorderColor = borderColor;
            radarChart.data.datasets[0].pointHoverBackgroundColor = borderColor;
            radarChart.data.datasets[0].pointHoverBorderColor = borderColor;
            radarChart.data.datasets[0].backgroundColor = state.isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
            radarChart.update('none');
        }
    });

    // Session Control
    const sessionBtn = document.getElementById('session-btn');
    sessionBtn.addEventListener('click', () => {
        if (state.isRecording) {
            stopSession();
        } else {
            startSession();
        }
    });

    // Set initial button state
    stopSession();
});
