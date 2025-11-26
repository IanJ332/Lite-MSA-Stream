// --- Configuration ---
const CONFIG = {
    wsUrl: 'ws://localhost:8000/ws/analyze',
    emotions: ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'neutral'],
    colors: {
        happy: '#10B981',
        sad: '#3B82F6',
        angry: '#EF4444',
        fearful: '#F59E0B',
        disgusted: '#8B5CF6',
        neutral: '#9CA3AF'
    }
};

// --- State ---
let state = {
    isConnected: false,
    isRecording: false,
    currentEmotions: { happy: 0, sad: 0, angry: 0, fearful: 0, disgusted: 0, neutral: 1 },
    targetColor: new THREE.Color(CONFIG.colors.neutral),
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
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Geometry
    const geometry = new THREE.IcosahedronGeometry(1.8, 10); // High detail sphere

    // Custom Shader Material
    const material = new THREE.ShaderMaterial({
        uniforms: {
            uTime: { value: 0 },
            uColor: { value: new THREE.Color(CONFIG.colors.neutral) },
            uAudioLevel: { value: 0.0 }
        },
        vertexShader: `
            uniform float uTime;
            uniform float uAudioLevel;
            varying vec3 vNormal;
            varying vec3 vPosition;

            // Simplex Noise (Simplified)
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
                float displacement = noise * (0.2 + uAudioLevel * 0.5);
                
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
                
                gl_FragColor = vec4(mix(baseColor, glowColor, fresnel), 0.8); // Semi-transparent
            }
        `,
        transparent: true,
        // blending: THREE.AdditiveBlending // Optional: for more "plasma" look
    });

    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);
    camera.position.z = 5;

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
        const radius = scale.drawingArea + 20; // Push out slightly

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
    // 1. Update Chart
    const emotionValues = CONFIG.emotions.map(e => data.emotions[e] || 0);
    radarChart.data.datasets[0].data = emotionValues;

    // Dynamic Border Color based on Top Emotion
    const topEmotion = data.sentiment;
    const topColor = CONFIG.colors[topEmotion] || CONFIG.colors.neutral;
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
    // Use confidence or audio level if available, else simulate breathing
    state.audioLevel = data.confidence || 0.5;

    // 4. Update Scoreboard
    const list = document.getElementById('top-emotions-list');
    list.innerHTML = '';
    sortedEmotions.forEach(([emotion, score], index) => {
        const color = CONFIG.colors[emotion];
        const width = (score * 100).toFixed(0) + '%';

        list.innerHTML += `
            <div class="flex flex-col gap-1">
                <div class="flex justify-between text-[10px] uppercase font-bold text-white/60">
                    <span style="color: ${color}">${emotion}</span>
                    <span>${width}</span>
                </div>
                <div class="w-full h-1 bg-white/10 rounded-full overflow-hidden">
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

    // Distribution Bars for Expanded View
    let distBars = '';
    CONFIG.emotions.forEach(e => {
        const val = (data.emotions[e] || 0) * 100;
        distBars += `
            <div class="flex items-center gap-2 text-[10px] mb-1">
                <span class="w-16 text-right opacity-60 capitalize">${e}</span>
                <div class="flex-1 h-1 bg-white/10 rounded-full">
                    <div class="h-full rounded-full" style="width: ${val}%; background-color: ${CONFIG.colors[e]}"></div>
                </div>
                <span class="w-8 opacity-60">${val.toFixed(0)}%</span>
            </div>
        `;
    });

    div.innerHTML = `
        <div class="flex justify-between items-start mb-1">
            <span class="text-[10px] font-mono opacity-40">${new Date().toLocaleTimeString()}</span>
            <span class="text-[10px] font-bold px-2 py-0.5 rounded bg-white/5 uppercase tracking-wider" style="color: ${color}">
                ${data.sentiment} ${(data.confidence * 100).toFixed(0)}%
            </span>
        </div>
        <p class="text-sm leading-relaxed text-white/90">${data.transcription || "..."}</p>
        
        <!-- Expanded Details -->
        <div class="msg-details">
            <div class="text-[10px] uppercase tracking-widest opacity-40 mb-2">Probability Distribution</div>
            ${distBars}
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
                state.audioLevel = sum / inputData.length * 5.0; // Boost gain for visual

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
        const btn = document.getElementById('session-btn');
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Start Analysis';
        btn.classList.remove('from-red-500', 'to-red-600');
        btn.classList.add('from-green-500', 'to-green-600');
        state.isRecording = false;
    };

    ws.onerror = (err) => {
        console.error('WebSocket error', err);
        ws.close();
    };
};

const startSession = () => {
    if (state.isRecording) return;
    state.isRecording = true;

    const btn = document.getElementById('session-btn');
    btn.innerHTML = '<div class="w-2 h-2 bg-white rounded-full animate-pulse"></div> End Session';
    btn.classList.add('from-red-500', 'to-red-600');
    btn.classList.remove('from-green-500', 'to-green-600');

    connectWebSocket();
};

const stopSession = () => {
    state.isRecording = false;

    const btn = document.getElementById('session-btn');
    btn.innerHTML = '<i class="fa-solid fa-play"></i> Start Analysis';
    btn.classList.remove('from-red-500', 'to-red-600');
    btn.classList.add('from-green-500', 'to-green-600');

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

    // Auto-start (optional, or wait for user click)
    // startSession(); 
    // Set initial button state
    stopSession();
});
