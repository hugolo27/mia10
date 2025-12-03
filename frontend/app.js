class VoiceRecognizer {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.sampleCount = 0;
        this.metricsInterval = null;

        this.initializeElements();
        this.setupEventListeners();
        this.initializeAudio();
        this.startAutoMetricsUpdate();
    }

    initializeElements() {
        this.elements = {
            userName: document.getElementById('userName'),
            wordSelect: document.getElementById('wordSelect'),
            trainRecord: document.getElementById('trainRecord'),
            trainStop: document.getElementById('trainStop'),
            testRecord: document.getElementById('testRecord'),
            testStop: document.getElementById('testStop'),
            sampleCount: document.getElementById('sampleCount'),
            predictionResult: document.getElementById('predictionResult'),
            fftCanvas: document.getElementById('fftCanvas'),
            mfccCanvas: document.getElementById('mfccCanvas'),
            dmdCanvas: document.getElementById('dmdCanvas'),
            spectrogramCanvas: document.getElementById('spectrogramCanvas'),
            // Elementos de métricas
            trainingMetrics: document.getElementById('trainingMetrics'),
            predictionMetrics: document.getElementById('predictionMetrics'),
            avgSpeed: document.getElementById('avgSpeed'),
            lastPredTime: document.getElementById('lastPredTime'),
            userAccuracy: document.getElementById('userAccuracy'),
            wordAccuracy: document.getElementById('wordAccuracy'),
            totalSamples: document.getElementById('totalSamples'),
            modelStatus: document.getElementById('modelStatus'),
            evaluationMethod: document.getElementById('evaluationMethod')
        };
    }

    setupEventListeners() {
        this.elements.trainRecord.addEventListener('click', () => this.startTrainingRecording());
        this.elements.trainStop.addEventListener('click', () => this.stopRecording('training'));
        this.elements.testRecord.addEventListener('click', () => this.startTestRecording());
        this.elements.testStop.addEventListener('click', () => this.stopRecording('test'));
    }

    async initializeAudio() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 44100,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.stream = stream;

            console.log('✅ Audio initialized successfully');
        } catch (error) {
            console.error('❌ Error accessing microphone:', error);
            this.showError('No se pudo acceder al micrófono. Por favor, permite el acceso.');
        }
    }

    startTrainingRecording() {
        const userName = this.elements.userName.value.trim();
        const word = this.elements.wordSelect.value;

        if (!userName) {
            this.showError('Por favor ingresa tu nombre');
            return;
        }

        this.startRecording('training', { userName, word });
    }

    startTestRecording() {
        this.startRecording('test');
    }

    async startRecording(mode, metadata = {}) {
        if (this.isRecording) return;

        try {
            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.processRecording(mode, metadata);
            };

            this.mediaRecorder.start();
            this.isRecording = true;

            if (mode === 'training') {
                this.elements.trainRecord.disabled = true;
                this.elements.trainStop.disabled = false;
                this.elements.trainRecord.textContent = '● Grabando...';
            } else {
                this.elements.testRecord.disabled = true;
                this.elements.testStop.disabled = false;
                this.elements.testRecord.textContent = '▶ Analizando...';
            }

            console.log(`📹 Started ${mode} recording`);
        } catch (error) {
            console.error('❌ Recording error:', error);
            this.showError('Error al iniciar la grabación');
        }
    }

    stopRecording(mode) {
        if (!this.isRecording || !this.mediaRecorder) return;

        this.mediaRecorder.stop();
        this.isRecording = false;

        if (mode === 'training') {
            this.elements.trainRecord.disabled = false;
            this.elements.trainStop.disabled = true;
            this.elements.trainRecord.textContent = '● Iniciar Grabación';
        } else {
            this.elements.testRecord.disabled = false;
            this.elements.testStop.disabled = true;
            this.elements.testRecord.textContent = '▶ Probar Reconocimiento';
        }

        console.log(`⏹️ Stopped ${mode} recording`);
    }

    async processRecording(mode, metadata = {}) {
        try {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(audioBlob);

            // Convert to WAV for processing
            const audioBuffer = await this.convertToWAV(audioBlob);

            if (mode === 'training') {
                await this.sendTrainingData(audioBuffer, metadata);
                this.sampleCount++;
                this.elements.sampleCount.textContent = this.sampleCount;
                this.showSuccess(`✅ Muestra de "${metadata.word}" guardada`);
            } else {
                await this.sendTestData(audioBuffer);
            }

        } catch (error) {
            console.error('❌ Processing error:', error);
            this.showError('Error al procesar la grabación');
        }
    }

    async convertToWAV(audioBlob) {
        return new Promise((resolve, reject) => {
            const fileReader = new FileReader();
            fileReader.onload = async () => {
                try {
                    const arrayBuffer = fileReader.result;
                    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

                    // Extract audio data
                    const audioData = audioBuffer.getChannelData(0);
                    const sampleRate = audioBuffer.sampleRate;

                    resolve({
                        data: Array.from(audioData),
                        sampleRate: sampleRate,
                        duration: audioBuffer.duration
                    });
                } catch (error) {
                    reject(error);
                }
            };
            fileReader.onerror = reject;
            fileReader.readAsArrayBuffer(audioBlob);
        });
    }

    async sendTrainingData(audioBuffer, metadata) {
        try {
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    audio_data: audioBuffer.data,
                    sample_rate: audioBuffer.sampleRate,
                    user_name: metadata.userName,
                    word: metadata.word,
                    timestamp: new Date().toISOString()
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ Training data sent:', result);

            // Update visualizations if available
            if (result.analysis) {
                this.updateVisualizations(result.analysis);
            }

            // Show training metrics
            if (result.metricas_rendimiento) {
                this.displayTrainingMetrics(result.metricas_rendimiento);
            }

            // Trigger immediate metrics update after training
            this.refreshSystemMetrics();

        } catch (error) {
            console.error('❌ Training request failed:', error);
            this.showError('Error al enviar datos de entrenamiento');
        }
    }

    async sendTestData(audioBuffer) {
        try {
            this.showLoading('Analizando audio...');

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    audio_data: audioBuffer.data,
                    sample_rate: audioBuffer.sampleRate,
                    timestamp: new Date().toISOString()
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ Prediction result:', result);

            this.displayPrediction(result);

            // Update visualizations
            if (result.analysis) {
                this.updateVisualizations(result.analysis);
            }

            // Show prediction metrics
            if (result.metricas_rendimiento) {
                this.displayPredictionMetrics(result.metricas_rendimiento);
                this.updateSpeedMetrics(result.metricas_rendimiento);
            }

        } catch (error) {
            console.error('❌ Prediction request failed:', error);
            this.showError('Error al analizar la grabación');
        }
    }

    displayPrediction(result) {
        const resultBox = this.elements.predictionResult;

        if (result.success) {
            resultBox.className = 'result-box success';
            resultBox.innerHTML = `
                <div>
                    <strong>👤 Usuario:</strong> ${result.predicted_user}<br>
                    <strong>💬 Palabra:</strong> ${result.predicted_word}<br>
                    <strong>🎯 Confianza:</strong> ${(result.confidence * 100).toFixed(1)}%
                </div>
            `;
        } else {
            resultBox.className = 'result-box error';
            resultBox.textContent = result.message || 'No se pudo reconocer la voz';
        }
    }

    updateVisualizations(analysis) {
        // Update FFT visualization
        if (analysis.fft_data) {
            this.drawFFT(analysis.fft_data);
        }

        // Update MFCC visualization
        if (analysis.mfcc_data) {
            this.drawMFCC(analysis.mfcc_data);
        }

        // Update DMD visualization
        if (analysis.dmd_data) {
            this.drawDMD(analysis.dmd_data);
        }

        // Update Spectrogram
        if (analysis.spectrogram_data) {
            this.drawSpectrogram(analysis.spectrogram_data);
        }
    }

    drawFFT(fftData) {
        const canvas = this.elements.fftCanvas;
        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw FFT spectrum
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const step = canvas.width / fftData.length;
        const maxVal = Math.max(...fftData);

        for (let i = 0; i < fftData.length; i++) {
            const x = i * step;
            const y = canvas.height - (fftData[i] / maxVal) * canvas.height * 0.8;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.stroke();
    }

    drawMFCC(mfccData) {
        const canvas = this.elements.mfccCanvas;
        const ctx = canvas.getContext('2d');

        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Simple MFCC coefficient visualization
        const barWidth = canvas.width / mfccData.length;
        const maxVal = Math.max(...mfccData.map(Math.abs));

        for (let i = 0; i < mfccData.length; i++) {
            const barHeight = (Math.abs(mfccData[i]) / maxVal) * canvas.height * 0.8;
            const x = i * barWidth;
            const y = canvas.height - barHeight;

            ctx.fillStyle = mfccData[i] >= 0 ? '#27ae60' : '#e74c3c';
            ctx.fillRect(x, y, barWidth * 0.8, barHeight);
        }
    }

    drawDMD(dmdData) {
        const canvas = this.elements.dmdCanvas;
        const ctx = canvas.getContext('2d');

        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw DMD modes
        if (dmdData.modes && dmdData.frequencies) {
            const barWidth = canvas.width / dmdData.frequencies.length;

            for (let i = 0; i < dmdData.frequencies.length; i++) {
                const freq = Math.abs(dmdData.frequencies[i]);
                const mode = Math.abs(dmdData.modes[i]) || 0;

                const barHeight = mode * canvas.height * 0.8;
                const x = i * barWidth;
                const y = canvas.height - barHeight;

                ctx.fillStyle = `hsl(${freq * 360}, 70%, 50%)`;
                ctx.fillRect(x, y, barWidth * 0.8, barHeight);
            }
        }
    }

    drawSpectrogram(spectrogramData) {
        const canvas = this.elements.spectrogramCanvas;
        const ctx = canvas.getContext('2d');

        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Simple spectrogram representation
        if (spectrogramData && spectrogramData.length > 0) {
            const timeSteps = spectrogramData.length;
            const freqBins = spectrogramData[0].length;
            const cellWidth = canvas.width / timeSteps;
            const cellHeight = canvas.height / freqBins;

            for (let t = 0; t < timeSteps; t++) {
                for (let f = 0; f < freqBins; f++) {
                    const intensity = spectrogramData[t][f];
                    const color = Math.floor(intensity * 255);

                    ctx.fillStyle = `rgb(${color}, ${color/2}, ${255-color})`;
                    ctx.fillRect(t * cellWidth, (freqBins - f - 1) * cellHeight, cellWidth, cellHeight);
                }
            }
        }
    }

    showError(message) {
        const resultBox = this.elements.predictionResult;
        resultBox.className = 'result-box error';
        resultBox.textContent = `❌ ${message}`;
    }

    showSuccess(message) {
        // Show training success in training status area instead of prediction area
        if (message.includes('Muestra')) {
            // This is a training success message
            const trainingStatus = document.querySelector('.training-status p');
            if (trainingStatus) {
                const currentText = trainingStatus.textContent;
                const countMatch = currentText.match(/(\d+)/);
                const count = countMatch ? parseInt(countMatch[1]) : 0;
                trainingStatus.innerHTML = `Muestras grabadas: <span id="sampleCount">${count}</span> - ${message}`;

                // Clear the message after 3 seconds
                setTimeout(() => {
                    trainingStatus.innerHTML = `Muestras grabadas: <span id="sampleCount">${count}</span>`;
                    this.elements.sampleCount = document.getElementById('sampleCount');
                }, 3000);
            }
        } else {
            // This is a prediction success message
            const resultBox = this.elements.predictionResult;
            resultBox.className = 'result-box success';
            resultBox.textContent = message;
        }
    }

    showLoading(message) {
        const resultBox = this.elements.predictionResult;
        resultBox.className = 'result-box';
        resultBox.textContent = `⏳ ${message}`;
    }

    displayTrainingMetrics(metrics) {
        this.elements.trainingMetrics.innerHTML = `
            <div class="metrics-summary">
                <small>
                    ⚡ ${metrics.tiempo_caracteristicas_ms}ms |
                    🎵 ${metrics.audio_duracion_s}s |
                    📊 ${metrics.total_caracteristicas} características
                </small>
            </div>
        `;
    }

    displayPredictionMetrics(metrics) {
        this.elements.predictionMetrics.innerHTML = `
            <div class="metrics-summary">
                <small>
                    ⚡ Total: ${metrics.tiempo_total_ms}ms |
                    🔍 Predicción: ${metrics.tiempo_prediccion_ms}ms |
                    📈 Velocidad: ${metrics.velocidad_procesamiento}x tiempo real
                </small>
            </div>
        `;
    }

    updateSpeedMetrics(metrics) {
        if (this.elements.lastPredTime) {
            this.elements.lastPredTime.textContent = `${metrics.tiempo_total_ms}`;
        }

        // Calcular velocidad promedio simple
        if (!this.speedHistory) this.speedHistory = [];
        this.speedHistory.push(metrics.tiempo_total_ms);
        if (this.speedHistory.length > 10) this.speedHistory.shift();

        const avgSpeed = this.speedHistory.reduce((a, b) => a + b, 0) / this.speedHistory.length;
        if (this.elements.avgSpeed) {
            this.elements.avgSpeed.textContent = `${avgSpeed.toFixed(1)}`;
        }
    }

    async refreshSystemMetrics() {
        try {
            const response = await fetch('/status');
            const status = await response.json();

            if (status.success) {
                if (this.elements.totalSamples) {
                    this.elements.totalSamples.textContent = status.total_samples;
                }

                if (this.elements.modelStatus) {
                    this.elements.modelStatus.textContent = status.classifier_trained ? 'Entrenado' : 'No entrenado';
                }

                // Update accuracy if available
                if (status.metricas_basicas) {
                    if (this.elements.userAccuracy) {
                        this.elements.userAccuracy.textContent = status.metricas_basicas.precision_usuarios_pct || '--';
                    }
                    if (this.elements.wordAccuracy) {
                        this.elements.wordAccuracy.textContent = status.metricas_basicas.precision_palabras_pct || '--';
                    }
                    if (this.elements.evaluationMethod && status.metricas_basicas.metodo_evaluacion) {
                        const method = status.metricas_basicas.metodo_evaluacion;
                        let methodText = '';
                        switch(method) {
                            case 'cross_validation':
                                methodText = 'Validación cruzada';
                                break;
                            case 'train_penalized':
                                methodText = 'Entrenamiento penalizado';
                                break;
                            case 'single_user':
                                methodText = 'Usuario único';
                                break;
                            default:
                                methodText = method;
                        }
                        this.elements.evaluationMethod.textContent = methodText;
                    }
                }
            }
        } catch (error) {
            console.error('Error refreshing metrics:', error);
        }
    }

    startAutoMetricsUpdate() {
        // Update metrics every 5 seconds
        this.metricsInterval = setInterval(() => {
            this.refreshSystemMetrics();
        }, 5000);

        console.log('🔄 Auto-update de métricas iniciado (cada 5 segundos)');
    }

    stopAutoMetricsUpdate() {
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
            this.metricsInterval = null;
            console.log('⏹️ Auto-update de métricas detenido');
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.voiceRecognizer = new VoiceRecognizer();
    console.log('🚀 Voice Recognizer initialized');

    // Load initial metrics
    setTimeout(() => {
        window.voiceRecognizer.refreshSystemMetrics();
    }, 1000);

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (window.voiceRecognizer) {
            window.voiceRecognizer.stopAutoMetricsUpdate();
        }
    });
});