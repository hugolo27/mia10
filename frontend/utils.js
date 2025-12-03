// Audio processing utilities

class AudioUtils {
    static calculateRMS(audioData) {
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        return Math.sqrt(sum / audioData.length);
    }

    static applyWindowFunction(audioData, windowType = 'hamming') {
        const windowed = new Array(audioData.length);
        const N = audioData.length;

        for (let i = 0; i < N; i++) {
            let windowValue = 1;

            switch (windowType) {
                case 'hamming':
                    windowValue = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (N - 1));
                    break;
                case 'hann':
                    windowValue = 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
                    break;
                case 'blackman':
                    windowValue = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)) +
                                 0.08 * Math.cos(4 * Math.PI * i / (N - 1));
                    break;
            }

            windowed[i] = audioData[i] * windowValue;
        }

        return windowed;
    }

    static preEmphasize(audioData, alpha = 0.97) {
        const preEmphasized = new Array(audioData.length);
        preEmphasized[0] = audioData[0];

        for (let i = 1; i < audioData.length; i++) {
            preEmphasized[i] = audioData[i] - alpha * audioData[i - 1];
        }

        return preEmphasized;
    }

    static frameSignal(audioData, frameSize, hopSize) {
        const frames = [];
        const numFrames = Math.floor((audioData.length - frameSize) / hopSize) + 1;

        for (let i = 0; i < numFrames; i++) {
            const frameStart = i * hopSize;
            const frame = audioData.slice(frameStart, frameStart + frameSize);
            frames.push(frame);
        }

        return frames;
    }

    static normalize(audioData) {
        const maxVal = Math.max(...audioData.map(Math.abs));
        if (maxVal === 0) return audioData;

        return audioData.map(sample => sample / maxVal);
    }

    static addNoise(audioData, noiseLevel = 0.001) {
        return audioData.map(sample =>
            sample + (Math.random() - 0.5) * noiseLevel
        );
    }
}

// Mathematical utilities
class MathUtils {
    static complex(real, imag = 0) {
        return { real, imag };
    }

    static complexMultiply(a, b) {
        return {
            real: a.real * b.real - a.imag * b.imag,
            imag: a.real * b.imag + a.imag * b.real
        };
    }

    static complexMagnitude(c) {
        return Math.sqrt(c.real * c.real + c.imag * c.imag);
    }

    static fft(x) {
        const N = x.length;
        if (N <= 1) return x;

        // Ensure power of 2
        const nextPowerOf2 = Math.pow(2, Math.ceil(Math.log2(N)));
        if (N !== nextPowerOf2) {
            const padded = [...x];
            while (padded.length < nextPowerOf2) {
                padded.push(this.complex(0, 0));
            }
            return this.fft(padded);
        }

        // Bit-reversal permutation
        const bitReversed = new Array(N);
        for (let i = 0; i < N; i++) {
            let j = 0;
            let temp = i;
            let bits = Math.log2(N);
            for (let k = 0; k < bits; k++) {
                j = (j << 1) | (temp & 1);
                temp >>= 1;
            }
            bitReversed[j] = Array.isArray(x[i]) ? x[i] : this.complex(x[i], 0);
        }

        // Cooley-Tukey FFT
        for (let size = 2; size <= N; size *= 2) {
            const halfSize = size / 2;
            const wn = this.complex(
                Math.cos(-2 * Math.PI / size),
                Math.sin(-2 * Math.PI / size)
            );

            for (let i = 0; i < N; i += size) {
                let w = this.complex(1, 0);
                for (let j = 0; j < halfSize; j++) {
                    const u = bitReversed[i + j];
                    const v = this.complexMultiply(w, bitReversed[i + j + halfSize]);

                    bitReversed[i + j] = {
                        real: u.real + v.real,
                        imag: u.imag + v.imag
                    };
                    bitReversed[i + j + halfSize] = {
                        real: u.real - v.real,
                        imag: u.imag - v.imag
                    };

                    w = this.complexMultiply(w, wn);
                }
            }
        }

        return bitReversed;
    }

    static melScale(freq) {
        return 2595 * Math.log10(1 + freq / 700);
    }

    static melToFreq(mel) {
        return 700 * (Math.pow(10, mel / 2595) - 1);
    }

    static createMelFilterBank(numFilters, fftSize, sampleRate) {
        const nyquist = sampleRate / 2;
        const minMel = this.melScale(0);
        const maxMel = this.melScale(nyquist);

        const melPoints = [];
        for (let i = 0; i <= numFilters + 1; i++) {
            melPoints.push(minMel + (maxMel - minMel) * i / (numFilters + 1));
        }

        const freqPoints = melPoints.map(mel => this.melToFreq(mel));
        const binPoints = freqPoints.map(freq =>
            Math.floor((fftSize + 1) * freq / sampleRate)
        );

        const filterBank = [];
        for (let i = 1; i <= numFilters; i++) {
            const filter = new Array(Math.floor(fftSize / 2) + 1).fill(0);

            for (let j = binPoints[i - 1]; j < binPoints[i]; j++) {
                if (j < filter.length) {
                    filter[j] = (j - binPoints[i - 1]) / (binPoints[i] - binPoints[i - 1]);
                }
            }

            for (let j = binPoints[i]; j < binPoints[i + 1]; j++) {
                if (j < filter.length) {
                    filter[j] = (binPoints[i + 1] - j) / (binPoints[i + 1] - binPoints[i]);
                }
            }

            filterBank.push(filter);
        }

        return filterBank;
    }
}

// Validation utilities
class ValidationUtils {
    static isValidAudioData(audioData) {
        return Array.isArray(audioData) &&
               audioData.length > 0 &&
               audioData.every(sample => typeof sample === 'number' && !isNaN(sample));
    }

    static isValidUserName(name) {
        return typeof name === 'string' &&
               name.trim().length >= 2 &&
               name.trim().length <= 50 &&
               /^[a-zA-ZáéíóúÁÉÍÓÚñÑ\s]+$/.test(name.trim());
    }

    static isValidWord(word) {
        const allowedWords = ['hola', 'adios', 'si', 'no', 'gracias'];
        return allowedWords.includes(word.toLowerCase());
    }

    static sanitizeFileName(filename) {
        return filename.replace(/[^a-z0-9.-]/gi, '_').toLowerCase();
    }
}

// Export utilities for use in main app
window.AudioUtils = AudioUtils;
window.MathUtils = MathUtils;
window.ValidationUtils = ValidationUtils;