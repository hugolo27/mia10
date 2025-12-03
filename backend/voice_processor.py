import numpy as np
import librosa
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import io
import base64
import logging

logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Clase de procesamiento de voz que implementa métodos data-driven:
    - FFT para análisis de frecuencia
    - MFCC para extracción de características
    - DMD (Descomposición Modal Dinámica) para patrones temporales
    - SVD para reducción de dimensionalidad
    """

    def __init__(self):
        self.mfcc_n_coeffs = 13
        self.fft_size = 2048
        self.hop_length = 512
        self.n_mels = 26

    def extract_features(self, audio_data, sample_rate):
        """
        Extraer características completas del audio usando métodos data-driven

        Args:
            audio_data (np.array): Señal de audio
            sample_rate (int): Frecuencia de muestreo

        Returns:
            dict: Diccionario con todas las características extraídas
        """
        try:
            # Asegurar que el audio esté normalizado
            audio_data = self._normalize_audio(audio_data)

            # 1. Características FFT
            fft_features = self._extract_fft_features(audio_data, sample_rate)

            # 2. Características MFCC
            mfcc_features = self._extract_mfcc_features(audio_data, sample_rate)

            # 3. Características DMD (dinámica temporal)
            dmd_features = self._extract_dmd_features(audio_data, sample_rate)

            # 4. Características SVD (descomposición espectral)
            svd_features = self._extract_svd_features(audio_data, sample_rate)

            # 5. Características Estadísticas
            stats_features = self._extract_statistical_features(audio_data)

            features = {
                'fft': fft_features,
                'mfcc': mfcc_features,
                'dmd': dmd_features,
                'svd': svd_features,
                'stats': stats_features
            }

            logger.info(f"Extracción de características completada. Formas: FFT={len(fft_features)}, "
                       f"MFCC={len(mfcc_features)}, DMD={len(dmd_features)}, SVD={len(svd_features)}")

            return features

        except Exception as e:
            logger.error(f"Error en extracción de características: {str(e)}")
            raise

    def _normalize_audio(self, audio_data):
        """Normalizar señal de audio"""
        if len(audio_data) == 0:
            return audio_data

        # Eliminar componente DC
        audio_data = audio_data - np.mean(audio_data)

        # Normalizar a [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        return audio_data

    def _extract_fft_features(self, audio_data, sample_rate):
        """Extraer características de frecuencia basadas en FFT"""
        try:
            # Aplicar función ventana
            windowed = audio_data * signal.windows.hamming(len(audio_data))

            # Calcular FFT
            fft_result = fft(windowed, n=self.fft_size)
            magnitude_spectrum = np.abs(fft_result[:self.fft_size//2])

            # Extraer características clave
            features = []

            # Centroide espectral
            freqs = fftfreq(self.fft_size, 1/sample_rate)[:self.fft_size//2]
            spectral_centroid = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
            features.append(spectral_centroid)

            # Rolloff espectral
            cumulative_sum = np.cumsum(magnitude_spectrum)
            rolloff_point = 0.85 * cumulative_sum[-1]
            rolloff_idx = np.where(cumulative_sum >= rolloff_point)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            features.append(spectral_rolloff)

            # Ancho de banda espectral
            mean_freq = spectral_centroid
            spectral_bandwidth = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * magnitude_spectrum) / np.sum(magnitude_spectrum))
            features.append(spectral_bandwidth)

            # Tasa de cruces por cero
            zcr = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
            features.append(zcr)

            # Frecuencias dominantes (top 5)
            peak_indices = signal.find_peaks(magnitude_spectrum, height=np.max(magnitude_spectrum) * 0.1)[0]
            peak_freqs = freqs[peak_indices]
            peak_mags = magnitude_spectrum[peak_indices]

            # Ordenar por magnitud y tomar los 5 primeros
            sorted_indices = np.argsort(peak_mags)[-5:][::-1]
            top_freqs = peak_freqs[sorted_indices] if len(sorted_indices) > 0 else [0, 0, 0, 0, 0]

            # Rellenar para asegurar exactamente 5 características
            while len(top_freqs) < 5:
                top_freqs = np.append(top_freqs, 0)

            features.extend(top_freqs[:5])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error en extracción de características FFT: {str(e)}")
            # Devolver características por defecto en caso de error
            return np.zeros(9, dtype=np.float32)

    def _extract_mfcc_features(self, audio_data, sample_rate):
        """Extraer características MFCC usando librosa"""
        try:
            # Calcular MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.mfcc_n_coeffs,
                n_fft=self.fft_size,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )

            # Tomar media y desviación estándar a través de la dimensión temporal
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            # Combinar características de media y desviación estándar
            features = np.concatenate([mfcc_mean, mfcc_std])

            return features.astype(np.float32)

        except Exception as e:
            logger.error(f"Error en extracción de características MFCC: {str(e)}")
            return np.zeros(self.mfcc_n_coeffs * 2, dtype=np.float32)

    def _extract_dmd_features(self, audio_data, sample_rate):
        """Extraer características basadas en DMD para dinámicas temporales"""
        try:
            # Crear embeddings de retardo temporal (enfoque de matriz Hankel)
            frame_size = min(1024, len(audio_data) // 4)
            hop_size = frame_size // 2

            if len(audio_data) < frame_size * 2:
                # Audio muy corto para DMD
                return np.zeros(10, dtype=np.float32)

            # Crear matriz de datos
            num_frames = (len(audio_data) - frame_size) // hop_size + 1
            X = np.zeros((frame_size, num_frames))

            for i in range(num_frames):
                start_idx = i * hop_size
                end_idx = start_idx + frame_size
                X[:, i] = audio_data[start_idx:end_idx]

            if X.shape[1] < 2:
                return np.zeros(10, dtype=np.float32)

            # Algoritmo DMD (versión simplificada)
            X1 = X[:, :-1]  # First n-1 snapshots
            X2 = X[:, 1:]   # Last n-1 snapshots

            # SVD de X1
            U, S, Vt = np.linalg.svd(X1, full_matrices=False)

            # Truncar para evitar problemas numéricos
            r = min(10, len(S), X1.shape[1] - 1)
            U_r = U[:, :r]
            S_r = S[:r]
            V_r = Vt[:r, :].T

            # Construir matriz Atilde
            Atilde = U_r.T @ X2 @ V_r @ np.diag(1/S_r)

            # Descomposición en autovalores
            eigenvals, eigenvecs = np.linalg.eig(Atilde)

            # Extraer características de los autovalores
            features = []

            # Contenido de frecuencia dominante
            angles = np.angle(eigenvals)
            frequencies = angles * sample_rate / (2 * np.pi * hop_size)

            # Tasas de crecimiento
            growth_rates = np.log(np.abs(eigenvals)) * sample_rate / hop_size

            # Tomar los 5 modos superiores por magnitud
            magnitudes = np.abs(eigenvals)
            sorted_indices = np.argsort(magnitudes)[-5:][::-1]

            top_freqs = frequencies[sorted_indices]
            top_growth = growth_rates[sorted_indices]

            # Asegurar exactamente 5 características cada una
            while len(top_freqs) < 5:
                top_freqs = np.append(top_freqs, 0)
            while len(top_growth) < 5:
                top_growth = np.append(top_growth, 0)

            features = np.concatenate([top_freqs[:5], top_growth[:5]])

            return features.astype(np.float32)

        except Exception as e:
            logger.error(f"Error en extracción de características DMD: {str(e)}")
            return np.zeros(10, dtype=np.float32)

    def _extract_svd_features(self, audio_data, sample_rate):
        """Extraer características basadas en SVD del espectrograma"""
        try:
            # Calcular espectrograma
            f, t, Sxx = signal.spectrogram(
                audio_data,
                fs=sample_rate,
                window='hamming',
                nperseg=min(1024, len(audio_data)//4),
                noverlap=None
            )

            # Tomar magnitud logarítmica
            log_spec = np.log(Sxx + 1e-10)

            # Descomposición SVD
            U, S, Vt = np.linalg.svd(log_spec, full_matrices=False)

            # Extraer características de los valores singulares
            features = []

            # Distribución de energía en valores singulares
            total_energy = np.sum(S**2)
            if total_energy > 0:
                energy_ratios = (S**2) / total_energy

                # Tomar las primeras 8 razones de energía
                features.extend(energy_ratios[:8] if len(energy_ratios) >= 8 else
                              np.pad(energy_ratios, (0, 8-len(energy_ratios))))

                # Complejidad espectral (entropía de la distribución de energía)
                entropy = -np.sum(energy_ratios * np.log(energy_ratios + 1e-10))
                features.append(entropy)

                # Rango efectivo
                eff_rank = np.sum(energy_ratios) ** 2 / np.sum(energy_ratios ** 2)
                features.append(eff_rank)
            else:
                features.extend([0] * 10)

            return np.array(features[:10], dtype=np.float32)

        except Exception as e:
            logger.error(f"Error en extracción de características SVD: {str(e)}")
            return np.zeros(10, dtype=np.float32)

    def _extract_statistical_features(self, audio_data):
        """Extraer características estadísticas básicas"""
        try:
            features = []

            # Estadísticas del dominio temporal
            features.append(np.mean(audio_data))
            features.append(np.std(audio_data))
            features.append(np.max(np.abs(audio_data)))
            features.append(np.sqrt(np.mean(audio_data**2)))  # RMS

            # Momentos de orden superior
            features.append(self._skewness(audio_data))
            features.append(self._kurtosis(audio_data))

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error en extracción de características estadísticas: {str(e)}")
            return np.zeros(6, dtype=np.float32)

    def _skewness(self, data):
        """Calcular asimetría"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _kurtosis(self, data):
        """Calcular curtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def generate_analysis(self, audio_data, sample_rate):
        """Generar datos de análisis para visualización"""
        try:
            analysis = {}

            # FFT para visualización
            fft_result = fft(audio_data, n=self.fft_size)
            magnitude_spectrum = np.abs(fft_result[:self.fft_size//2])
            freqs = fftfreq(self.fft_size, 1/sample_rate)[:self.fft_size//2]

            # Submuestrear para frontend
            step = max(1, len(magnitude_spectrum) // 200)
            analysis['fft_data'] = magnitude_spectrum[::step].tolist()

            # MFCC para visualización
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.mfcc_n_coeffs
            )
            analysis['mfcc_data'] = np.mean(mfccs, axis=1).tolist()

            # Modos DMD simplificados para visualización
            try:
                dmd_features = self._extract_dmd_features(audio_data, sample_rate)
                analysis['dmd_data'] = {
                    'modes': dmd_features[:5].tolist(),
                    'frequencies': dmd_features[5:].tolist()
                }
            except:
                analysis['dmd_data'] = {
                    'modes': [0] * 5,
                    'frequencies': [0] * 5
                }

            # Espectrograma para visualización
            try:
                f, t, Sxx = signal.spectrogram(
                    audio_data,
                    fs=sample_rate,
                    nperseg=min(512, len(audio_data)//8)
                )
                # Submuestrear espectrograma
                spec_step_f = max(1, len(f) // 20)
                spec_step_t = max(1, len(t) // 50)

                spec_data = Sxx[::spec_step_f, ::spec_step_t]
                spec_normalized = (spec_data - np.min(spec_data)) / (np.max(spec_data) - np.min(spec_data) + 1e-10)

                analysis['spectrogram_data'] = spec_normalized.tolist()
            except:
                analysis['spectrogram_data'] = [[0]]

            return analysis

        except Exception as e:
            logger.error(f"Error generando análisis: {str(e)}")
            return {
                'fft_data': [0] * 100,
                'mfcc_data': [0] * self.mfcc_n_coeffs,
                'dmd_data': {'modes': [0] * 5, 'frequencies': [0] * 5},
                'spectrogram_data': [[0]]
            }