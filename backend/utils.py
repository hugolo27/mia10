import numpy as np
import librosa
from scipy import signal
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioValidator:
    """Utilidades de validación para datos de audio"""

    @staticmethod
    def validate_audio_data(audio_data, min_length=0.5, max_length=10.0, sample_rate=44100):
        """
        Validar datos de audio para procesamiento

        Args:
            audio_data (np.array): Señal de audio
            min_length (float): Duración mínima en segundos
            max_length (float): Duración máxima en segundos
            sample_rate (int): Frecuencia de muestreo

        Returns:
            dict: Resultado de validación con bandera de éxito y mensaje
        """
        try:
            if not isinstance(audio_data, np.ndarray):
                return {'valid': False, 'message': 'Audio data must be numpy array'}

            if len(audio_data) == 0:
                return {'valid': False, 'message': 'Audio data is empty'}

            duration = len(audio_data) / sample_rate

            if duration < min_length:
                return {'valid': False, 'message': f'Audio too short: {duration:.2f}s < {min_length}s'}

            if duration > max_length:
                return {'valid': False, 'message': f'Audio too long: {duration:.2f}s > {max_length}s'}

            # Verificar valores NaN o infinitos
            if np.any(~np.isfinite(audio_data)):
                return {'valid': False, 'message': 'Audio contains NaN or infinite values'}

            # Verificar rango dinámico
            if np.max(np.abs(audio_data)) < 1e-6:
                return {'valid': False, 'message': 'Audio signal too quiet'}

            return {'valid': True, 'message': 'Audio validation passed', 'duration': duration}

        except Exception as e:
            logger.error(f"Error de validación de audio: {str(e)}")
            return {'valid': False, 'message': f'Validation error: {str(e)}'}

class AudioPreprocessor:
    """Utilidades de preprocesamiento de audio"""

    @staticmethod
    def normalize_audio(audio_data, target_rms=0.1):
        """
        Normalizar audio al nivel RMS objetivo

        Args:
            audio_data (np.array): Audio de entrada
            target_rms (float): Nivel RMS objetivo

        Returns:
            np.array: Audio normalizado
        """
        try:
            # Eliminar componente DC
            audio_data = audio_data - np.mean(audio_data)

            # Calcular RMS actual
            current_rms = np.sqrt(np.mean(audio_data ** 2))

            if current_rms > 0:
                # Normalizar al RMS objetivo
                audio_data = audio_data * (target_rms / current_rms)

            # Recortar para prevenir saturación
            audio_data = np.clip(audio_data, -1.0, 1.0)

            return audio_data.astype(np.float32)

        except Exception as e:
            logger.error(f"Error de normalización de audio: {str(e)}")
            return audio_data

    @staticmethod
    def remove_silence(audio_data, sample_rate, threshold_db=-40, min_duration=0.1):
        """
        Eliminar porciones silenciosas del audio

        Args:
            audio_data (np.array): Audio de entrada
            sample_rate (int): Frecuencia de muestreo
            threshold_db (float): Umbral de silencio en dB
            min_duration (float): Duración mínima a mantener

        Returns:
            np.array: Audio con silencio eliminado
        """
        try:
            # Usar detección de actividad vocal de librosa
            intervals = librosa.effects.split(
                audio_data,
                top_db=-threshold_db,
                frame_length=2048,
                hop_length=512
            )

            if len(intervals) == 0:
                return audio_data

            # Concatenar segmentos no silenciosos
            non_silent_audio = []
            for interval in intervals:
                segment = audio_data[interval[0]:interval[1]]
                if len(segment) / sample_rate >= min_duration:
                    non_silent_audio.append(segment)

            if len(non_silent_audio) > 0:
                return np.concatenate(non_silent_audio)
            else:
                return audio_data

        except Exception as e:
            logger.error(f"Error de eliminación de silencio: {str(e)}")
            return audio_data

    @staticmethod
    def apply_pre_emphasis(audio_data, alpha=0.97):
        """
        Aplicar filtro de pre-énfasis para realzar altas frecuencias

        Args:
            audio_data (np.array): Audio de entrada
            alpha (float): Coeficiente de pre-énfasis

        Returns:
            np.array: Audio con pre-énfasis aplicado
        """
        try:
            return np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
        except Exception as e:
            logger.error(f"Error de pre-énfasis: {str(e)}")
            return audio_data

class FeatureUtils:
    """Utilidades para extracción de características y análisis"""

    @staticmethod
    def compute_spectral_features(magnitude_spectrum, frequencies, sample_rate):
        """
        Calcular características espectrales avanzadas

        Args:
            magnitude_spectrum (np.array): Espectro de magnitud
            frequencies (np.array): Bins de frecuencia
            sample_rate (int): Frecuencia de muestreo

        Returns:
            dict: Diccionario de características espectrales
        """
        try:
            features = {}

            # Centroide espectral
            features['spectral_centroid'] = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)

            # Dispersión espectral
            features['spectral_spread'] = np.sqrt(
                np.sum(((frequencies - features['spectral_centroid']) ** 2) * magnitude_spectrum) /
                np.sum(magnitude_spectrum)
            )

            # Asimetría espectral
            mean_freq = features['spectral_centroid']
            spread = features['spectral_spread']
            if spread > 0:
                features['spectral_skewness'] = np.sum(
                    (((frequencies - mean_freq) / spread) ** 3) * magnitude_spectrum
                ) / np.sum(magnitude_spectrum)
            else:
                features['spectral_skewness'] = 0

            # Curtosis espectral
            if spread > 0:
                features['spectral_kurtosis'] = np.sum(
                    (((frequencies - mean_freq) / spread) ** 4) * magnitude_spectrum
                ) / np.sum(magnitude_spectrum) - 3
            else:
                features['spectral_kurtosis'] = 0

            # Rolloff espectral (percentil 85)
            cumsum = np.cumsum(magnitude_spectrum)
            rolloff_threshold = 0.85 * cumsum[-1]
            rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
            features['spectral_rolloff'] = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]

            # Flujo espectral (tasa de cambio en el espectro)
            features['spectral_flux'] = np.sum(np.diff(magnitude_spectrum) ** 2)

            return features

        except Exception as e:
            logger.error(f"Error de características espectrales: {str(e)}")
            return {
                'spectral_centroid': 0,
                'spectral_spread': 0,
                'spectral_skewness': 0,
                'spectral_kurtosis': 0,
                'spectral_rolloff': 0,
                'spectral_flux': 0
            }

    @staticmethod
    def compute_temporal_features(audio_data):
        """
        Calcular características del dominio temporal

        Args:
            audio_data (np.array): Señal de audio de entrada

        Returns:
            dict: Diccionario de características temporales
        """
        try:
            features = {}

            # Tasa de cruces por cero
            features['zcr'] = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))

            # Energía RMS
            features['rms_energy'] = np.sqrt(np.mean(audio_data ** 2))

            # Energía de corto plazo
            frame_length = min(1024, len(audio_data) // 4)
            hop_length = frame_length // 2

            if len(audio_data) >= frame_length:
                frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
                frame_energies = np.sum(frames ** 2, axis=0)

                features['energy_mean'] = np.mean(frame_energies)
                features['energy_std'] = np.std(frame_energies)
                features['energy_max'] = np.max(frame_energies)
            else:
                features['energy_mean'] = features['rms_energy'] ** 2
                features['energy_std'] = 0
                features['energy_max'] = features['rms_energy'] ** 2

            # Características de autocorrelación
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]

            # Encontrar período de pitch (estimación de frecuencia fundamental)
            if len(autocorr) > 1:
                # Buscar picos en la autocorrelación
                peaks, _ = signal.find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))
                if len(peaks) > 0:
                    features['pitch_period'] = peaks[0] + 1
                else:
                    features['pitch_period'] = 0
            else:
                features['pitch_period'] = 0

            return features

        except Exception as e:
            logger.error(f"Error de características temporales: {str(e)}")
            return {
                'zcr': 0,
                'rms_energy': 0,
                'energy_mean': 0,
                'energy_std': 0,
                'energy_max': 0,
                'pitch_period': 0
            }

class DataManager:
    """Utilidades de gestión de datos"""

    @staticmethod
    def save_training_sample(sample_data, filepath):
        """
        Guardar muestra de entrenamiento en archivo JSON

        Args:
            sample_data (dict): Datos de muestra a guardar
            filepath (str): Ruta del archivo de salida

        Returns:
            bool: Bandera de éxito
        """
        try:
            # Convertir arrays numpy a listas para serialización JSON
            serializable_data = DataManager._make_serializable(sample_data)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)

            logger.info(f"Muestra de entrenamiento guardada: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error guardando muestra de entrenamiento: {str(e)}")
            return False

    @staticmethod
    def load_training_sample(filepath):
        """
        Cargar muestra de entrenamiento desde archivo JSON

        Args:
            filepath (str): Ruta del archivo de entrada

        Returns:
            dict: Datos de muestra cargados o None si hay error
        """
        try:
            with open(filepath, 'r') as f:
                sample_data = json.load(f)

            # Convertir listas de vuelta a arrays numpy
            if 'features' in sample_data:
                for key, value in sample_data['features'].items():
                    sample_data['features'][key] = np.array(value)

            return sample_data

        except Exception as e:
            logger.error(f"Error cargando muestra de entrenamiento: {str(e)}")
            return None

    @staticmethod
    def _make_serializable(obj):
        """
        Convertir arrays numpy y otros objetos no serializables a formato compatible con JSON

        Args:
            obj: Objeto a hacer serializable

        Returns:
            Objeto serializable en JSON
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        elif isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: DataManager._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DataManager._make_serializable(item) for item in obj]
        else:
            return obj

class PerformanceMonitor:
    """Utilidades de monitoreo de rendimiento"""

    @staticmethod
    def measure_processing_time(func):
        """
        Decorador para medir tiempo de procesamiento

        Args:
            func: Función a medir

        Returns:
            Función decorada
        """
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()

            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"Tiempo de procesamiento de {func.__name__}: {processing_time:.3f}s")

            return result
        return wrapper

    @staticmethod
    def get_system_stats():
        """
        Obtener estadísticas de rendimiento del sistema

        Returns:
            dict: Estadísticas del sistema
        """
        try:
            import psutil

            stats = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                'timestamp': datetime.now().isoformat()
            }

            return stats

        except ImportError:
            return {
                'message': 'psutil not available',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas del sistema: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }