from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import os
from datetime import datetime
import logging
import time
from functools import wraps

from voice_processor import VoiceProcessor
from models import VoiceClassifier

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_time(operation_name):
    """Decorador para medir tiempo de operaciones"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # en ms

            # Agregar métricas al resultado si es un dict
            if isinstance(result, tuple) and len(result) == 2:
                response, status_code = result
                if hasattr(response, 'json'):
                    data = response.json
                    if isinstance(data, dict):
                        data[f'{operation_name}_tiempo_ms'] = round(processing_time, 2)
                return response, status_code
            elif hasattr(result, 'json'):
                if isinstance(result.json, dict):
                    result.json[f'{operation_name}_tiempo_ms'] = round(processing_time, 2)

            logger.info(f"{operation_name} completado en {processing_time:.2f}ms")
            return result
        return wrapper
    return decorator

app = Flask(__name__)
CORS(app)  # Activar CORS

# Inicializar componentes
voice_processor = VoiceProcessor()
classifier = VoiceClassifier()

# Rutas de almacenamiento de datos
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAINING_DIR = os.path.join(DATA_DIR, 'training_samples')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Rutas del frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

# Asegurar que los directorios existen
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

@app.route('/')
def home():
    """Página principal de la aplicación"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Estáticos (CSS, JS, etc.)"""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route('/train', methods=['POST'])
@measure_time('entrenamiento')
def train_sample():
    """
    Endpoint para recibir muestras de audio de entrenamiento
    JSON esperado: {
        'audio_data': [float],
        'sample_rate': int,
        'user_name': str,
        'word': str,
        'timestamp': str
    }
    """
    try:
        data = request.get_json()

        # Validar
        required_fields = ['audio_data', 'sample_rate', 'user_name', 'word']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Campo requerido no encontrado: {field}'
                }), 400

        audio_data = np.array(data['audio_data'], dtype=np.float32)
        sample_rate = int(data['sample_rate'])
        user_name = data['user_name'].strip()
        word = data['word'].lower().strip()

        # Validar datos de audio
        if len(audio_data) == 0:
            return jsonify({
                'success': False,
                'error': 'Dato de audio sin contenido'
            }), 400

        logger.info(f"Procesando muestra de entrenamiento: user={user_name}, word={word}")

        # Medir tiempo de extracción de características
        start_features = time.time()
        features = voice_processor.extract_features(audio_data, sample_rate)
        tiempo_caracteristicas = (time.time() - start_features) * 1000

        # Guardar muestra de entrenamiento
        sample_filename = f"{user_name}_{word}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        sample_path = os.path.join(TRAINING_DIR, sample_filename)

        sample_data = {
            'user_name': user_name,
            'word': word,
            'features': features,
            'sample_rate': sample_rate,
            'audio_length': len(audio_data),
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        }

        with open(sample_path, 'w') as f:
            json.dump(sample_data, f, cls=NumpyEncoder)

        # Agregar a los datos de entrenamiento del clasificador
        classifier.add_training_sample(features, user_name, word)

        # Entrenar el modelo después de agregar nueva muestra (si tenemos suficientes datos)
        if len(classifier.training_features) >= 2:  # Minimum samples to train
            try:
                train_result = classifier.train()
                logger.info(f"Modelo re-entrenado: {train_result}")
            except Exception as e:
                logger.warning(f"Entrenamiento fallido: {str(e)}")

        # Generar análisis para visualización
        analysis = voice_processor.generate_analysis(audio_data, sample_rate)

        logger.info(f"Muestra de entrenamiento guardada: {sample_filename}")

        return jsonify({
            'success': True,
            'message': f'Muestra de entrenamiento guardada para {user_name} - {word}',
            'sample_id': sample_filename,
            'features_shape': [len(features[key]) for key in features.keys()],
            'analysis': analysis,
            'metricas_rendimiento': {
                'tiempo_caracteristicas_ms': round(tiempo_caracteristicas, 2),
                'audio_duracion_s': round(len(audio_data) / sample_rate, 2),
                'total_caracteristicas': sum(len(features[key]) for key in features.keys())
            }
        })

    except Exception as e:
        logger.error(f"Error al entrenar: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
@measure_time('prediccion')
def predict_voice():
    """
    Endpoint para predecir voz a partir de muestra de audio
    JSON esperado: {
        'audio_data': [float],
        'sample_rate': int,
        'timestamp': str
    }
    """
    try:
        data = request.get_json()

        # Validar entrada
        if 'audio_data' not in data or 'sample_rate' not in data:
            return jsonify({
                'success': False,
                'error': 'audio_data o sample_rate no encontrados'
            }), 400

        audio_data = np.array(data['audio_data'], dtype=np.float32)
        sample_rate = int(data['sample_rate'])

        if len(audio_data) == 0:
            return jsonify({
                'success': False,
                'error': 'Datos de audio vacíos'
            }), 400

        logger.info(f"Procesando solicitud de predicción para longitud de audio: {len(audio_data)}")

        # Extraer características con medición de tiempo
        start_features = time.time()
        features = voice_processor.extract_features(audio_data, sample_rate)
        tiempo_caracteristicas = (time.time() - start_features) * 1000

        # Hacer predicción con medición de tiempo
        start_pred = time.time()
        prediction_result = classifier.predict(features)
        tiempo_prediccion = (time.time() - start_pred) * 1000

        # Generar análisis para visualización
        analysis = voice_processor.generate_analysis(audio_data, sample_rate)

        if prediction_result['success']:
            logger.info(f"Predicción exitosa: {prediction_result['predicted_user']} - {prediction_result['predicted_word']}")
            return jsonify({
                'success': True,
                'predicted_user': prediction_result['predicted_user'],
                'predicted_word': prediction_result['predicted_word'],
                'confidence': prediction_result['confidence'],
                'analysis': analysis,
                'timestamp': datetime.now().isoformat(),
                'metricas_rendimiento': {
                    'tiempo_caracteristicas_ms': round(tiempo_caracteristicas, 2),
                    'tiempo_prediccion_ms': round(tiempo_prediccion, 2),
                    'tiempo_total_ms': round(tiempo_caracteristicas + tiempo_prediccion, 2),
                    'audio_duracion_s': round(len(audio_data) / sample_rate, 2),
                    'velocidad_procesamiento': round((len(audio_data) / sample_rate) / ((tiempo_caracteristicas + tiempo_prediccion) / 1000), 2)
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': prediction_result['message'],
                'analysis': analysis
            })

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Reentrenar el clasificador con todas las muestras de entrenamiento disponibles"""
    try:
        logger.info("Reentrenando clasificador con todas las muestras disponibles")

        # Cargar todas las muestras de entrenamiento
        training_files = [f for f in os.listdir(TRAINING_DIR) if f.endswith('.json')]

        if len(training_files) == 0:
            return jsonify({
                'success': False,
                'error': 'No hay muestras de entrenamiento disponibles'
            }), 400

        # Reiniciar clasificador
        classifier.reset()

        samples_loaded = 0
        for filename in training_files:
            try:
                with open(os.path.join(TRAINING_DIR, filename), 'r') as f:
                    sample_data = json.load(f)

                features = sample_data['features']
                user_name = sample_data['user_name']
                word = sample_data['word']

                # Convertir características de vuelta a arrays numpy
                for key in features:
                    features[key] = np.array(features[key])

                classifier.add_training_sample(features, user_name, word)
                samples_loaded += 1

            except Exception as e:
                logger.warning(f"Falló al cargar muestra de entrenamiento {filename}: {str(e)}")
                continue

        # Entrenar el clasificador
        train_result = classifier.train()

        logger.info(f"Reentrenamiento completado: {samples_loaded} muestras cargadas")

        return jsonify({
            'success': True,
            'message': f'Modelo reentrenado con {samples_loaded} muestras',
            'training_result': train_result
        })

    except Exception as e:
        logger.error(f"Error en endpoint de reentrenamiento: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    """Evaluar modelo con métricas de precisión detalladas"""
    try:
        if not classifier.is_trained:
            return jsonify({
                'success': False,
                'error': 'Modelo no entrenado. Agregue muestras de entrenamiento primero.'
            }), 400

        # Obtener evaluación completa del modelo
        evaluation = classifier.evaluate()

        # Obtener estadísticas de entrenamiento
        training_stats = classifier.get_training_statistics()

        # Calcular métricas adicionales de rendimiento
        metricas_precision = {
            'precision_usuarios': evaluation.get('user_accuracy', 0) * 100,
            'precision_palabras': evaluation.get('word_accuracy', 0) * 100,
            'total_muestras': training_stats['n_samples'],
            'usuarios_entrenados': training_stats['n_users'],
            'palabras_entrenadas': training_stats['n_words'],
            'distribucion_usuarios': training_stats['user_distribution'],
            'distribucion_palabras': training_stats['word_distribution'],
            'dimension_caracteristicas': training_stats['feature_dimension']
        }

        return jsonify({
            'success': True,
            'metricas_precision': metricas_precision,
            'evaluacion_detallada': evaluation,
            'estadisticas_entrenamiento': training_stats,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error en evaluación de modelo: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Obtener estado actual del sistema"""
    try:
        training_files = [f for f in os.listdir(TRAINING_DIR) if f.endswith('.json')]

        # Contar muestras por usuario y palabra
        user_counts = {}
        word_counts = {}

        for filename in training_files:
            try:
                with open(os.path.join(TRAINING_DIR, filename), 'r') as f:
                    sample_data = json.load(f)

                user = sample_data['user_name']
                word = sample_data['word']

                user_counts[user] = user_counts.get(user, 0) + 1
                word_counts[word] = word_counts.get(word, 0) + 1

            except Exception:
                continue

        # Agregar métricas básicas si el modelo está entrenado
        metricas_basicas = {}
        if classifier.is_trained:
            try:
                evaluation = classifier.evaluate()
                metricas_basicas = {
                    'precision_usuarios_pct': round(evaluation.get('user_accuracy', 0) * 100, 1),
                    'precision_palabras_pct': round(evaluation.get('word_accuracy', 0) * 100, 1),
                    'metodo_evaluacion': evaluation.get('evaluation_method', 'unknown'),
                    'modelo_listo': True
                }
            except:
                metricas_basicas = {'modelo_listo': False}

        return jsonify({
            'success': True,
            'total_samples': len(training_files),
            'users': user_counts,
            'words': word_counts,
            'classifier_trained': classifier.is_trained,
            'metricas_basicas': metricas_basicas,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error en endpoint de estado: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

class NumpyEncoder(json.JSONEncoder):
    """Codificador JSON personalizado para arrays numpy"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

if __name__ == '__main__':
    logger.info("Iniciando Servidor de Reconocimiento de Voz...")
    logger.info(f"Directorio de entrenamiento: {TRAINING_DIR}")
    logger.info(f"Directorio de modelos: {MODELS_DIR}")

    # Puerto dinámico para Render o 8000 para desarrollo local
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port)