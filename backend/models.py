import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class VoiceClassifier:
    """
    Sistema de clasificación de voz usando múltiples enfoques data-driven
    Combina características de FFT, MFCC, DMD y SVD para reconocimiento robusto
    """

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.user_encoder = LabelEncoder()
        self.word_encoder = LabelEncoder()

        # Múltiples clasificadores para enfoque de conjunto
        self.user_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        self.word_classifier = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )

        # Almacenamiento de datos de entrenamiento
        self.training_features = []
        self.training_users = []
        self.training_words = []

        self.is_trained = False
        self.feature_importance = None

    def add_training_sample(self, features, user_name, word):
        """
        Agregar una muestra de entrenamiento al conjunto de datos

        Args:
            features (dict): Diccionario de características con claves: fft, mfcc, dmd, svd, stats
            user_name (str): Identificador de usuario
            word (str): Palabra/comando hablado
        """
        try:
            # Aplanar todas las características en un solo vector
            feature_vector = self._flatten_features(features)

            self.training_features.append(feature_vector)
            self.training_users.append(user_name)
            self.training_words.append(word)

            logger.info(f"Muestra de entrenamiento agregada: {user_name} - {word} (características: {len(feature_vector)})")

        except Exception as e:
            logger.error(f"Error agregando muestra de entrenamiento: {str(e)}")
            raise

    def _flatten_features(self, features):
        """Aplanar diccionario de características en un solo vector"""
        feature_vector = []

        # Asegurar orden consistente
        for key in ['fft', 'mfcc', 'dmd', 'svd', 'stats']:
            if key in features:
                feature_array = np.array(features[key])
                feature_vector.extend(feature_array.flatten())
            else:
                # Agregar ceros si falta el tipo de característica
                default_sizes = {'fft': 9, 'mfcc': 26, 'dmd': 10, 'svd': 10, 'stats': 6}
                feature_vector.extend([0] * default_sizes.get(key, 1))

        return np.array(feature_vector, dtype=np.float32)

    def train(self):
        """Entrenar los modelos de clasificación"""
        try:
            if len(self.training_features) == 0:
                raise ValueError("No training samples available")

            X = np.array(self.training_features)
            users = np.array(self.training_users)
            words = np.array(self.training_words)

            logger.info(f"Entrenando con {len(X)} muestras, dimensión de características: {X.shape[1]}")

            # Verificar mínimo de muestras por clase
            unique_users = np.unique(users)
            unique_words = np.unique(words)

            if len(unique_users) < 1 or len(unique_words) < 1:
                raise ValueError("Need at least 1 user and 1 word for training")

            # Normalizar características
            X_scaled = self.feature_scaler.fit_transform(X)

            # Codificar etiquetas
            users_encoded = self.user_encoder.fit_transform(users)
            words_encoded = self.word_encoder.fit_transform(words)

            # Entrenar clasificador de usuarios
            if len(unique_users) > 1:
                self.user_classifier.fit(X_scaled, users_encoded)

                # Calcular puntaje de validación cruzada para clasificación de usuarios
                # Necesita al menos 2 muestras por clase para VC
                min_class_size = min([np.sum(users_encoded == class_id) for class_id in np.unique(users_encoded)])
                if len(X) >= 6 and min_class_size >= 2:  # At least 2 samples per class
                    cv_folds = min(3, min_class_size)
                    user_cv_scores = cross_val_score(
                        self.user_classifier, X_scaled, users_encoded, cv=cv_folds
                    )
                    user_accuracy = np.mean(user_cv_scores)
                else:
                    # Precisión de entrenamiento simple para conjuntos de datos pequeños
                    user_pred = self.user_classifier.predict(X_scaled)
                    user_accuracy = np.mean(user_pred == users_encoded)
            else:
                user_accuracy = 1.0  # Precisión perfecta con usuario único

            # Entrenar clasificador de palabras
            if len(unique_words) > 1:
                self.word_classifier.fit(X_scaled, words_encoded)

                # Calcular puntaje de validación cruzada para clasificación de palabras
                min_word_class_size = min([np.sum(words_encoded == class_id) for class_id in np.unique(words_encoded)])
                if len(X) >= 6 and min_word_class_size >= 2:
                    cv_folds = min(3, min_word_class_size)
                    word_cv_scores = cross_val_score(
                        self.word_classifier, X_scaled, words_encoded, cv=cv_folds
                    )
                    word_accuracy = np.mean(word_cv_scores)
                else:
                    # Precisión de entrenamiento simple para conjuntos de datos pequeños
                    word_pred = self.word_classifier.predict(X_scaled)
                    word_accuracy = np.mean(word_pred == words_encoded)
            else:
                word_accuracy = 1.0  # Precisión perfecta con palabra única

            # Calcular importancia de características
            if hasattr(self.user_classifier, 'feature_importances_'):
                self.feature_importance = self.user_classifier.feature_importances_

            self.is_trained = True

            training_result = {
                'success': True,
                'n_samples': len(X),
                'n_users': len(unique_users),
                'n_words': len(unique_words),
                'feature_dimension': X.shape[1],
                'user_accuracy': float(user_accuracy),
                'word_accuracy': float(word_accuracy),
                'training_timestamp': datetime.now().isoformat()
            }

            logger.info(f"Entrenamiento completado exitosamente: {training_result}")
            return training_result

        except Exception as e:
            logger.error(f"Error durante entrenamiento: {str(e)}")
            self.is_trained = False
            raise

    def predict(self, features):
        """
        Predecir usuario y palabra a partir de características

        Args:
            features (dict): Diccionario de características

        Returns:
            dict: Resultados de predicción con puntajes de confianza
        """
        try:
            if not self.is_trained:
                return {
                    'success': False,
                    'message': 'Model not trained yet. Please add training samples first.'
                }

            # Aplanar características
            feature_vector = self._flatten_features(features).reshape(1, -1)

            # Escalar características
            feature_vector_scaled = self.feature_scaler.transform(feature_vector)

            # Predecir usuario
            user_pred = None
            user_confidence = 0.0

            if len(self.user_encoder.classes_) > 1:
                user_pred_encoded = self.user_classifier.predict(feature_vector_scaled)[0]
                user_pred = self.user_encoder.inverse_transform([user_pred_encoded])[0]

                # Obtener probabilidades de predicción para confianza
                if hasattr(self.user_classifier, 'predict_proba'):
                    user_proba = self.user_classifier.predict_proba(feature_vector_scaled)[0]
                    user_confidence = np.max(user_proba)
                else:
                    user_confidence = 1.0
            else:
                # Solo un usuario en los datos de entrenamiento
                user_pred = self.user_encoder.classes_[0]
                user_confidence = 1.0

            # Predecir palabra
            word_pred = None
            word_confidence = 0.0

            if len(self.word_encoder.classes_) > 1:
                word_pred_encoded = self.word_classifier.predict(feature_vector_scaled)[0]
                word_pred = self.word_encoder.inverse_transform([word_pred_encoded])[0]

                # Obtener probabilidades de predicción para confianza
                word_proba = self.word_classifier.predict_proba(feature_vector_scaled)[0]
                word_confidence = np.max(word_proba)
            else:
                # Solo una palabra en los datos de entrenamiento
                word_pred = self.word_encoder.classes_[0]
                word_confidence = 1.0

            # Confianza combinada (media geométrica)
            combined_confidence = np.sqrt(user_confidence * word_confidence)

            # Establecer umbral de confianza mínimo
            min_confidence = 0.3
            if combined_confidence < min_confidence:
                return {
                    'success': False,
                    'message': f'Low confidence prediction ({combined_confidence:.2f}). Need more training data.',
                    'predicted_user': user_pred,
                    'predicted_word': word_pred,
                    'confidence': combined_confidence
                }

            return {
                'success': True,
                'predicted_user': user_pred,
                'predicted_word': word_pred,
                'confidence': combined_confidence,
                'user_confidence': user_confidence,
                'word_confidence': word_confidence
            }

        except Exception as e:
            logger.error(f"Error durante predicción: {str(e)}")
            return {
                'success': False,
                'message': f'Prediction error: {str(e)}'
            }

    def evaluate(self):
        """Evaluar el modelo entrenado usando validación cruzada"""
        try:
            if not self.is_trained or len(self.training_features) == 0:
                return {'error': 'No trained model or data available'}

            X = np.array(self.training_features)
            X_scaled = self.feature_scaler.transform(X)
            users = np.array(self.training_users)
            words = np.array(self.training_words)

            users_encoded = self.user_encoder.transform(users)
            words_encoded = self.word_encoder.transform(words)

            evaluation = {}

            # Evaluación de clasificación de usuarios con validación cruzada
            if len(np.unique(users)) > 1:
                # Verificar si tenemos suficientes muestras para validación cruzada
                min_samples_per_class = min([np.sum(users_encoded == class_id) for class_id in np.unique(users_encoded)])

                if len(X) >= 6 and min_samples_per_class >= 2:
                    # Validación cruzada estratificada
                    cv_folds = min(3, min_samples_per_class)
                    user_cv_scores = cross_val_score(
                        self.user_classifier, X_scaled, users_encoded,
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                        scoring='accuracy'
                    )
                    evaluation['user_accuracy'] = np.mean(user_cv_scores)
                    evaluation['user_accuracy_std'] = np.std(user_cv_scores)
                    evaluation['evaluation_method'] = 'cross_validation'
                else:
                    # Para datasets pequeños, usar train accuracy con penalización
                    user_predictions = self.user_classifier.predict(X_scaled)
                    train_accuracy = np.mean(user_predictions == users_encoded)
                    # Penalizar por overfitting en datasets pequeños
                    penalty = 0.1 * (1 - min(len(X) / 10, 1))
                    evaluation['user_accuracy'] = max(0, train_accuracy - penalty)
                    evaluation['evaluation_method'] = 'train_penalized'
            else:
                evaluation['user_accuracy'] = 1.0
                evaluation['evaluation_method'] = 'single_user'

            # Evaluación de clasificación de palabras con validación cruzada
            if len(np.unique(words)) > 1:
                min_word_samples_per_class = min([np.sum(words_encoded == class_id) for class_id in np.unique(words_encoded)])

                if len(X) >= 6 and min_word_samples_per_class >= 2:
                    # Validación cruzada estratificada
                    cv_folds = min(3, min_word_samples_per_class)
                    word_cv_scores = cross_val_score(
                        self.word_classifier, X_scaled, words_encoded,
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                        scoring='accuracy'
                    )
                    evaluation['word_accuracy'] = np.mean(word_cv_scores)
                    evaluation['word_accuracy_std'] = np.std(word_cv_scores)
                else:
                    # Para datasets pequeños, usar train accuracy con penalización
                    word_predictions = self.word_classifier.predict(X_scaled)
                    train_accuracy = np.mean(word_predictions == words_encoded)
                    # Penalizar por overfitting en datasets pequeños
                    penalty = 0.1 * (1 - min(len(X) / 10, 1))
                    evaluation['word_accuracy'] = max(0, train_accuracy - penalty)
            else:
                evaluation['word_accuracy'] = 1.0

            # Información adicional sobre la evaluación
            evaluation['n_samples'] = len(X)
            evaluation['n_users'] = len(np.unique(users))
            evaluation['n_words'] = len(np.unique(words))

            # Importancia de características
            if self.feature_importance is not None:
                evaluation['feature_importance'] = self.feature_importance.tolist()

            return evaluation

        except Exception as e:
            logger.error(f"Error durante evaluación: {str(e)}")
            return {'error': str(e)}

    def save_model(self, filepath):
        """Guardar el modelo entrenado en disco"""
        try:
            model_data = {
                'feature_scaler': self.feature_scaler,
                'user_encoder': self.user_encoder,
                'word_encoder': self.word_encoder,
                'user_classifier': self.user_classifier,
                'word_classifier': self.word_classifier,
                'training_features': self.training_features,
                'training_users': self.training_users,
                'training_words': self.training_words,
                'is_trained': self.is_trained,
                'feature_importance': self.feature_importance,
                'save_timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Modelo guardado en {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            return False

    def load_model(self, filepath):
        """Cargar un modelo entrenado desde disco"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.feature_scaler = model_data['feature_scaler']
            self.user_encoder = model_data['user_encoder']
            self.word_encoder = model_data['word_encoder']
            self.user_classifier = model_data['user_classifier']
            self.word_classifier = model_data['word_classifier']
            self.training_features = model_data['training_features']
            self.training_users = model_data['training_users']
            self.training_words = model_data['training_words']
            self.is_trained = model_data['is_trained']
            self.feature_importance = model_data.get('feature_importance')

            logger.info(f"Modelo cargado desde {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            return False

    def reset(self):
        """Reiniciar el clasificador al estado no entrenado"""
        self.training_features = []
        self.training_users = []
        self.training_words = []
        self.is_trained = False
        self.feature_importance = None

        # Reiniciar escaladores y codificadores
        self.feature_scaler = StandardScaler()
        self.user_encoder = LabelEncoder()
        self.word_encoder = LabelEncoder()

        logger.info("Clasificador reiniciado al estado no entrenado")

    def get_training_statistics(self):
        """Obtener estadísticas sobre los datos de entrenamiento"""
        if len(self.training_features) == 0:
            return {'n_samples': 0, 'users': [], 'words': []}

        users = np.array(self.training_users)
        words = np.array(self.training_words)

        # Contar muestras por usuario y palabra
        unique_users, user_counts = np.unique(users, return_counts=True)
        unique_words, word_counts = np.unique(words, return_counts=True)

        user_stats = dict(zip(unique_users, user_counts.tolist()))
        word_stats = dict(zip(unique_words, word_counts.tolist()))

        return {
            'n_samples': len(self.training_features),
            'n_users': len(unique_users),
            'n_words': len(unique_words),
            'user_distribution': user_stats,
            'word_distribution': word_stats,
            'feature_dimension': len(self.training_features[0]) if self.training_features else 0
        }